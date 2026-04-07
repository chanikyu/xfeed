#!/usr/bin/env python3
"""xFeed command-line interface.

Three subcommands:

    xfeed setup   — download pre-built profiles + pretrained model
    xfeed train   — optionally retrain FluxMLP on a user-provided abundance table
    xfeed predict — run a trained FluxMLP on a new abundance table

Usage:
    xfeed setup
    xfeed predict --abundance samples.tsv
    xfeed train   --abundance data.tsv --output-dir xfeed_model/
"""
from __future__ import annotations

import argparse
import logging
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import __version__
from .config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    RELEASE_ASSET,
    RELEASE_URL,
)
from .data import FluxDataset, collate_fn, load_abundance, load_profiles
from .model import FluxMLP, FluxMLPConfig, load_checkpoint, save_checkpoint
from .visualize import generate_visualizations

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
)


# ═══════════════════════════════════════════════════════════════════════
# Device selection
# ═══════════════════════════════════════════════════════════════════════

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


# ═══════════════════════════════════════════════════════════════════════
# xfeed setup — download pre-built data to ~/.xfeed/
# ═══════════════════════════════════════════════════════════════════════

def cmd_setup(args: argparse.Namespace) -> None:
    data_dir = Path(args.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if (data_dir / "xfeed_model.pt").exists() and (data_dir / "compounds.json").exists():
        print(f"xFeed data already exists at: {data_dir}")
        print("  To re-download, delete the directory and run setup again.")
        return

    print(f"Downloading xFeed data to: {data_dir}")
    archive_path = data_dir / RELEASE_ASSET

    resp = requests.get(RELEASE_URL, stream=True, timeout=60)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(archive_path, "wb") as f:
        with tqdm(total=total, unit="B", unit_scale=True, desc="  Downloading") as pbar:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print("  Extracting...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    archive_path.unlink()

    print(f"\nDone. xFeed data saved to: {data_dir}")
    print(f"  Profiles: {data_dir}/species_caps.json, compounds.json")
    print(f"  Model:    {data_dir}/xfeed_model.pt")
    print("\nNow run: xfeed predict --abundance your_data.tsv")


# ═══════════════════════════════════════════════════════════════════════
# xfeed train — retrain FluxMLP on user data
# ═══════════════════════════════════════════════════════════════════════

def _per_column_pearson(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-column Pearson correlation, skipping constant columns."""
    vals: list[float] = []
    for j in range(pred.shape[1]):
        p = pred[:, j]
        t = true[:, j]
        if p.std() < 1e-8 or t.std() < 1e-8:
            continue
        r = float(np.corrcoef(p, t)[0, 1])
        if not np.isnan(r):
            vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def _train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="  train", unit="batch", leave=False)
    for batch in pbar:
        x = batch["features"].to(device)
        y = batch["flow_log"].to(device)
        pred = model(x)
        loss = model.compute_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total += float(loss.item())
        n += 1
        pbar.set_postfix(loss=f"{total/n:.4f}")
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, device) -> dict:
    model.eval()
    preds, trues = [], []
    total, n = 0.0, 0
    for batch in loader:
        x = batch["features"].to(device)
        y = batch["flow_log"].to(device)
        pred = model(x)
        loss = model.compute_loss(pred, y)
        total += float(loss.item())
        n += 1
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0)
    true_arr = np.concatenate(trues, axis=0)
    return {
        "loss": total / max(n, 1),
        "mean_pearson": _per_column_pearson(pred_arr, true_arr),
    }


def cmd_train(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  xFeed v{__version__} Training  (device: {device})")
    print("=" * 60)

    profile_dir = Path(args.profile_dir)
    species_caps, compound_list = load_profiles(profile_dir)

    abd = load_abundance(Path(args.abundance))

    rng = np.random.default_rng(args.seed)
    all_samples = abd.index.tolist()
    rng.shuffle(all_samples)
    split = int(len(all_samples) * 0.8)
    train_ids, val_ids = all_samples[:split], all_samples[split:]
    logger.info("Train: %d samples  |  Val: %d samples",
                len(train_ids), len(val_ids))

    species_list = list(abd.columns)
    train_ds = FluxDataset(
        abundance=abd.loc[train_ids],
        species_caps=species_caps,
        compound_list=compound_list,
        species_list=species_list,
    )
    val_ds = FluxDataset(
        abundance=abd.loc[val_ids],
        species_caps=species_caps,
        compound_list=compound_list,
        species_list=species_list,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
    )

    cfg = FluxMLPConfig(
        n_species=train_ds.input_dim,
        n_compounds=train_ds.output_dim,
    )
    model = FluxMLP(cfg).to(device)
    logger.info("FluxMLP parameters: %d", model.n_parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=DEFAULT_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    history = []
    best_loss = float("inf")
    best_state = None
    best_metrics: dict = {}
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = _evaluate(model, val_loader, device)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_pearson": round(val_metrics["mean_pearson"], 6),
        }
        history.append(row)
        logger.info(
            "Epoch %3d/%d  loss=%.4f  val_loss=%.4f  val_pearson=%.4f",
            epoch, args.epochs, train_loss,
            val_metrics["loss"], val_metrics["mean_pearson"],
        )

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = val_metrics
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping at epoch %d (no val improvement)", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    ckpt_path = output_dir / "xfeed_model.pt"
    save_checkpoint(
        model=model,
        path=str(ckpt_path),
        species_list=species_list,
        compound_list=compound_list,
        metrics=best_metrics,
    )
    logger.info("Saved checkpoint: %s", ckpt_path)

    pd.DataFrame(history).to_csv(
        output_dir / "training_history.tsv", sep="\t", index=False,
    )
    elapsed = time.time() - t0
    print(f"\nDone. Best val loss = {best_loss:.4f}, {elapsed:.0f}s")


# ═══════════════════════════════════════════════════════════════════════
# xfeed predict — run FluxMLP on new abundance data
# ═══════════════════════════════════════════════════════════════════════

def cmd_predict(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    t0 = time.time()
    print("=" * 60)
    print(f"  xFeed v{__version__} Predict  (device: {device})")
    print("=" * 60)

    model, meta = load_checkpoint(args.checkpoint, device)
    species_list = meta["species_list"]
    compound_list = meta["compound_list"]
    logger.info(
        "Checkpoint loaded: %d species × %d compounds",
        len(species_list), len(compound_list),
    )

    abd = load_abundance(Path(args.abundance))
    logger.info("Input: %d samples × %d species columns",
                abd.shape[0], abd.shape[1])

    abd_aligned = abd.reindex(columns=species_list).fillna(0.0)
    features = np.log1p(abd_aligned.values.astype(np.float32))

    all_preds = []
    with torch.no_grad():
        for start in range(0, len(features), 256):
            batch = torch.as_tensor(
                features[start:start + 256], dtype=torch.float32,
            ).to(device)
            pred = model(batch).cpu().numpy()
            all_preds.append(pred)
    pred_log = np.concatenate(all_preds, axis=0)
    pred_raw = np.expm1(pred_log)

    # ── Try to load species_caps (needed for the species network figure
    #    and, optionally, to let downstream users interpret the output).
    species_caps: dict | None = None
    species_caps_path = Path(args.profile_dir) / "species_caps.json"
    if species_caps_path.exists():
        try:
            _, _ = load_profiles(Path(args.profile_dir))  # sanity-check files
            import json as _json
            with open(species_caps_path) as f:
                raw_caps = _json.load(f)
            species_caps = {
                sp: {
                    "produces": set(v.get("produces", [])),
                    "consumes": set(v.get("consumes", [])),
                }
                for sp, v in raw_caps.items()
            }
            logger.info(
                "Loaded species_caps for %d species from %s",
                len(species_caps), species_caps_path,
            )
        except Exception as e:
            logger.warning("Could not load species_caps: %s", e)
            species_caps = None
    else:
        logger.info(
            "species_caps.json not found at %s — species network figure "
            "will be skipped", species_caps_path,
        )

    # ── Resolve compound names once (KEGG API, with on-disk cache) ──
    name_cache = Path(args.checkpoint).parent / "compound_names_cache.json"
    from .visualize import fetch_compound_names
    compound_names = fetch_compound_names(compound_list, cache_path=name_cache)

    # ── Build the long-format TSV with compound_name column ──
    sample_ids = list(abd.index)
    records = []
    threshold = float(args.min_flux)
    for i, sid in enumerate(sample_ids):
        idxs = np.where(pred_raw[i] >= threshold)[0]
        for j in idxs:
            cid = compound_list[j]
            records.append({
                "sample_id": sid,
                "compound": cid,
                "compound_name": compound_names.get(cid, cid),
                "flux_log": round(float(pred_log[i, j]), 4),
                "flux_raw": round(float(pred_raw[i, j]), 2),
            })

    out_path = Path(args.output)
    if records:
        df = pd.DataFrame(records)
        df.to_csv(out_path, sep="\t", index=False)
        n_samples_with_output = df["sample_id"].nunique()
        mean_rows = len(df) / max(n_samples_with_output, 1)
        print(f"\nSaved: {out_path}")
        print(f"  {len(df):,} rows, {n_samples_with_output} samples "
              f"(avg {mean_rows:.0f} compounds above {threshold} / sample)")
        print("  Top 10 compounds by mean predicted flux:")
        top = (
            df.groupby(["compound", "compound_name"])["flux_raw"]
            .mean().sort_values(ascending=False).head(10)
        )
        for (cid, cname), mean_flux in top.items():
            print(f"    {cid}  {cname[:30]}: {mean_flux:,.0f}")
    else:
        pd.DataFrame(
            columns=["sample_id", "compound", "compound_name",
                     "flux_log", "flux_raw"],
        ).to_csv(out_path, sep="\t", index=False)
        print(f"\nNo compounds above the threshold {threshold}. Empty TSV saved.")

    # ── Save dense NPZ with names included ──
    dense_path = out_path.with_suffix(".npz")
    np.savez_compressed(
        dense_path,
        sample_ids=np.array(sample_ids),
        compound_list=np.array(compound_list),
        compound_names=np.array([compound_names.get(c, c) for c in compound_list]),
        flux_log=pred_log.astype(np.float32),
        flux_raw=pred_raw.astype(np.float32),
    )
    print(f"  Dense matrix: {dense_path}")

    # ── Auto-generate visualizations (unless explicitly disabled) ──
    if not args.no_visualize:
        print(f"\nGenerating visualizations (top {args.top_n} compounds)...")
        try:
            raw_abundance_matrix = abd_aligned.values.astype(np.float32)
            figs, _ = generate_visualizations(
                flux_log=pred_log,
                flux_raw=pred_raw,
                sample_ids=sample_ids,
                compound_list=compound_list,
                output_dir=out_path.parent,
                top_n=args.top_n,
                name_cache=name_cache,
                abundance=raw_abundance_matrix,
                species_list=species_list,
                species_caps=species_caps,
                image_format=args.image_format,
                flux_top_species=args.flux_top_species,
            )
            print(f"  Saved {len(figs)} figures to {out_path.parent / 'images'}/")
        except Exception as e:
            logger.warning("Visualization step failed: %s", e)

    print(f"\nElapsed: {time.time() - t0:.1f}s")


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="xfeed",
        description="xFeed: predict microbial cross-feeding flux from shotgun species abundance",
    )
    parser.add_argument(
        "--version", action="version", version=f"xFeed {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    subparsers.required = True

    # ---- setup -------------------------------------------------------
    sp_setup = subparsers.add_parser(
        "setup",
        help="Download pre-built profiles and pretrained model",
        description=(
            "Downloads the pre-built cross-feedable compound list, species "
            "capability profiles, and the pretrained FluxMLP checkpoint "
            "from GitHub Releases. Run this once before 'xfeed predict'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp_setup.add_argument(
        "--output-dir", default=str(DEFAULT_DATA_DIR),
        help=f"Directory for profiles and model (default: {DEFAULT_DATA_DIR})",
    )
    sp_setup.set_defaults(func=cmd_setup)

    # ---- train -------------------------------------------------------
    sp_train = subparsers.add_parser(
        "train",
        help="Retrain FluxMLP on your own abundance table",
        description=(
            "Rebuilds flux labels on the fly from the provided species "
            "capability profiles and trains a new FluxMLP checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp_train.add_argument("--abundance", required=True,
                          help="Species abundance table (TSV / CSV)")
    sp_train.add_argument("--profile-dir", default=str(DEFAULT_DATA_DIR),
                          help=f"Profile directory (default: {DEFAULT_DATA_DIR})")
    sp_train.add_argument("--output-dir", default="xfeed_model",
                          help="Where to save the checkpoint & history")
    sp_train.add_argument("--device", default="auto",
                          choices=["auto", "cpu", "cuda", "mps"])
    sp_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    sp_train.add_argument("--lr", type=float, default=DEFAULT_LR)
    sp_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    sp_train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    sp_train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp_train.set_defaults(func=cmd_train)

    # ---- predict -----------------------------------------------------
    sp_predict = subparsers.add_parser(
        "predict",
        help="Predict cross-feeding flux for new samples",
        description=(
            "Applies a trained FluxMLP to a new species abundance table "
            "and writes a sparse long-format TSV of predicted compound "
            "flux values plus a dense .npz matrix for downstream analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp_predict.add_argument("--abundance", required=True,
                            help="Species abundance table (TSV / CSV)")
    sp_predict.add_argument(
        "--checkpoint", default=str(DEFAULT_DATA_DIR / "xfeed_model.pt"),
        help=f"Path to FluxMLP checkpoint (default: {DEFAULT_DATA_DIR / 'xfeed_model.pt'})",
    )
    sp_predict.add_argument(
        "--profile-dir", default=str(DEFAULT_DATA_DIR),
        help=(
            f"Profile directory from 'xfeed setup' "
            f"(needed for the species network figure; default: {DEFAULT_DATA_DIR})"
        ),
    )
    sp_predict.add_argument("--output", default="xfeed_predictions.tsv",
                            help="Output TSV path (dense .npz will use same stem)")
    sp_predict.add_argument(
        "--min-flux", type=float, default=1.0,
        help="Minimum predicted raw flux to include in the long-format TSV (default: 1.0)",
    )
    sp_predict.add_argument(
        "--top-n", type=int, default=20,
        help="Number of top compounds to show in visualizations (default: 20)",
    )
    sp_predict.add_argument(
        "--no-visualize", action="store_true",
        help="Skip auto-generating the publication-ready figures",
    )
    sp_predict.add_argument(
        "--image-format", default="png", choices=["png", "pdf"],
        help=(
            "File format for all generated figures — 'png' for web/slides "
            "(default) or 'pdf' for vector figures suitable for manuscripts."
        ),
    )
    sp_predict.add_argument(
        "--flux-top-species", type=int, default=10,
        help=(
            "Number of most-abundant species to draw per-species flux "
            "diagrams for, in images/flux/. Set to 0 to skip this step. "
            "(default: 10)"
        ),
    )
    sp_predict.add_argument("--device", default="auto",
                            choices=["auto", "cpu", "cuda", "mps"])
    sp_predict.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
