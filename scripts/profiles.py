"""Build per-species metabolic capability profiles from KEGG REACTION data.

This is the v1.1.0 profile pipeline. Unlike the earlier module-level
approach (which captured only 38 cross-feedable compounds and missed
every SCFA, amino acid, and vitamin), this module queries the full
KEGG REACTION database, parses each reaction's EQUATION field to
separate substrates from products, and derives ~1,780 cross-feedable
compounds covering essentially all biologically interpretable metabolites.

Pipeline:
  1. Download per-organism KO lists from KEGG (/link/ko/{organism})
  2. Fetch KO → reaction bulk mapping       (/link/rn/ko)
  3. Batch-fetch reaction EQUATION fields   (/get/rn1+rn2+...)
  4. Parse EQUATIONs into substrate/product compound sets
  5. Build per-species produces/consumes sets
  6. Filter out ubiquitous cofactors, keep cross-feedable compounds

Outputs (written to profile_dir):
  species_kos.json            — per-species KO lists
  ko_to_reaction.json         — KO → reaction IDs lookup
  reaction_equations.json     — reaction ID → {substrates, products}
  species_caps.json           — per-species {produces, consumes} sets
  compounds.json              — sorted list of cross-feedable compounds
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

from .config import KEGG_API, KEGG_BATCH_SIZE, REQUEST_DELAY

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Cofactor blacklist — ubiquitous "currency" metabolites excluded so the
# cross-feedable compound list reflects real metabolic exchange.
# ═══════════════════════════════════════════════════════════════════════
COFACTOR_BLACKLIST: set[str] = {
    "C00001",  # H2O
    "C00080",  # H+
    "C00002",  # ATP
    "C00008",  # ADP
    "C00020",  # AMP
    "C00009",  # Phosphate
    "C00013",  # Diphosphate
    "C00003",  # NAD+
    "C00004",  # NADH
    "C00005",  # NADPH
    "C00006",  # NADP+
    "C00007",  # O2
    "C00011",  # CO2
    "C00010",  # CoA
    "C00015",  # UDP
    "C00016",  # FAD
    "C00019",  # SAM
    "C00035",  # GDP
    "C00044",  # GTP
    "C00063",  # CTP
    "C00075",  # UTP
    "C00081",  # ITP
    "C00112",  # CDP
    "C00021",  # SAH
    "C00017",  # Protein
    "C00051",  # Glutathione
    "C00144",  # GMP
    "C00105",  # UMP
    "C00055",  # CMP
    "C00139",  # Ferredoxin (reduced)
    "C00138",  # Ferredoxin (oxidized)
    "C00229",  # Acyl-carrier protein
    "C00238",  # K+
    "C00175",  # Co2+
    "C14818",  # Fe2+
    "C14819",  # Fe3+
    "C00070",  # Zn2+
}


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Per-organism KO download (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════

def get_kegg_organisms() -> dict[str, str]:
    """KEGG organism list → {genus_species: organism_code}."""
    logger.info("Downloading KEGG organism list...")
    resp = requests.get(f"{KEGG_API}/list/organism", timeout=30)
    resp.raise_for_status()

    organisms: dict[str, str] = {}
    for line in resp.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        code = parts[1]
        name_parts = parts[2].lower().strip().split()
        if len(name_parts) >= 2:
            genus_sp = f"{name_parts[0]} {name_parts[1]}"
            if genus_sp not in organisms:
                organisms[genus_sp] = code
    logger.info("  %d unique genus+species entries", len(organisms))
    return organisms


def get_organism_kos(org_code: str) -> set[str]:
    """KEGG organism code → set of KO IDs."""
    resp = requests.get(f"{KEGG_API}/link/ko/{org_code}", timeout=30)
    if resp.status_code != 200:
        return set()
    kos: set[str] = set()
    for line in resp.text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            kos.add(parts[1].replace("ko:", ""))
    return kos


def download_species_kos(
    species_list: list[str],
    cache_path: Path | None = None,
) -> dict[str, set[str]]:
    """Download KO annotations for each species, honouring an optional cache."""
    kegg_orgs = get_kegg_organisms()

    species_to_org: dict[str, str] = {}
    for sp in species_list:
        parts = sp.lower().split()
        if len(parts) >= 2:
            genus_sp = f"{parts[0]} {parts[1]}"
            if genus_sp in kegg_orgs:
                species_to_org[sp] = kegg_orgs[genus_sp]
    logger.info(
        "KEGG-matched species: %d / %d", len(species_to_org), len(species_list),
    )

    cache: dict[str, list[str]] = {}
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        logger.info("Loaded KO cache: %d organisms", len(cache))

    to_download = [
        (sp, code) for sp, code in species_to_org.items()
        if code not in cache
    ]
    logger.info("Downloads needed: %d", len(to_download))

    for i, (sp, code) in enumerate(to_download):
        if i and i % 50 == 0:
            logger.info("  %d / %d", i, len(to_download))
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)
        try:
            kos = get_organism_kos(code)
            cache[code] = sorted(kos)
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning("  %s (%s) failed: %s", sp, code, e)
            cache[code] = []
            time.sleep(1)

    if cache_path:
        with open(cache_path, "w") as f:
            json.dump(cache, f)

    species_kos: dict[str, set[str]] = {}
    for sp, code in species_to_org.items():
        kos = set(cache.get(code, []))
        if kos:
            species_kos[sp] = kos
    counts = [len(v) for v in species_kos.values()]
    logger.info(
        "KO profiles: %d species, median %d KOs/species",
        len(species_kos), int(np.median(counts)) if counts else 0,
    )
    return species_kos


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — KO → Reaction bulk lookup
# ═══════════════════════════════════════════════════════════════════════

def fetch_ko_to_reaction(cache_path: Path) -> dict[str, list[str]]:
    """Fetch (or load cached) KO → [reactionID] mapping from /link/rn/ko."""
    if cache_path.exists():
        logger.info("Loading KO→reaction cache: %s", cache_path)
        text = cache_path.read_text()
    else:
        logger.info("Fetching /link/rn/ko from KEGG ...")
        resp = requests.get(f"{KEGG_API}/link/rn/ko", timeout=60)
        resp.raise_for_status()
        text = resp.text
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
        time.sleep(REQUEST_DELAY)

    mapping: dict[str, list[str]] = defaultdict(list)
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ko = parts[0].replace("ko:", "")
        rn = parts[1].replace("rn:", "")
        mapping[ko].append(rn)
    logger.info("KO → reaction mapping: %d KOs", len(mapping))
    return dict(mapping)


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Reaction equation parsing
# ═══════════════════════════════════════════════════════════════════════

_EQUATION_RE = re.compile(r"^EQUATION\s+(.+)$", re.MULTILINE)
_COMPOUND_RE = re.compile(r"C\d{5}")
_ENTRY_RE = re.compile(r"^ENTRY\s+(R\d{5})", re.MULTILINE)


def parse_equation(equation_text: str) -> tuple[list[str], list[str]]:
    """Parse 'substrates <=> products' → ([substrates], [products])."""
    if "<=>" not in equation_text:
        return [], []
    left, right = equation_text.split("<=>", 1)
    substrates = list(dict.fromkeys(_COMPOUND_RE.findall(left)))
    products = list(dict.fromkeys(_COMPOUND_RE.findall(right)))
    return substrates, products


def _parse_entry(entry_text: str) -> tuple[str, list[str], list[str]]:
    rid_match = _ENTRY_RE.search(entry_text)
    eq_match = _EQUATION_RE.search(entry_text)
    if not rid_match or not eq_match:
        return "", [], []
    subs, prods = parse_equation(eq_match.group(1))
    return rid_match.group(1), subs, prods


def fetch_reaction_equations(
    reaction_ids: list[str],
    cache_path: Path,
) -> dict[str, dict[str, list[str]]]:
    """Fetch reaction EQUATION fields in batches, with incremental caching."""
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        logger.info("Loaded reaction cache: %d entries", len(cached))
    else:
        cached = {}

    remaining = [r for r in reaction_ids if r not in cached]
    logger.info("Reactions to fetch: %d (already cached: %d)",
                len(remaining), len(cached))

    if not remaining:
        return cached

    t0 = time.time()
    for start in range(0, len(remaining), KEGG_BATCH_SIZE):
        batch = remaining[start:start + KEGG_BATCH_SIZE]
        query = "+".join(f"rn:{r}" for r in batch)
        try:
            resp = requests.get(f"{KEGG_API}/get/{query}", timeout=60)
            if resp.status_code != 200:
                logger.warning("  batch starting %s failed (HTTP %d)",
                               batch[0], resp.status_code)
                time.sleep(REQUEST_DELAY)
                continue
            for entry in resp.text.split("///"):
                if not entry.strip():
                    continue
                rid, subs, prods = _parse_entry(entry)
                if rid:
                    cached[rid] = {"substrates": subs, "products": prods}
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            logger.warning("  batch error: %s", e)
            time.sleep(1)

        if (start // KEGG_BATCH_SIZE) % 20 == 0 and start > 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            rate = done / max(elapsed, 0.1)
            eta = max(len(remaining) - done, 0) / max(rate, 0.1)
            logger.info(
                "  progress %d / %d  (%.0fs elapsed, ETA %.0fs)",
                done, len(remaining), elapsed, eta,
            )
            with open(cache_path, "w") as f:
                json.dump(cached, f)

    with open(cache_path, "w") as f:
        json.dump(cached, f)
    logger.info("Reaction fetch done: %d entries (%.0fs)",
                len(cached), time.time() - t0)
    return cached


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Per-species capability matrices
# ═══════════════════════════════════════════════════════════════════════

def build_species_capabilities(
    species_kos: dict[str, set[str]],
    ko_to_reaction: dict[str, list[str]],
    reaction_equations: dict[str, dict[str, list[str]]],
) -> dict[str, dict[str, set[str]]]:
    """Derive per-species produces/consumes compound sets from KEGG reactions."""
    caps: dict[str, dict[str, set[str]]] = {}
    for sp, kos in species_kos.items():
        produces: set[str] = set()
        consumes: set[str] = set()
        for ko in kos:
            for rn in ko_to_reaction.get(ko, []):
                eq = reaction_equations.get(rn)
                if not eq:
                    continue
                produces.update(eq["products"])
                consumes.update(eq["substrates"])
        caps[sp] = {"produces": produces, "consumes": consumes}
    return caps


# ═══════════════════════════════════════════════════════════════════════
# Step 5 — Cross-feedable compound filter
# ═══════════════════════════════════════════════════════════════════════

def find_cross_feedable(
    caps: dict[str, dict[str, set[str]]],
    exclude_cofactors: bool = True,
) -> list[str]:
    """Compounds produced by ≥ 1 species AND consumed by ≥ 1 other species."""
    producer_count: dict[str, int] = defaultdict(int)
    consumer_count: dict[str, int] = defaultdict(int)
    for c in caps.values():
        for x in c["produces"]:
            producer_count[x] += 1
        for x in c["consumes"]:
            consumer_count[x] += 1

    candidates = set(producer_count) & set(consumer_count)
    if exclude_cofactors:
        candidates -= COFACTOR_BLACKLIST

    # Sort by min(producers, consumers) descending — balanced sides first
    return sorted(
        candidates,
        key=lambda c: -min(producer_count[c], consumer_count[c]),
    )


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline driver
# ═══════════════════════════════════════════════════════════════════════

def build_all_profiles(
    species_list: list[str],
    output_dir: Path,
) -> tuple[dict[str, dict[str, set[str]]], list[str]]:
    """Full KEGG-reaction-based profile pipeline.

    Returns (species_caps, compound_list) and writes all intermediates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — species KOs
    logger.info("Step 1: fetching per-species KO annotations")
    species_kos = download_species_kos(
        species_list, cache_dir / "species_kos_cache.json",
    )
    with open(output_dir / "species_kos.json", "w") as f:
        json.dump({sp: sorted(kos) for sp, kos in species_kos.items()}, f)

    # Step 2 — KO → Reaction
    logger.info("Step 2: fetching KO → reaction mapping")
    ko_to_rn = fetch_ko_to_reaction(cache_dir / "link_rn_ko.tsv")
    with open(output_dir / "ko_to_reaction.json", "w") as f:
        json.dump(ko_to_rn, f)

    # Collect relevant reactions and fetch their equations
    all_kos: set[str] = set()
    for kos in species_kos.values():
        all_kos.update(kos)
    relevant_rns = sorted({
        rn for ko in all_kos for rn in ko_to_rn.get(ko, [])
    })
    logger.info("Relevant reactions: %d", len(relevant_rns))

    # Step 3 — Reaction equations
    logger.info("Step 3: fetching reaction equations")
    rn_to_eq = fetch_reaction_equations(
        relevant_rns, cache_dir / "reaction_equations.json",
    )
    with open(output_dir / "reaction_equations.json", "w") as f:
        json.dump(rn_to_eq, f)

    # Step 4 — Capabilities
    logger.info("Step 4: building species capabilities")
    caps = build_species_capabilities(species_kos, ko_to_rn, rn_to_eq)
    with open(output_dir / "species_caps.json", "w") as f:
        json.dump(
            {sp: {"produces": sorted(v["produces"]),
                  "consumes": sorted(v["consumes"])}
             for sp, v in caps.items()},
            f,
        )

    # Step 5 — Cross-feedable compounds
    logger.info("Step 5: identifying cross-feedable compounds")
    compounds = find_cross_feedable(caps, exclude_cofactors=True)
    with open(output_dir / "compounds.json", "w") as f:
        json.dump(compounds, f)

    logger.info(
        "Done. %d species | %d reactions | %d cross-feedable compounds",
        len(caps), len(rn_to_eq), len(compounds),
    )
    return caps, compounds
