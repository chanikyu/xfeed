from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="xfeed",
    version="1.0.0",
    author="Kyu-Chan Lee",
    description="Predicting microbial cross-feeding flux from shotgun species abundance (neural, KEGG reaction-based)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chanikyu/xfeed",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "xfeed=scripts.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    include_package_data=True,
    package_data={
        "": ["model/*.pt"],
    },
)
