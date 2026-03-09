from pathlib import Path
import argparse

STRUCTURE = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/{pkg}",
    "tests",
    "scripts",
    "results",
    "reports",
    "models",
    "artifacts",
]

PYPROJECT = """\
[build-system]
requires = ["hatchling>=1.25.0"]
build-backend = "hatchling.build"

[project]
name = "{name}"
version = "0.1.0"
description = "Odds-based value betting + ML bet selection"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.26",
  "pandas>=2.1",
  "pyarrow>=14.0",
  "scikit-learn>=1.4",
  "xgboost>=2.0",
  "matplotlib>=3.8",
  "tqdm>=4.66",
  "typer>=0.12"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "ruff>=0.6",
  "black>=24.0",
  "ipykernel>=6.29"
]
app = ["streamlit>=1.34"]

[project.scripts]
{name} = "{pkg}.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/{pkg}"]

[tool.ruff]
line-length = 100

[tool.black]
line-length = 100
"""

REQUIREMENTS = """\
numpy>=1.26
pandas>=2.1
pyarrow>=14.0
scikit-learn>=1.4
xgboost>=2.0
matplotlib>=3.8
tqdm>=4.66
typer>=0.12

pytest>=8.0
ruff>=0.6
black>=24.0
ipykernel>=6.29
streamlit>=1.34
"""

GITIGNORE = """\
__pycache__/
*.py[cod]
.venv/
venv/
.env
.ipynb_checkpoints/

dist/
build/
*.egg-info/

.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

.vscode/
.idea/
.DS_Store
Thumbs.db

data/raw/
data/processed/
results/
reports/
models/
artifacts/
logs/

*.parquet
*.csv
*.pkl
*.joblib
"""

README = """\
# Betting project

Scaffold for:
- market efficiency checks
- value betting baseline
- ML bet selection
- backtests

## Setup
python -m venv .venv
pip install -r requirements.txt
"""

CLI = """\
import typer

app = typer.Typer(no_args_is_help=True)

@app.command()
def hello():
    typer.echo("OK")

if __name__ == "__main__":
    app()
"""

INIT = "__all__ = []\n"

FEATURES = """\
import numpy as np
import pandas as pd

def implied_prob(odds: pd.Series) -> pd.Series:
    return 1.0 / odds

def normalize_probs(p: pd.DataFrame) -> pd.DataFrame:
    return p.div(p.sum(axis=1), axis=0)

def consensus_prob_mean_odds(odds: pd.Series) -> float:
    return 1.0 / float(np.mean(odds))

def expected_value(p: float, odds: float) -> float:
    return p * odds - 1.0
"""

def write(path: Path, content: str, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def mkdirs(root: Path, pkg: str) -> None:
    for item in STRUCTURE:
        (root / item.format(pkg=pkg)).mkdir(parents=True, exist_ok=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="betproj", help="project name (pyproject [project].name and script name)")
    ap.add_argument("--pkg", default="betproj", help="python package name under src/")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    args = ap.parse_args()

    root = Path(".").resolve()
    name = args.name.strip()
    pkg = args.pkg.strip().replace("-", "_")

    mkdirs(root, pkg)

    write(root / "pyproject.toml", PYPROJECT.format(name=name, pkg=pkg), args.force)
    write(root / "requirements.txt", REQUIREMENTS, args.force)
    write(root / ".gitignore", GITIGNORE, args.force)
    write(root / "README.md", README, args.force)

    write(root / f"src/{pkg}/__init__.py", INIT, args.force)
    write(root / f"src/{pkg}/cli.py", CLI, args.force)
    write(root / f"src/{pkg}/features.py", FEATURES, args.force)

    (root / "tests/__init__.py").touch(exist_ok=True)

    print(f"Scaffold created. name={name} pkg={pkg}")

if __name__ == "__main__":
    main()