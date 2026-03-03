from __future__ import annotations

from pathlib import Path


def ensure_file(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(content, encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parent

    dirs = [
        "data/raw",
        "data/processed",
        "data/external",
        "notebooks",
        "src/plbias",
        "src/plbias/ingest",
        "src/plbias/clean",
        "src/plbias/odds",
        "src/plbias/analysis",
        "src/plbias/viz",
        "scripts",
        "app",
        "reports",
        "tests",
    ]
    for d in dirs:
        ensure_dir(root / d)

    gitkeeps = [
        "data/raw/.gitkeep",
        "data/processed/.gitkeep",
        "data/external/.gitkeep",
        "reports/.gitkeep",
    ]
    for gk in gitkeeps:
        ensure_file(root / gk, "")

    files: dict[str, str] = {
        "src/plbias/__init__.py": "",
        "src/plbias/config.py": "LEAGUE = 'EPL'\nDIVISION_CODE = 'E0'\n",
        "src/plbias/ingest/__init__.py": "",
        "src/plbias/ingest/football_data.py": "",
        "src/plbias/clean/__init__.py": "",
        "src/plbias/clean/standardize.py": "",
        "src/plbias/odds/__init__.py": "",
        "src/plbias/odds/implied.py": "",
        "src/plbias/odds/devig.py": "",
        "src/plbias/analysis/__init__.py": "",
        "src/plbias/analysis/calibration.py": "",
        "src/plbias/analysis/bias.py": "",
        "src/plbias/analysis/metrics.py": "",
        "src/plbias/viz/__init__.py": "",
        "src/plbias/viz/plots.py": "",
        "scripts/01_fetch_football_data.py": "",
        "scripts/02_build_matches_table.py": "",
        "scripts/03_run_calibration_report.py": "",
        "app/streamlit_app.py": "",
        "tests/test_implied.py": "",
        "tests/test_devig.py": "",
    }

    for rel, content in files.items():
        ensure_file(root / rel, content)

    default_gitignore = (
        "# venv\n"
        ".venv/\n"
        "venv/\n"
        "env/\n\n"
        "# python\n"
        "__pycache__/\n"
        "*.pyc\n"
        ".pytest_cache/\n\n"
        "# notebooks\n"
        ".ipynb_checkpoints/\n\n"
        "# data (keep folders via .gitkeep)\n"
        "data/raw/*\n"
        "data/processed/*\n"
        "data/external/*\n"
        "!data/raw/.gitkeep\n"
        "!data/processed/.gitkeep\n"
        "!data/external/.gitkeep\n\n"
        "# OS\n"
        ".DS_Store\n"
        "Thumbs.db\n"
    )
    ensure_file(root / ".gitignore", default_gitignore)

    print("Project structure ensured (no overwrites).")


if __name__ == "__main__":
    main()