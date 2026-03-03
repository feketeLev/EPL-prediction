from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

folders = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src/plbet",
    "app",
    "scripts",
    "tests"
]

files = {
    "src/plbet/__init__.py": "",
    "src/plbet/config.py": "",
    "src/plbet/io.py": "",
    "src/plbet/features.py": "",
    "src/plbet/models.py": "",
    "src/plbet/backtest.py": "",
    "app/streamlit_app.py": "",
    "scripts/fetch_data.py": "",
    "scripts/train.py": "",
    "scripts/backtest.py": ""
}

for folder in folders:
    (BASE_DIR / folder).mkdir(parents=True, exist_ok=True)

for file_path, content in files.items():
    file_full_path = BASE_DIR / file_path
    file_full_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_full_path.exists():
        file_full_path.write_text(content)

print("Project structure created successfully.")