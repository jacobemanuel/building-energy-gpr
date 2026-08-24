from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

DATA_URL = "https://archive.ics.uci.edu/static/public/242/data.csv"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "ENB2012_data.csv"

    if output_path.exists():
        print(f"Data already exists at {output_path}")
        return

    print(f"Downloading dataset to {output_path} ...")
    urlretrieve(DATA_URL, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
