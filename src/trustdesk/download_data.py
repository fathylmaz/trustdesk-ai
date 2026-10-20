"""Download the original licensed BANKING77 CSV files with SHA-256 integrity checks."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

FILES = {
    "train.csv": "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b",
    "test.csv": "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d",
}
BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"


def download_data(target_dir: str | Path = "data/raw") -> None:
    folder = Path(target_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in FILES.items():
        destination = folder / name
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_hash:
            print(f"[OK] Existing verified file: {destination}")
            continue
        print(f"[DOWNLOAD] {name}")
        with urlopen(f"{BASE_URL}/{name}", timeout=45) as response:
            content = response.read()
        actual_hash = hashlib.sha256(content).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {name}. Do not use downloaded file. Expected {expected_hash}, got {actual_hash}")
        temporary = destination.with_suffix(".csv.tmp")
        temporary.write_bytes(content)
        temporary.replace(destination)
        print(f"[OK] Saved verified file: {destination}")


def main():
    parser = argparse.ArgumentParser(description="Download validated BANKING77 dataset")
    parser.add_argument("--target-dir", default="data/raw")
    args = parser.parse_args()
    download_data(args.target_dir)


if __name__ == "__main__":
    main()
