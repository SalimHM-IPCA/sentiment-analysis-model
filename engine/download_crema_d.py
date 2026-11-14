"""Download the CREMA-D audio clips listed in data/voice/manifest.json and verify their SHA-256.
Run from project root: python engine/download_crema_d.py [--split train|dev|test]

Only needed to recompute acoustic features or rerun voice experiments; the engine itself
does not use these files. Already downloaded files that pass the hash check are skipped.
CREMA-D: https://github.com/CheyneyComputerScience/CREMA-D (Open Database License).
"""
import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
DATA = ENGINE / "data" / "voice"
BASE_URL = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV/"
# Same audio (same SHA-256) under a different name in the GitHub copy of the dataset.
REMOTE_NAME = {"1040_ITH_SAD_X.wav": "1040_ITH_SAD_XX.wav"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def fetch(row, target, attempts=3):
    path = target / row["file"]
    if path.is_file() and sha256(path.read_bytes()) == row["sha256"]:
        return "skipped", None
    error = None
    for _ in range(attempts):
        try:
            url = BASE_URL + REMOTE_NAME.get(row["file"], row["file"])
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if sha256(data) != row["sha256"]:
                raise ValueError("hash mismatch")
            path.write_bytes(data)
            return "downloaded", None
        except (OSError, ValueError) as exc:
            error = exc
    return "failed", f"{row['file']}: {error}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    args = parser.parse_args()
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in manifest["rows"] if args.split in (None, r["split"])]
    target = DATA / "audio"
    target.mkdir(exist_ok=True)
    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i, (status, error) in enumerate(pool.map(lambda r: fetch(r, target), rows), 1):
            counts[status] += 1
            if error:
                failures.append(error)
            if i % 500 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} clips processed", flush=True)
    print(f"done: {counts['downloaded']} downloaded, {counts['skipped']} already present, {counts['failed']} failed")
    for failure in failures:
        print("  failed:", failure)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
