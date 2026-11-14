"""Compute acoustic_features.analyse() for every CREMA-D clip and cache the result as CSV.
Run from project root after download_crema_d.py: python engine/extract_acoustic_features.py
"""
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE.parent))
from engine import acoustic_features as af  # noqa: E402

DATA = ENGINE / "data" / "voice"
SCALARS = ("duration_s", "loudness_db", "loudness_sd_db", "pitch_median_hz", "pitch_sd_semitones",
           "pitch_range_semitones", "voiced_ratio", "flatness", "energy_range_db")


def extract(file_name):
    features = af.analyse(af.load_audio(DATA / "audio" / file_name))
    row = {k: round(features[k], 6) for k in SCALARS}
    row.update({f"mfcc{i + 2}_mean": round(float(v), 6) for i, v in enumerate(features["mfcc_mean"])})
    row.update({f"mfcc{i + 2}_sd": round(float(v), 6) for i, v in enumerate(features["mfcc_sd"])})
    return row


def main():
    rows = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["rows"]
    with ProcessPoolExecutor() as pool:
        features = list(pool.map(extract, [r["file"] for r in rows], chunksize=64))
    out = DATA / "acoustic_features.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "actor", "split", "label", *features[0]])
        writer.writeheader()
        for r, feats in zip(rows, features):
            writer.writerow({"file": r["file"], "actor": r["actor"], "split": r["split"], "label": r["label"], **feats})
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
