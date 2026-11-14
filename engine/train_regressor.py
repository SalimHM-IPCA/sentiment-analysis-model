"""Frozen local DistilBERT embeddings + Ridge for EmoBank V/A/D.
Run from project root: python -u engine/train_regressor.py
"""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

ENGINE = Path(__file__).resolve().parent


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_data(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"id", "split", "V", "A", "D", "text"} <= set(reader.fieldnames or []):
            raise ValueError("CSV must contain id, split, V, A, D, text")
        rows = list(reader)
    ids = set()
    groups = {s: [] for s in ("train", "dev", "test")}
    for row in rows:
        if row["id"] in ids:
            raise ValueError("Duplicate id: " + row["id"])
        ids.add(row["id"])
        if row["split"] not in groups or not row["text"].strip():
            raise ValueError("Invalid split or empty text: " + row["id"])
        values = np.array([float(row["V"]), float(row["A"]), float(row["D"])])
        if not np.all(np.isfinite(values)) or np.any((values < 1) | (values > 5)):
            raise ValueError("V/A/D must be finite and in [1,5]: " + row["id"])
        groups[row["split"]].append(row)
    # Keep within-split repetitions; remove exact text overlap across splits.
    seen = set()
    dropped = []
    clean = {}
    for split in ("train", "dev", "test"):
        clean[split] = []
        for row in groups[split]:
            if row["text"].strip() in seen:
                dropped.append({"id": row["id"], "split": split,
                                "reason": "exact stripped text occurs in earlier split"})
            else:
                clean[split].append(row)
        seen.update(r["text"].strip() for r in groups[split])
        if not clean[split]:
            raise ValueError("Empty split: " + split)
    return clean, {"original_counts": {s: len(v) for s, v in groups.items()},
                   "used_counts": {s: len(v) for s, v in clean.items()},
                   "dropped_cross_split_duplicates": dropped,
                   "policy": "train > dev > test; strip-only exact text matching; retain within-split repetitions"}


def targets(rows):
    return (np.array([[float(r["V"]), float(r["A"]), float(r["D"])] for r in rows]) - 3.0) / 2.0


def metrics(actual, predicted):
    result = {}
    for i, name in enumerate(("valence", "activity", "dominance")):
        a, p = actual[:, i], predicted[:, i]
        correlation = None
        if len(a) > 1 and np.std(a) > 1e-12 and np.std(p) > 1e-12:
            correlation = float(np.corrcoef(a, p)[0, 1])
        result[name] = {"mae": float(mean_absolute_error(a, p)), "pearson": correlation}
    result["mean_mae"] = float(mean_absolute_error(actual, predicted))
    return result


OS_JUNK = {".DS_Store", "Thumbs.db", "desktop.ini"}


def model_fingerprint(path):
    # Posix paths sorted as text and skipped OS junk files give the same hash on Windows and macOS.
    files = sorted(
        (p.relative_to(path).as_posix(), p) for p in path.rglob("*")
        if p.is_file() and p.name not in OS_JUNK and not p.name.startswith("._")
        and "__MACOSX" not in p.parts
    )
    digest = hashlib.sha256()
    for name, file in files:
        digest.update(name.encode())
        with file.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def embeddings(model, rows, split, cache_root, fingerprint):
    # Save every 128 rows. An interrupted run can reuse completed chunks.
    text_list = [r["text"] for r in rows]
    signature = hashlib.sha256(json.dumps({"texts": text_list,
        "model": fingerprint, "max_seq_length": 128, "normalise": False},
        ensure_ascii=False).encode()).hexdigest()[:24]
    folder = cache_root / split / signature
    folder.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(rows), 128):
        texts = text_list[start:start + 128]
        file = folder / f"{start:06d}.npy"
        if file.exists():
            chunk = np.load(file, allow_pickle=False)
            if chunk.shape != (len(texts), 768) or not np.isfinite(chunk).all():
                raise ValueError(f"Invalid cache: {file}; remove this file and rerun")
        else:
            chunk = model.encode(texts, batch_size=16, convert_to_numpy=True,
                                 normalize_embeddings=False, show_progress_bar=False)
            if chunk.shape != (len(texts), 768) or not np.isfinite(chunk).all():
                raise ValueError("Expected finite DistilBERT embeddings with 768 dimensions")
            temp = file.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.save(f, chunk, allow_pickle=False)
            temp.replace(file)
        chunks.append(chunk)
        print(f"{split}: {min(start + 128, len(rows))}/{len(rows)} texts ready", flush=True)
    return np.concatenate(chunks)


def main():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    data_path = ENGINE / "data" / "text" / "emobank.csv"
    model_path = ENGINE / "models" / "text" / "encoder"
    if not data_path.is_file() or not model_path.is_dir():
        raise FileNotFoundError("Need engine/data/text/emobank.csv and engine/models/text/encoder")
    output = ENGINE / "results" / "text"
    output.mkdir(parents=True, exist_ok=True)
    cache = ENGINE / "data" / "text" / "embedding_cache"
    rows, audit = load_data(data_path)
    save_json(output / "data_audit.json", audit)
    print("Split counts:", audit["used_counts"], flush=True)
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    print("Loading saved local encoder (no encoder fine-tuning)...", flush=True)
    model = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
    model.max_seq_length = 128
    fingerprint = model_fingerprint(model_path)
    x = {s: embeddings(model, rows[s], s, cache, fingerprint)
         for s in ("train", "dev")}
    y = {s: targets(rows[s]) for s in rows}
    candidates = []
    best, best_mae = None, float("inf")
    for alpha in (0.1, 1.0, 10.0, 100.0):
        regressor = Ridge(alpha=alpha, solver="svd")
        regressor.fit(x["train"], y["train"])
        result = metrics(y["dev"], np.clip(regressor.predict(x["dev"]), -1, 1))
        candidates.append({"alpha": alpha, "dev": result})
        print(f"alpha={alpha}: dev mean MAE={result['mean_mae']:.4f}", flush=True)
        if result["mean_mae"] < best_mae:
            best, best_mae = regressor, result["mean_mae"]
    # Evaluate the held-out test set only after alpha selection.
    test_x = embeddings(model, rows["test"], "test", cache, fingerprint)
    prediction = np.clip(best.predict(test_x), -1, 1)
    baseline = np.tile(y["train"].mean(axis=0), (len(y["test"]), 1))
    metadata = {"encoder": "distilbert/distilbert-base-uncased + saved pooling",
        "model_sha256": fingerprint, "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "targets": ["valence", "activity", "dominance"], "normalisation": "(EmoBank score - 3) / 2",
        "max_seq_length": 128, "normalize_embeddings": False,
        "selected_alpha": float(best.alpha), "prediction_clip": [-1, 1],
        "training": "frozen encoder; Ridge fit on train only; alpha chosen using dev mean MAE",
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "scikit-learn", "sentence-transformers", "torch", "joblib")}}
    bundle = {"regressor": best, "metadata": metadata}
    joblib.dump(bundle, ENGINE / "models" / "text" / "regressor.joblib")
    report = {"metadata": metadata, "data_audit": audit, "alpha_search": candidates,
              "test": metrics(y["test"], prediction),
              "test_train_mean_baseline": metrics(y["test"], baseline),
              "note": "Metrics use [-1,1] targets and clipped predictions. Pearson null means undefined. Test must not be used to tune alpha."}
    save_json(output / "metrics.json", report)
    with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "true_valence", "true_activity", "true_dominance",
                         "pred_valence", "pred_activity", "pred_dominance"])
        for row, actual, pred in zip(rows["test"], y["test"], prediction):
            writer.writerow([row["id"], row["text"], *actual, *pred])
    print("\nTest results:", json.dumps(report["test"], indent=2), flush=True)
    print("Mean baseline:", json.dumps(report["test_train_mean_baseline"], indent=2), flush=True)
    print("Training complete. Model in models/text/, reports in", output, flush=True)


if __name__ == "__main__":
    main()
