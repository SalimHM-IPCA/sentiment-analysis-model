"""Train the CREMA-D voice emotion classifier on cached frozen emotion2vec embeddings.
Run from project root: python engine/train_voice_classifier.py

Uses the actor-independent split in data/voice/manifest.json: C is chosen on dev actors only,
test actors are evaluated once. Writes models/voice/voice_classifier.joblib and results/voice/.
"""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENGINE = Path(__file__).resolve().parent
DATA = ENGINE / "data" / "voice"
MODEL_DIR = ENGINE / "models" / "voice"
RESULTS = ENGINE / "results" / "voice"
C_GRID = (0.01, 0.1, 1.0, 10.0)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split(manifest, split):
    rows = [r for r in manifest["rows"] if r["split"] == split]
    x = np.stack([np.load(DATA / "features" / (Path(r["file"]).stem + ".npy")) for r in rows])
    if x.shape[1] != 768 or not np.isfinite(x).all():
        raise ValueError(f"Invalid cached features in split {split}")
    return x, np.array([r["label"] for r in rows]), rows


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def write_predictions(path, rows, y_true, probabilities, classes):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "actor", "true_emotion", "predicted_emotion", *[f"p_{c}" for c in classes]])
        for row, true, p in zip(rows, y_true, probabilities):
            writer.writerow([row["file"], row["actor"], true, classes[int(np.argmax(p))], *np.round(p, 6)])


def main():
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    (x_train, y_train, _), (x_dev, y_dev, dev_rows), (x_test, y_test, test_rows) = (
        load_split(manifest, s) for s in ("train", "dev", "test"))
    print(f"clips: train {len(y_train)}, dev {len(y_dev)}, test {len(y_test)}", flush=True)

    candidates, best, best_f1 = [], None, -1.0
    for c in C_GRID:
        pipeline = make_pipeline(StandardScaler(),
                                 LogisticRegression(C=c, class_weight="balanced", max_iter=3000))
        pipeline.fit(x_train, y_train)
        result = scores(y_dev, pipeline.predict(x_dev))
        candidates.append({"C": c, **result})
        print(f"C={c}: dev macro F1 {result['macro_f1']:.4f}", flush=True)
        if result["macro_f1"] > best_f1:
            best, best_f1 = pipeline, result["macro_f1"]

    classes = [str(c) for c in best.classes_]
    majority = max(set(y_train), key=list(y_train).count)
    test_pred = best.predict(x_test)
    versions = {p: importlib.metadata.version(p) for p in ("numpy", "scikit-learn", "joblib")}

    joblib.dump({"pipeline": best, "manifest": manifest}, MODEL_DIR / "voice_classifier.joblib")
    bundle_path = MODEL_DIR / "bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["sha256"]["voice_classifier.joblib"] = sha256(MODEL_DIR / "voice_classifier.joblib")
    save_json(bundle_path, bundle)

    RESULTS.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS / "dev_metrics.json", {
        "selection": "highest dev macro F1; ties keep the smaller C; no refit on dev",
        "candidates": candidates, "selected_C": float(best[-1].C), "versions": versions})
    save_json(RESULTS / "test_metrics.json", {
        "test": scores(y_test, test_pred),
        "majority_baseline": scores(y_test, np.full(len(y_test), majority)),
        "labels": classes,
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(y_test, test_pred, labels=classes).tolist(),
        "classification_report": classification_report(y_test, test_pred, labels=classes, output_dict=True),
        "scope": "Unseen CREMA-D actors, acted speech; not real customer-service validation."})
    write_predictions(RESULTS / "dev_predictions.csv", dev_rows, y_dev, best.predict_proba(x_dev), classes)
    write_predictions(RESULTS / "test_predictions.csv", test_rows, y_test, best.predict_proba(x_test), classes)

    t = scores(y_test, test_pred)
    print(f"selected C={best[-1].C}; test accuracy {t['accuracy']:.2%}, macro F1 {t['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
