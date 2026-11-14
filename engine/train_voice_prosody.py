"""Train the prosodic activity estimator on cached CREMA-D acoustic features.
Run from project root after extract_acoustic_features.py: python engine/train_voice_prosody.py

Target: reference activity (NRC VAD) of the intended emotion. Ridge alpha chosen on dev actors, test actors
evaluated once. Absolute pitch is excluded so the estimate does not depend on a speaker's natural pitch;
the variants with absolute pitch and with MFCCs are trained only for comparison.
"""
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE.parent))
from engine.voice_affect import REFERENCE_VAD, expected_vad  # noqa: E402

DATA, RESULTS, MODELS = ENGINE / "data" / "voice", ENGINE / "results" / "voice", ENGINE / "models" / "voice"
PROSODY = ["loudness_db", "loudness_sd_db", "pitch_sd_semitones", "pitch_range_semitones", "voiced_ratio"]
VARIANTS = {
    "speaker_invariant": PROSODY,
    "with_absolute_pitch": PROSODY + ["log_pitch_median"],
    "with_mfcc": PROSODY + [f"mfcc{i}_mean" for i in range(2, 14)] + [f"mfcc{i}_sd" for i in range(2, 14)],
}
ALPHAS = (0.1, 1.0, 10.0, 100.0)


def load():
    rows = list(csv.DictReader((DATA / "acoustic_features.csv").open(encoding="utf-8")))
    for r in rows:
        pitch = float(r["pitch_median_hz"])
        r["log_pitch_median"] = np.log(pitch) if np.isfinite(pitch) else np.nan
    return rows


def matrix(rows, columns):
    x = np.array([[float(r[c]) for c in columns] for r in rows])
    return np.where(np.isfinite(x), x, np.nanmedian(x, axis=0))


def fit_variant(rows, columns):
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "dev", "test")}
    y = {s: np.array([REFERENCE_VAD[r["label"]][1] for r in split[s]]) for s in split}
    x = {s: matrix(split[s], columns) for s in split}
    best, best_mae, search = None, np.inf, []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x["train"], y["train"])
        mae = np.abs(model.predict(x["dev"]) - y["dev"]).mean()
        search.append({"alpha": alpha, "dev_mae": float(mae)})
        if mae < best_mae:
            best, best_mae = model, mae
    pred = {s: best.predict(x[s]) for s in ("dev", "test")}
    return best, split, y, pred, search


def speaker_bias(split, y, pred):
    """Correlation, across held-out actors, between an actor's median pitch and their mean residual."""
    actors = {}
    for s in ("dev", "test"):
        for r, true, p in zip(split[s], y[s], pred[s]):
            actors.setdefault(r["actor"], {"pitch": [], "res": []})
            actors[r["actor"]]["pitch"].append(float(r["pitch_median_hz"]))
            actors[r["actor"]]["res"].append(p - true)
    pitch = np.array([np.nanmedian(a["pitch"]) for a in actors.values()])
    residual = np.array([np.mean(a["res"]) for a in actors.values()])
    return float(np.corrcoef(pitch, residual)[0, 1])


def main():
    rows = load()
    e2v = {r["file"]: expected_vad({k[2:]: float(v) for k, v in r.items() if k.startswith("p_")})[1]
           for split in ("dev", "test")
           for r in csv.DictReader((RESULTS / f"{split}_predictions.csv").open(encoding="utf-8"))}
    report = {}
    for name, columns in VARIANTS.items():
        model, split, y, pred, search = fit_variant(rows, columns)
        scale = float(np.percentile(np.abs(pred["dev"]), 99))
        dev_res = (y["dev"] - pred["dev"]) / scale
        e2v_res = np.array([REFERENCE_VAD[r["label"]][1] - e2v[r["file"]] for r in split["dev"]])
        report[name] = {
            "features": columns, "alpha_search": search, "selected_alpha": float(model[-1].alpha),
            "scale": scale, "sigma_dev": float(np.sqrt((dev_res ** 2).mean())),
            "dev_pearson": float(np.corrcoef(y["dev"], pred["dev"])[0, 1]),
            "test_pearson": float(np.corrcoef(y["test"], pred["test"])[0, 1]),
            "test_mae_rescaled": float(np.abs((y["test"] - pred["test"]) / scale).mean()),
            "speaker_pitch_vs_residual_corr": speaker_bias(split, y, pred),
            "residual_corr_with_emotion2vec": float(np.corrcoef(dev_res, e2v_res)[0, 1]),
        }
        r = report[name]
        print(f"{name:20s} sigma(dev) {r['sigma_dev']:.2f}  pearson dev {r['dev_pearson']:.2f} test {r['test_pearson']:.2f}  "
              f"speaker bias {r['speaker_pitch_vs_residual_corr']:+.2f}  residual corr with emotion2vec "
              f"{r['residual_corr_with_emotion2vec']:+.2f}")
        if name == "speaker_invariant":
            joblib.dump({"pipeline": model, "features": columns, "scale": scale, "sigma": r["sigma_dev"]},
                        MODELS / "prosody_activity.joblib")
    (RESULTS / "prosody_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
