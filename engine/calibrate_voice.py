"""Measure the voice channel's scale and error on dev actors, evaluate on test actors.
Run from project root after train_voice_classifier.py: python engine/calibrate_voice.py

"Truth" for a clip is the reference VAD of its intended (acted) emotion; CREMA-D has no VAD ratings.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from engine.voice_affect import REFERENCE_VAD, expected_vad  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results" / "voice"
DIMS = ("valence", "activity", "dominance")


def load(split):
    rows = list(csv.DictReader((RESULTS / f"{split}_predictions.csv").open(encoding="utf-8")))
    pred = np.array([expected_vad({k[2:]: float(v) for k, v in r.items() if k.startswith("p_")}) for r in rows])
    true = np.array([REFERENCE_VAD[r["true_emotion"]] for r in rows])
    return pred, true


dev_pred, dev_true = load("dev")
test_pred, test_true = load("test")

scale = np.percentile(np.abs(dev_pred), 99, axis=0)
dev_res = (dev_true - dev_pred) / scale
mae, rmse = np.abs(dev_res).mean(axis=0), np.sqrt((dev_res ** 2).mean(axis=0))
fmt = lambda values, digits: "(" + ", ".join(f"{v:.{digits}f}" for v in values) + ")"

print(f"dev clips: {len(dev_pred)}   test clips: {len(test_pred)}")
print(f"VOICE_SCALE = {fmt(scale, 3)}   # 99th percentile of |voice VAD| on dev actors")
print(f"VOICE_SIGMA = {fmt(rmse, 2)}   # RMSE on the rescaled axes, dev actors")
print(f"voice MAE (dev, rescaled) = {fmt(mae, 2)}; bias = {fmt(dev_res.mean(axis=0), 2)}")
z = dev_res / rmse
print(f"within 1 sigma {fmt((np.abs(z) < 1).mean(axis=0), 2)}, within 2 sigma {fmt((np.abs(z) < 2).mean(axis=0), 2)}"
      " (Gaussian: 0.68 / 0.95)")

test_res = (test_true - test_pred) / scale
print("\nTest actors (held out):")
for i, name in enumerate(DIMS):
    r = np.corrcoef(test_true[:, i], test_pred[:, i])[0, 1]
    print(f"  {name:9s} MAE {np.abs(test_res[:, i]).mean():.2f}  RMSE {np.sqrt((test_res[:, i] ** 2).mean()):.2f}"
          f"  Pearson with reference {r:.2f}")
