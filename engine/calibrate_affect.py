"""Recompute the AffectiveState constants from held-out EmoBank predictions.
Run from project root after train_regressor.py: python engine/calibrate_affect.py
"""
import csv
from pathlib import Path

import numpy as np

DIMS = ("valence", "activity", "dominance")
LENGTH_BINS = (2, 5, 10)

rows = list(csv.DictReader((Path(__file__).resolve().parent / "results" / "text" / "test_predictions.csv")
                           .open(encoding="utf-8")))
pred = np.array([[float(r[f"pred_{d}"]) for d in DIMS] for r in rows])
true = np.array([[float(r[f"true_{d}"]) for d in DIMS] for r in rows])
words = np.array([len(r["text"].split()) for r in rows])

scale = np.percentile(np.abs(pred), 99, axis=0)
residual = (true - pred) / scale
mae = np.abs(residual).mean(axis=0)
rmse = np.sqrt((residual ** 2).mean(axis=0))
prior = (true / scale).std(axis=0)

edges = (0,) + LENGTH_BINS + (10 ** 9,)
base = (residual ** 2).mean()
length_noise = [round(float((residual[(words > lo) & (words <= hi)] ** 2).mean() / base), 2)
                for lo, hi in zip(edges, edges[1:])]

print(f"held-out sentences: {len(rows)}")
fmt = lambda values, digits: "(" + ", ".join(f"{v:.{digits}f}" for v in values) + ")"
print(f"SCALE       = {fmt(scale, 3)}   # 99th percentile of |prediction|")
print(f"SIGMA       = {fmt(rmse, 2)}   # RMSE on the rescaled axes")
print(f"PRIOR_STD   = {fmt(prior, 2)}   # spread of true affect, rescaled")
print(f"NEAR_ZERO_V = {mae[0]:.2f}   # 1x valence MAE")
print(f"STRONG_A    = {2 * mae[1]:.2f}   # 2x activity MAE")
print(f"LOW_D       = {mae[2]:.2f}   # 1x dominance MAE")
print(f"LENGTH_NOISE factors for <= {LENGTH_BINS} words and longer: {length_noise}")
