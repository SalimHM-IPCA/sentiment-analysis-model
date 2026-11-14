"""Measure how added noise degrades the voice estimates, and which audio cue reveals the noise.
Run from project root after train_voice_prosody.py: python engine/calibrate_voice_noise.py [--per-emotion 50]

White noise is added to dev-actor clips at fixed SNRs; errors are measured against the reference VAD of the
intended emotion, on the rescaled axes (valence and dominance from emotion2vec, activity from prosody).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ENGINE = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGINE.parent))
from engine import acoustic_features as af  # noqa: E402
from engine.voice_affect import REFERENCE_VAD, expected_vad  # noqa: E402
from engine.voice_estimator import VoiceEstimator  # noqa: E402

DATA, RESULTS = ENGINE / "data" / "voice", ENGINE / "results" / "voice"
VOICE_SCALE = (0.895, 0.727, 0.664)
SNR_LEVELS = (None, 20, 10, 5, 0)


def add_noise(audio, snr_db, rng):
    if snr_db is None:
        return audio
    noise_power = np.mean(audio ** 2) / 10 ** (snr_db / 10)
    return (audio + rng.normal(0, np.sqrt(noise_power), len(audio))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-emotion", type=int, default=50)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    dev = [r for r in manifest["rows"] if r["split"] == "dev"]
    sample = []
    for emotion in sorted(REFERENCE_VAD):
        rows = [r for r in dev if r["label"] == emotion]
        sample += [rows[i] for i in rng.choice(len(rows), args.per_emotion, replace=False)]

    prosody = joblib.load(ENGINE / "models" / "voice" / "prosody_activity.joblib")
    estimator = VoiceEstimator()
    records = []
    for n, row in enumerate(sample, 1):
        clean = af.load_audio(DATA / "audio" / row["file"])
        ref = REFERENCE_VAD[row["label"]]
        for snr in SNR_LEVELS:
            audio = add_noise(clean, snr, rng)
            probs = estimator.estimate_audio(audio)["class_probabilities_uncalibrated"]
            vad = expected_vad(probs)
            feats = af.analyse(audio)
            activity = prosody["pipeline"].predict(np.array([[feats[c] for c in prosody["features"]]]))[0]
            records.append({
                "file": row["file"], "snr": "clean" if snr is None else snr,
                "err_v": (ref[0] - vad[0]) / VOICE_SCALE[0],
                "err_a": (ref[1] - activity) / prosody["scale"],
                "err_d": (ref[2] - vad[2]) / VOICE_SCALE[2],
                "flatness": feats["flatness"], "energy_range_db": feats["energy_range_db"],
            })
        if n % 25 == 0:
            print(f"{n}/{len(sample)} clips", flush=True)

    with (RESULTS / "noise_experiment.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    clean = {d: np.sqrt(np.mean([r[f"err_{d}"] ** 2 for r in records if r["snr"] == "clean"])) for d in "vad"}
    print(f"\n{'SNR':>6}  error variance factor (V / A / D)    median flatness  median energy range")
    for snr in SNR_LEVELS:
        key = "clean" if snr is None else snr
        sub = [r for r in records if r["snr"] == key]
        factor = [np.mean([r[f"err_{d}"] ** 2 for r in sub]) / clean[d] ** 2 for d in "vad"]
        print(f"{str(key):>6}  {factor[0]:5.2f} / {factor[1]:5.2f} / {factor[2]:5.2f}"
              f"              {np.median([r['flatness'] for r in sub]):.4f}"
              f"           {np.median([r['energy_range_db'] for r in sub]):5.1f} dB")


if __name__ == "__main__":
    main()
