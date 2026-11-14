"""Offline text perception using your saved DistilBERT + Ridge bundle.

From the project root:
    from engine.text_estimator import TextEstimator
    estimator = TextEstimator()  # load once, reuse for every message
    result = estimator.estimate("I am unhappy with my order.")

Includes an unvalidated possible_sarcasm flag that never changes scores.
This module estimates affect only. Confidence, labels, conversation state,
response actions and replies are separate integration responsibilities.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import joblib
import numpy as np


OS_JUNK = {".DS_Store", "Thumbs.db", "desktop.ini"}


def model_fingerprint(path):
    """Match the fingerprint procedure used by train_regressor.py."""
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


def possible_sarcasm(text):
    """Unvalidated surface cue only; never changes learned affect scores.

    Looks for opening praise followed by a problem. Repair language can
    suppress the cue. Context, mixed messages and implicit sarcasm can fail.
    """
    lower = text.lower().replace("’", "'")
    praise = re.match(r"^\s*(?:(?:oh|well)[, ]+)?(?:great|brilliant|fantastic|wonderful|excellent|amazing|perfect|how thoughtful)\b", lower)
    problem = re.search(r"\b(?:broken|defective|damaged|delay|delayed|overdue|silence|failed|failure)\b|\bcharg(?:e|ed|ing)\s+(?:me\s+)?twice\b", lower)
    repair = re.search(r"\b(?:fixed|resolved|repaired|replaced|solved)\b", lower)
    if repair:
        previous = re.findall(r"[a-z']+", lower[:repair.start()])[-3:]
        negators = {"not", "never", "no", "neither", "nor", "cannot", "can't", "without", "hardly"}
        repair = not any(w in negators or w.endswith("n't") for w in previous)
    return bool(praise and problem and not repair)


class TextEstimator:
    def __init__(self, engine_dir=None):
        root = Path(engine_dir).resolve() if engine_dir else Path(__file__).resolve().parent
        model_path = root / "emotion_model"
        bundle_path = root / "training_output" / "regressor.joblib"
        if not model_path.is_dir() or not bundle_path.is_file():
            raise FileNotFoundError(
                "Expected emotion_model/ and training_output/regressor.joblib "
                f"inside {root}. Keep the saved training files in place."
            )
        # Load only your own trusted training output: joblib uses pickle internally.
        bundle = joblib.load(bundle_path)
        self.regressor = bundle["regressor"]
        self.metadata = bundle["metadata"]
        if self.metadata.get("targets") != ["valence", "activity", "dominance"]:
            raise ValueError("The regressor must predict valence, activity, then dominance.")
        if self.metadata.get("prediction_clip") != [-1, 1]:
            raise ValueError("Unexpected prediction clipping settings.")
        if self.metadata.get("normalisation") != "(EmoBank score - 3) / 2":
            raise ValueError("Unexpected target normalisation.")
        if self.regressor.n_features_in_ != 768:
            raise ValueError("Expected a Ridge regressor with 768 input features.")
        if model_fingerprint(model_path) != self.metadata.get("model_sha256"):
            raise ValueError("The encoder differs from the one used for training. Restore the original emotion_model folder.")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.encoder = SentenceTransformer(str(model_path), device="cpu", local_files_only=True)
        self.encoder.max_seq_length = int(self.metadata["max_seq_length"])

    def estimate(self, text):
        """Return finite raw affect scores in [-1,1] for one non-empty text.

        'raw' means before conversation tracking or label mapping. Scores are
        already on the project's scale; do not normalise them a second time.
        Empty input raises ValueError rather than inventing a neutral estimate.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text.strip():
            raise ValueError("Enter a non-empty message.")
        vector = self.encoder.encode(
            [text], batch_size=1, convert_to_numpy=True,
            normalize_embeddings=bool(self.metadata["normalize_embeddings"]),
            show_progress_bar=False,
        )
        if vector.shape != (1, 768) or not np.isfinite(vector).all():
            raise ValueError("The encoder returned invalid embeddings.")
        prediction = np.asarray(self.regressor.predict(vector))
        if prediction.shape != (1, 3) or not np.isfinite(prediction).all():
            raise ValueError("The regressor returned invalid predictions.")
        valence, activity, dominance = np.clip(prediction[0], -1.0, 1.0)
        return {
            "raw_valence": float(valence), "raw_activity": float(activity),
            "raw_dominance": float(dominance),
            "flags": ["possible_sarcasm"] if possible_sarcasm(text) else [],
        }


if __name__ == "__main__":
    print("Loading your saved local estimator...", flush=True)
    estimator = TextEstimator()
    print("Ready. Enter an English customer message; type /quit to stop.", flush=True)
    while True:
        try:
            message = input("Message: ")
        except (EOFError, KeyboardInterrupt):
            print("\nFinished.")
            break
        if message.strip().lower() == "/quit":
            break
        try:
            print(json.dumps(estimator.estimate(message), indent=2, allow_nan=False))
        except (TypeError, ValueError) as error:
            print(error)
