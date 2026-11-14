"""Local-only frozen emotion2vec plus Salim's saved CREMA-D classifier."""
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "emotion2vec/emotion2vec_base"
REVISION = "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664"
CLASSES = {"anger", "disgust", "fear", "happiness", "neutral", "sadness"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class VoiceEstimator:
    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models" / "voice"
        self._encoder = None
        self._classifier = None

    def _load(self):
        if self._encoder is not None:
            return
        manifest_path = self.model_dir / "bundle_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError("Voice model files missing in engine/models/voice/.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
            raise ValueError("Unexpected voice encoder version.")
        for relative, expected in manifest["sha256"].items():
            path = (self.model_dir / relative).resolve()
            if not path.is_relative_to(self.model_dir.resolve()):
                raise ValueError("Invalid bundle path.")
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Voice asset missing or changed: {relative}")

        # No hub download functions are called during prediction. AutoModel is
        # given a real local directory and its update check is disabled.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        import joblib
        import torch
        from funasr import AutoModel
        # Load only the trusted classifier created by your own experiment.
        saved = joblib.load(self.model_dir / "voice_classifier.joblib")
        meta = saved["manifest"]
        if meta["model"] != MODEL_ID or meta["revision"] != REVISION:
            raise ValueError("Voice classifier and encoder metadata do not match.")
        pipeline = saved["pipeline"]
        if set(pipeline.classes_) != CLASSES or pipeline.n_features_in_ != 768:
            raise ValueError("Expected six classes and 768 input features.")
        torch.set_num_threads(4)
        encoder = AutoModel(model=str(self.model_dir / "encoder"), hub="hf",
                            device="cpu", disable_update=True, disable_pbar=True, disable_log=True)
        encoder.model.eval()
        for parameter in encoder.model.parameters():
            parameter.requires_grad_(False)
        self._classifier = pipeline
        self._encoder = encoder

    @staticmethod
    def load_audio(audio_path):
        """Read a recording as validated 16 kHz mono float32 samples."""
        import numpy as np
        import soundfile as sf
        import librosa

        path = Path(audio_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("Attached audio file does not exist.")
        with sf.SoundFile(path) as handle:
            duration = len(handle) / handle.samplerate
            if not 0 < duration <= 60:
                raise ValueError("Use a recording longer than zero and at most 60 seconds.")
            sample_rate = handle.samplerate
            audio = handle.read(dtype="float32", always_2d=True).mean(axis=1)
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid samples.")
        if np.max(np.abs(audio)) < 1e-6:
            raise ValueError("Recording is silent or effectively silent.")
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        return audio

    def estimate(self, audio_path):
        result = self.estimate_audio(self.load_audio(audio_path))
        result["recording"] = Path(audio_path).name
        return result

    def estimate_audio(self, audio):
        """Emotion probabilities for 16 kHz mono float32 samples."""
        import numpy as np
        import torch

        duration = len(audio) / 16000
        self._load()
        with torch.inference_mode():
            output = self._encoder.generate(input=audio, granularity="utterance",
                                            extract_embedding=True)
        features = np.asarray(output[0]["feats"], dtype=np.float32)
        if features.shape != (768,) or not np.isfinite(features).all():
            raise ValueError("Invalid voice feature vector.")
        probabilities = np.asarray(self._classifier.predict_proba(features[None, :])[0])
        if probabilities.shape != (6,) or not np.isfinite(probabilities).all():
            raise ValueError("Invalid voice classifier output.")
        order = np.argsort(-probabilities)
        labels = self._classifier.classes_
        return {
            "status": "processed",
            "duration_seconds": round(duration, 3),
            "predicted_emotion": str(labels[order[0]]),
            "class_probabilities_uncalibrated": {
                str(labels[i]): round(float(probabilities[i]), 4) for i in order
            },
            "estimator": "frozen emotion2vec_base + logistic regression",
            "scope": "CREMA-D acted-emotion categories; real-world performance not established",
            "valence_activity_status": "not predicted by voice classifier",
            "confidence_status": "class probabilities uncalibrated; not main engine confidence",
        }
