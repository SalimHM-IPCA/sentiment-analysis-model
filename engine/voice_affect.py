"""Turns a recording into a voice reading on the affective map: valence and dominance from the emotion2vec
classifier, activity from speaker-invariant prosody (loudness and pitch movement, never absolute pitch)."""

import numpy as np

# NRC VAD Lexicon (Mohammad, ACL 2018) scores for the emotion names themselves, converted from 0..1 to -1..1.
NRC_VAD_SCORES = {
    "anger": (0.167, 0.865, 0.657),
    "disgust": (0.052, 0.775, 0.317),
    "fear": (0.073, 0.840, 0.293),
    "happiness": (0.960, 0.732, 0.850),
    "neutral": (0.469, 0.184, 0.357),
    "sadness": (0.052, 0.288, 0.164),
}
REFERENCE_VAD = {emotion: tuple(2 * s - 1 for s in scores) for emotion, scores in NRC_VAD_SCORES.items()}

# Measured on dev actors (calibrate_voice.py, train_voice_prosody.py): +-1 = top 1% of the channel's readings.
VOICE_SCALE = (0.895, 0.727, 0.664)
VOICE_SIGMA = (0.57, 0.56, 0.53)


def expected_vad(probabilities):
    """Probability-weighted average of the reference VAD of each emotion."""
    total = sum(probabilities.values())
    return tuple(sum(p * REFERENCE_VAD[e][i] for e, p in probabilities.items()) / total for i in range(3))


# Measured with calibrate_voice_noise.py (300 dev clips, white noise at 20/10/5/0 dB SNR): median spectral
# flatness at each level and the resulting error-variance factor for valence, activity, dominance.
NOISE_FLATNESS = (0.0076, 0.0648, 0.2930, 0.4376, 0.5097)
NOISE_FACTOR = ((1.00, 1.38, 2.06, 2.36, 2.53),
                (1.00, 1.01, 1.01, 1.03, 1.08),
                (1.00, 1.70, 2.86, 3.46, 4.11))


def noise_factors(features):
    """Error-variance multipliers for (valence, activity, dominance) given the measured spectral flatness."""
    return tuple(float(np.interp(features["flatness"], NOISE_FLATNESS, f)) for f in NOISE_FACTOR)


class VoiceAffect:
    def __init__(self):
        self._estimator = None
        self._prosody = None

    def _load(self):
        if self._estimator is None:
            from pathlib import Path
            import joblib
            from .voice_estimator import VoiceEstimator
            self._estimator = VoiceEstimator()
            self._prosody = joblib.load(Path(__file__).resolve().parent / "models" / "voice" / "prosody_activity.joblib")

    def analyse(self, audio_path):
        from . import acoustic_features as af
        self._load()
        audio = self._estimator.load_audio(audio_path)
        emotion = self._estimator.estimate_audio(audio)
        probabilities = emotion["class_probabilities_uncalibrated"]
        valence, _, dominance = expected_vad(probabilities)
        features = af.analyse(audio)
        x = np.array([[features[c] for c in self._prosody["features"]]])
        activity = float(self._prosody["pipeline"].predict(x)[0])
        observation = (valence / VOICE_SCALE[0], activity / self._prosody["scale"], dominance / VOICE_SCALE[2])
        return {
            "observation": observation,
            "noise": noise_factors(features),
            "probabilities": probabilities,
            "predicted_emotion": emotion["predicted_emotion"],
            "acoustics": {
                "duration_s": round(features["duration_s"], 2),
                "loudness_db": round(features["loudness_db"], 1),
                "pitch_median_hz": round(features["pitch_median_hz"], 1),
                "pitch_sd_semitones": round(features["pitch_sd_semitones"], 2),
                "flatness": round(features["flatness"], 4),
            },
        }
