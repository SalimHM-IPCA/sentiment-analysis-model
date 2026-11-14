import math

# All constants are measured on held-out EmoBank predictions; recompute with engine/calibrate_affect.py.
SCALE = (0.299, 0.198, 0.144)
SIGMA = (0.42, 0.54, 0.66)
PRIOR_STD = (0.58, 0.62, 0.73)
NEAR_ZERO_V = 0.31
STRONG_A = 0.84
LOW_D = 0.49
LENGTH_NOISE = ((2, 1.26), (5, 1.14), (10, 1.03), (math.inf, 0.93))

# Not measurable from data: sarcasm is an unvalidated cue, so it only doubles the message noise.
SARCASM_NOISE = 2.0
# Weight of a normal message once the conversation is under way (the original EMA alpha).
TARGET_GAIN = 0.6

DIMENSIONS = ("valence", "activity", "dominance")


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def derive_label(valence, activity, dominance):
    if abs(valence) < NEAR_ZERO_V and activity < STRONG_A:
        return "neutral"
    if valence <= 0:
        if activity >= STRONG_A and dominance >= 0:
            return "anger"
        if dominance <= -LOW_D:
            return "sadness"
        return "frustration"
    return "joy" if activity >= STRONG_A else "calm"


def label_probabilities(valence, activity, dominance, sd_v, sd_a, sd_d):
    """Probability that the true affect lies in each label region, assuming Gaussian independent errors."""
    below = lambda x, threshold, sd: _phi((threshold - x) / sd)
    v_neg_strong = below(valence, -NEAR_ZERO_V, sd_v)
    v_neg_weak = below(valence, 0.0, sd_v) - v_neg_strong
    v_pos_weak = below(valence, NEAR_ZERO_V, sd_v) - below(valence, 0.0, sd_v)
    v_pos_strong = 1.0 - below(valence, NEAR_ZERO_V, sd_v)
    a_strong = 1.0 - below(activity, STRONG_A, sd_a)
    d_non_negative = 1.0 - below(dominance, 0.0, sd_d)
    d_low = below(dominance, -LOW_D, sd_d)
    v_neg = v_neg_strong + v_neg_weak
    probs = {
        "neutral": (v_neg_weak + v_pos_weak) * (1.0 - a_strong),
        "anger": v_neg * a_strong * d_non_negative,
        "sadness": d_low * (v_neg_strong + v_neg_weak * a_strong),
        "joy": (v_pos_weak + v_pos_strong) * a_strong,
        "calm": v_pos_strong * (1.0 - a_strong),
    }
    probs["frustration"] = max(0.0, v_neg - probs["anger"] - probs["sadness"] - v_neg_weak * (1.0 - a_strong))
    return probs


class AffectiveState:
    """Tracks valence, activity and dominance over a conversation with a per-dimension Kalman filter."""

    def __init__(self):
        self._voice = None

    def reset(self):
        self.state = [0.0, 0.0, 0.0]
        self.variance = [s ** 2 for s in PRIOR_STD]
        self.history = []
        self.message_count = 0

    def _message_noise(self, word_count, flags):
        noise = next(factor for limit, factor in LENGTH_NOISE if word_count <= limit)
        if "possible_sarcasm" in flags:
            noise *= SARCASM_NOISE
        return noise

    def _kalman_step(self, observation, sigma, noise, new_message=True):
        # Process noise chosen so a normal text message settles at TARGET_GAIN weight. A second channel of the
        # same message (voice) is a further measurement of the same moment, so no process noise is added.
        process_factor = TARGET_GAIN ** 2 / (1.0 - TARGET_GAIN)
        gains = []
        for i in range(3):
            measurement_var = sigma[i] ** 2 * noise[i]
            predicted_var = self.variance[i]
            if new_message and self.message_count:
                predicted_var += process_factor * SIGMA[i] ** 2
            gain = predicted_var / (predicted_var + measurement_var)
            self.state[i] += gain * (observation[i] - self.state[i])
            self.variance[i] = (1.0 - gain) * predicted_var
            gains.append(gain)
        return gains

    @staticmethod
    def _cap_confidence(confidence, word_count, flags, voice_noise=None):
        # Short-text caps only apply when the text is the only evidence for this message.
        if voice_noise is None and word_count <= 2:
            confidence = min(confidence, 0.35)
        elif voice_noise is None and word_count <= 5:
            confidence = min(confidence, 0.65)
        if voice_noise is not None:
            # Rule-based quality penalty: a noisy recording is weaker evidence, so confidence drops with it.
            confidence /= math.sqrt(sum(voice_noise) / len(voice_noise))
        if "possible_sarcasm" in flags:
            confidence -= 0.2
        return max(0.1, min(0.95, confidence))

    def _voice_step(self, audio_path, flags):
        """Second measurement of the same message from the recording; a failure never affects the text result."""
        try:
            if self._voice is None:
                from .voice_affect import VoiceAffect
                self._voice = VoiceAffect()
            from .voice_affect import VOICE_SIGMA
            voice = self._voice.analyse(audio_path)
        except Exception as error:
            flags.append("voice_processing_failed")
            return {"status": "failed", "error_type": type(error).__name__, "error": str(error)}
        gains = self._kalman_step(voice["observation"], VOICE_SIGMA, voice["noise"], new_message=False)
        reading = [max(-1.0, min(1.0, v)) for v in voice["observation"]]
        return {
            "status": "processed",
            "voice_only": {**{d: round(v, 4) for d, v in zip(DIMENSIONS, reading)},
                           "label": derive_label(*voice["observation"])},
            "predicted_emotion": voice["predicted_emotion"],
            "emotion_probabilities": voice["probabilities"],
            "noise_factor": [round(n, 2) for n in voice["noise"]],
            "message_weight": [round(g, 3) for g in gains],
            "acoustics": voice["acoustics"],
        }

    def update(self, text, raw, audio_path=None):
        flags = list(raw.get("flags", []))
        word_count = len(text.split())
        raw_values = [raw[f"raw_{d}"] for d in DIMENSIONS]
        # The state stays unclipped so strong evidence is not lost; only the reported values are clipped.
        observation = [value / scale for value, scale in zip(raw_values, SCALE)]

        noise = self._message_noise(word_count, flags)
        gains = self._kalman_step(observation, SIGMA, (noise, noise, noise))
        voice = self._voice_step(audio_path, flags) if audio_path else None
        voice_noise = voice["noise_factor"] if voice and voice["status"] == "processed" else None
        self.message_count += 1

        valence, activity, dominance = self.state
        label = derive_label(valence, activity, dominance)
        probs = label_probabilities(valence, activity, dominance, *(math.sqrt(v) for v in self.variance))
        confidence = self._cap_confidence(probs[label], word_count, flags, voice_noise)

        reported = [max(-1.0, min(1.0, value)) for value in self.state]
        self.history.append(dict(zip(DIMENSIONS, reported)))
        details_voice = {"voice": voice} if voice else {}
        return {
            "valence": round(reported[0], 4),
            "activity": round(reported[1], 4),
            "dominance": round(reported[2], 4),
            "confidence": round(confidence, 4),
            "label": label,
            "flags": flags,
            "details": {
                "raw_valence": raw_values[0],
                "raw_activity": raw_values[1],
                "raw_dominance": raw_values[2],
                "message_count": self.message_count,
                "word_count": word_count,
                "message_weight": [round(g, 3) for g in gains],
                "label_probabilities": {k: round(p, 3) for k, p in probs.items()},
                **details_voice,
            },
        }
