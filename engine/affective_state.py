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

    def _kalman_step(self, observation, noise):
        # Process noise chosen so a normal message settles at TARGET_GAIN weight.
        process_factor = TARGET_GAIN ** 2 / (1.0 - TARGET_GAIN)
        gains = []
        for i in range(3):
            measurement_var = SIGMA[i] ** 2 * noise
            predicted_var = self.variance[i]
            if self.message_count:
                predicted_var += process_factor * SIGMA[i] ** 2
            gain = predicted_var / (predicted_var + measurement_var)
            self.state[i] += gain * (observation[i] - self.state[i])
            self.variance[i] = (1.0 - gain) * predicted_var
            gains.append(gain)
        return gains

    @staticmethod
    def _cap_confidence(confidence, word_count, flags):
        if word_count <= 2:
            confidence = min(confidence, 0.35)
        elif word_count <= 5:
            confidence = min(confidence, 0.65)
        if "possible_sarcasm" in flags:
            confidence -= 0.2
        return max(0.1, min(0.95, confidence))

    def update(self, text, raw):
        flags = raw.get("flags", [])
        word_count = len(text.split())
        raw_values = [raw[f"raw_{d}"] for d in DIMENSIONS]
        # The state stays unclipped so strong evidence is not lost; only the reported values are clipped.
        observation = [value / scale for value, scale in zip(raw_values, SCALE)]

        gains = self._kalman_step(observation, self._message_noise(word_count, flags))
        self.message_count += 1

        valence, activity, dominance = self.state
        label = derive_label(valence, activity, dominance)
        probs = label_probabilities(valence, activity, dominance, *(math.sqrt(v) for v in self.variance))
        confidence = self._cap_confidence(probs[label], word_count, flags)

        reported = [max(-1.0, min(1.0, value)) for value in self.state]
        self.history.append(dict(zip(DIMENSIONS, reported)))
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
            },
        }
