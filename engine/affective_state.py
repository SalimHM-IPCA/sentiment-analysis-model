import math


class AffectiveState:
    ALPHA             = 0.6
    NEUTRAL_THRESHOLD = 0.08   # calibrated to DistilBERT+Ridge output range (~±0.30)
    ANGER_THRESHOLD   = 0.20   # calibrated to DistilBERT+Ridge output range (~±0.30)

    def reset(self):
        self.smoothed_valence  = None
        self.smoothed_activity = None
        self.history           = []
        self.message_count     = 0

    def _apply_ema(self, raw_valence, raw_activity):
        if self.smoothed_valence is None:
            self.smoothed_valence  = raw_valence
            self.smoothed_activity = raw_activity
        else:
            self.smoothed_valence  = self.ALPHA * raw_valence  + (1 - self.ALPHA) * self.smoothed_valence
            self.smoothed_activity = self.ALPHA * raw_activity + (1 - self.ALPHA) * self.smoothed_activity

    def _derive_label(self, valence, activity):
        if abs(valence) < self.NEUTRAL_THRESHOLD and abs(activity) < self.NEUTRAL_THRESHOLD:
            return "neutral"
        if valence <= 0:
            if activity >= 0:
                return "anger" if valence < -self.ANGER_THRESHOLD else "frustration"
            return "sadness"
        return "joy" if activity >= 0 else "calm"

    def _compute_confidence(self, text, raw_valence, raw_activity, flags):
        distance   = math.sqrt(raw_valence ** 2 + raw_activity ** 2)
        confidence = min(1.0, distance / math.sqrt(2))
        word_count = len(text.split())
        if word_count <= 2:
            confidence = min(confidence, 0.35)
        elif word_count <= 5:
            confidence = min(confidence, 0.65)
        if "possible_sarcasm" in flags:
            confidence = max(0.0, confidence - 0.2)
        return max(0.1, min(0.95, confidence))

    def update(self, text, raw):
        raw_valence  = raw["raw_valence"]
        raw_activity = raw["raw_activity"]
        flags        = raw.get("flags", [])

        self._apply_ema(raw_valence, raw_activity)
        self.message_count += 1
        self.history.append({"valence": self.smoothed_valence,
                              "activity": self.smoothed_activity})

        label      = self._derive_label(self.smoothed_valence, self.smoothed_activity)
        confidence = self._compute_confidence(text, raw_valence, raw_activity, flags)

        return {
            "valence":    round(self.smoothed_valence,  4),
            "activity":   round(self.smoothed_activity, 4),
            "confidence": round(confidence, 4),
            "label":      label,
            "flags":      flags,
            "details": {
                "raw_valence":   raw_valence,
                "raw_activity":  raw_activity,
                "message_count": self.message_count,
                "alpha":         self.ALPHA,
                "word_count":    len(text.split()),
            },
        }
