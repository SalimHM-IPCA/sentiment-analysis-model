"""Run with:  python tests/test_voice.py   (or: python -m unittest discover tests)
Voice channel tests. The full-engine tests are skipped when the voice encoder weights are not downloaded."""
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import acoustic_features as af  # noqa: E402

AUDIO = os.path.join(ROOT, "assets", "audio")
ENCODER_WEIGHTS = os.path.join(ROOT, "engine", "models", "voice", "encoder", "emotion2vec_base.pt")
NEUTRAL_TEXT = "Here is my message."


def professor_file(name):
    return os.path.join(AUDIO, name + ".wav")


def sine(frequency, seconds=1.0, amplitude=0.3):
    t = np.arange(int(af.SAMPLE_RATE * seconds)) / af.SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class AcousticFeatureTests(unittest.TestCase):
    def test_pitch_of_pure_tone(self):
        self.assertAlmostEqual(af.analyse(sine(200.0))["pitch_median_hz"], 200.0, delta=6.0)

    def test_frames_and_mfcc_shape(self):
        frames = af.frame_signal(sine(200.0), af.FRAME_MS, af.HOP_MS)
        self.assertEqual(frames.shape, (98, 400))  # 25 ms frames every 10 ms at 16 kHz
        self.assertEqual(af.mfcc(af.power_spectrum(frames)).shape, (98, 12))  # coefficients 2-13

    def test_mel_scale_matches_course_values(self):
        self.assertAlmostEqual(af.hz_to_mel(300.0), 401.25, places=1)
        self.assertAlmostEqual(af.hz_to_mel(4000.0), 2142.27, places=2)
        self.assertAlmostEqual(af.mel_to_hz(af.hz_to_mel(1234.0)), 1234.0, places=6)

    def test_flatness_separates_noise_from_voice_like_tone(self):
        noise = np.random.default_rng(0).normal(0, 0.1, af.SAMPLE_RATE).astype(np.float32)
        self.assertGreater(af.analyse(noise)["flatness"], 0.3)
        self.assertLess(af.analyse(sine(200.0))["flatness"], 0.01)


class ProsodyAndNoiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import joblib
        from engine.voice_affect import noise_factors
        cls.noise_factors = staticmethod(noise_factors)
        cls.prosody = joblib.load(os.path.join(ROOT, "engine", "models", "voice", "prosody_activity.joblib"))
        cls.features = {n: af.analyse(af.load_audio(professor_file(n))) for n in
                        ("calm_clean", "calm_noisy", "agitated_clean", "agitated_noisy", "speaker_low", "speaker_high")}

    def activity(self, name):
        x = np.array([[self.features[name][c] for c in self.prosody["features"]]])
        return float(self.prosody["pipeline"].predict(x)[0]) / self.prosody["scale"]

    def test_absolute_pitch_is_not_an_input(self):
        self.assertNotIn("pitch_median_hz", self.prosody["features"])

    def test_speaker_pitch_does_not_change_activity(self):
        self.assertLess(abs(self.activity("speaker_high") - self.activity("speaker_low")), 0.15)

    def test_agitated_is_more_activated_than_calm(self):
        self.assertGreater(self.activity("agitated_clean"), self.activity("calm_clean") + 0.3)
        self.assertGreater(self.activity("agitated_noisy"), self.activity("calm_noisy") + 0.3)

    def test_noise_increases_measurement_noise_only_for_noisy_audio(self):
        for name in ("calm_clean", "agitated_clean", "speaker_low", "speaker_high"):
            self.assertLess(max(self.noise_factors(self.features[name])), 1.1, name)
        for name in ("calm_noisy", "agitated_noisy"):
            self.assertGreater(self.noise_factors(self.features[name])[0], 1.5, name)


@unittest.skipUnless(os.path.isfile(ENCODER_WEIGHTS), "voice encoder weights not downloaded")
class VoiceEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from harness.contract import normalize
        from harness.loader import load_engine
        cls.normalize = staticmethod(normalize)
        cls.engine = load_engine(ROOT)

    def run_one(self, text, audio=None):
        self.engine.reset()
        return self.engine.process(text, audio_path=audio)

    def test_contract_holds_with_audio(self):
        for name in ("calm_clean", "agitated_noisy", "speaker_high"):
            result = self.normalize(self.run_one(NEUTRAL_TEXT, professor_file(name)))
            self.assertEqual(result["problems"], [], name)

    def test_voice_reading_is_logged_and_moves_the_state(self):
        text_only = self.run_one(NEUTRAL_TEXT)
        with_voice = self.run_one(NEUTRAL_TEXT, professor_file("agitated_clean"))
        self.assertNotIn("voice", text_only["details"])
        self.assertEqual(with_voice["details"]["voice"]["status"], "processed")
        self.assertGreater(with_voice["activity"], text_only["activity"] + 0.1)

    def test_noise_lowers_confidence(self):
        for clean, noisy in (("calm_clean", "calm_noisy"), ("agitated_clean", "agitated_noisy")):
            self.assertLess(self.run_one(NEUTRAL_TEXT, professor_file(noisy))["confidence"],
                            self.run_one(NEUTRAL_TEXT, professor_file(clean))["confidence"] - 0.05)

    def test_broken_audio_keeps_the_text_result(self):
        text_only = self.run_one(NEUTRAL_TEXT)
        broken = self.run_one(NEUTRAL_TEXT, os.path.join(AUDIO, "does_not_exist.wav"))
        self.assertIn("voice_processing_failed", broken["flags"])
        for key in ("valence", "activity", "label", "action"):
            self.assertEqual(broken[key], text_only[key])
        self.assertEqual(self.normalize(broken)["problems"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
