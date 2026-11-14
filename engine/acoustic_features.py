"""Acoustic features computed from scratch with numpy: framing, windowing, spectrum, MFCC, pitch, loudness, flatness.

Follows the course pipeline: 25 ms frames every 10 ms, Hamming window, power spectrum (512-point FFT, 257 bins),
26 triangular mel filters, log, DCT, keep coefficients 2-13.
"""
import numpy as np

SAMPLE_RATE = 16000
FRAME_MS, HOP_MS = 25, 10
PITCH_FRAME_MS = 40          # long enough to hold two periods of a 60 Hz voice
N_FFT = 512
N_MELS = 26
MEL_FMIN = 300.0
PITCH_MIN_HZ, PITCH_MAX_HZ = 60.0, 400.0
VOICING_THRESHOLD = 0.5      # normalised autocorrelation peak needed to call a frame voiced
ACTIVE_RANGE_DB = 30.0       # frames within 30 dB of the loudest frame count as active (not silence)
EPS = 1e-10


def load_audio(path):
    import librosa
    import soundfile as sf
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return audio


def frame_signal(signal, frame_ms, hop_ms, sample_rate=SAMPLE_RATE):
    """Split the signal into overlapping frames (the signal is treated as stationary inside each frame)."""
    frame_len, hop = int(sample_rate * frame_ms / 1000), int(sample_rate * hop_ms / 1000)
    if len(signal) < frame_len:
        signal = np.pad(signal, (0, frame_len - len(signal)))
    count = 1 + (len(signal) - frame_len) // hop
    index = np.arange(frame_len)[None, :] + hop * np.arange(count)[:, None]
    return signal[index]


def power_spectrum(frames):
    """Hamming window (frame edges go to zero), FFT, then the periodogram |X(k)|^2 / N."""
    windowed = frames * np.hamming(frames.shape[1])
    return np.abs(np.fft.rfft(windowed, n=N_FFT)) ** 2 / frames.shape[1]


def hz_to_mel(f):
    return 1125.0 * np.log(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (np.exp(m / 1125.0) - 1.0)


def mel_filterbank(sample_rate=SAMPLE_RATE):
    """Triangular filters equally spaced on the mel scale (so denser at low frequencies, like the ear)."""
    mel_points = np.linspace(hz_to_mel(MEL_FMIN), hz_to_mel(sample_rate / 2), N_MELS + 2)
    bins = np.floor((N_FFT + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    bank = np.zeros((N_MELS, N_FFT // 2 + 1))
    for m in range(1, N_MELS + 1):
        left, centre, right = bins[m - 1], bins[m], bins[m + 1]
        bank[m - 1, left:centre] = (np.arange(left, centre) - left) / max(centre - left, 1)
        bank[m - 1, centre:right] = (right - np.arange(centre, right)) / max(right - centre, 1)
    return bank


def dct_matrix(n):
    k, i = np.arange(n)[:, None], np.arange(n)[None, :]
    matrix = np.sqrt(2.0 / n) * np.cos(np.pi * k * (2 * i + 1) / (2 * n))
    matrix[0] /= np.sqrt(2.0)
    return matrix


def mfcc(power):
    """Mel energies, log, DCT; keep coefficients 2-13 (index 1..12), dropping the overall-energy coefficient."""
    mel_energy = power @ mel_filterbank().T
    return (np.log(mel_energy + EPS) @ dct_matrix(N_MELS).T)[:, 1:13]


def spectral_flatness(power):
    """Geometric / arithmetic mean of the spectrum: near 0 for harmonic voice, higher for noise."""
    return np.exp(np.mean(np.log(power + EPS), axis=1)) / (np.mean(power, axis=1) + EPS)


def frame_pitch(frames, sample_rate=SAMPLE_RATE):
    """Pitch per frame from the autocorrelation peak; NaN for unvoiced frames."""
    frames = frames - frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(frames, n=2 * frames.shape[1])
    autocorr = np.fft.irfft(np.abs(spectrum) ** 2)[:, :frames.shape[1]]
    autocorr /= autocorr[:, :1] + EPS
    low, high = int(sample_rate / PITCH_MAX_HZ), int(sample_rate / PITCH_MIN_HZ)
    lag = low + np.argmax(autocorr[:, low:high], axis=1)
    peak = autocorr[np.arange(len(frames)), lag]
    pitch = sample_rate / lag
    pitch[peak < VOICING_THRESHOLD] = np.nan
    return pitch


def analyse(audio, sample_rate=SAMPLE_RATE):
    """Utterance-level summary of the frame features."""
    frames = frame_signal(audio, FRAME_MS, HOP_MS, sample_rate)
    power = power_spectrum(frames)
    loudness = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + EPS)
    active = loudness >= loudness.max() - ACTIVE_RANGE_DB
    coefficients = mfcc(power)

    pitch_frames = frame_signal(audio, PITCH_FRAME_MS, HOP_MS, sample_rate)
    pitch_loudness = 10.0 * np.log10(np.mean(pitch_frames ** 2, axis=1) + EPS)
    pitch = frame_pitch(pitch_frames, sample_rate)
    pitch[pitch_loudness < pitch_loudness.max() - ACTIVE_RANGE_DB] = np.nan
    voiced = pitch[~np.isnan(pitch)]
    if len(voiced) >= 3:
        median_pitch = float(np.median(voiced))
        semitones = 12.0 * np.log2(voiced / median_pitch)
        # Median absolute deviation (scaled to match a standard deviation) ignores occasional octave errors.
        pitch_sd = float(1.4826 * np.median(np.abs(semitones)))
        pitch_range = float(np.percentile(semitones, 90) - np.percentile(semitones, 10))
    else:
        median_pitch, pitch_sd, pitch_range = float("nan"), 0.0, 0.0

    return {
        "duration_s": len(audio) / sample_rate,
        "loudness_db": float(np.percentile(loudness, 95)),
        "loudness_sd_db": float(np.std(loudness[active])),
        "pitch_median_hz": median_pitch,
        "pitch_sd_semitones": pitch_sd,
        "pitch_range_semitones": pitch_range,
        "voiced_ratio": len(voiced) / len(pitch),
        "flatness": float(np.median(spectral_flatness(power)[active])),
        "energy_range_db": float(np.percentile(loudness, 95) - np.percentile(loudness, 10)),
        "mfcc_mean": coefficients[active].mean(axis=0),
        "mfcc_sd": coefficients[active].std(axis=0),
    }
