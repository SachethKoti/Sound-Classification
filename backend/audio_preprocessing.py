"""Audio validation + preprocessing.

Steps: validate -> decode -> mono -> resample to 16 kHz -> pick a 10 s window
-> extract log-mel (filterbank) features exactly as the pretrained model expects.
"""
import os
import tempfile

import librosa
import numpy as np

from . import config


class AudioError(Exception):
    """Raised for any problem with the uploaded audio (shown to the user)."""


def validate_upload(filename: str, data: bytes) -> str:
    if not data:
        raise AudioError("The uploaded file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise AudioError(f"File is too large (limit {config.MAX_UPLOAD_MB} MB).")
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        raise AudioError(f"Unsupported file type '{ext or 'unknown'}'. Allowed: {allowed}.")
    return ext


def load_audio(data: bytes, ext: str) -> np.ndarray:
    """Decode any supported format -> mono float32 waveform at 16 kHz."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        # librosa converts to mono and resamples in one go
        waveform, _ = librosa.load(tmp_path, sr=config.TARGET_SAMPLE_RATE, mono=True)
    except Exception as exc:  # decoder errors vary a lot between formats
        raise AudioError(
            "Could not decode this audio file. It may be corrupted, or your system "
            "may need FFmpeg installed for this format (see README)."
        ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise AudioError("The audio contains no usable samples.")
    return waveform.astype(np.float32)


def check_audio_quality(waveform: np.ndarray) -> float:
    """Return the duration in seconds, or raise AudioError."""
    sr = config.TARGET_SAMPLE_RATE
    duration = len(waveform) / sr
    if duration < config.MIN_SECONDS:
        raise AudioError("The audio is too short (needs at least half a second).")
    if duration > config.MAX_DECODE_SECONDS:
        raise AudioError("The audio is too long (limit 10 minutes).")
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    if rms < config.SILENCE_RMS_THRESHOLD:
        raise AudioError("The audio appears to be silent.")
    return duration


def select_window(waveform: np.ndarray):
    """The model takes ~10 s. For longer clips, use the loudest 10 s window."""
    sr = config.TARGET_SAMPLE_RATE
    win = int(config.WINDOW_SECONDS * sr)
    if len(waveform) <= win:
        return waveform, 0.0, len(waveform) / sr
    hop = sr  # 1 second steps
    best_start, best_energy = 0, -1.0
    for start in range(0, len(waveform) - win + 1, hop):
        energy = float(np.mean(waveform[start:start + win] ** 2))
        if energy > best_energy:
            best_start, best_energy = start, energy
    segment = waveform[best_start:best_start + win]
    return segment, best_start / sr, config.WINDOW_SECONDS


def extract_features(segment: np.ndarray, feature_extractor):
    """Model-specific features (128-bin log-mel fbank, padded/cropped to 1024 frames,
    normalised). Done by the model's own feature extractor so it matches training."""
    return feature_extractor(
        segment,
        sampling_rate=config.TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )
