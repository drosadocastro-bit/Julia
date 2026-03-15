"""
birdclef.features — Audio feature extraction for bird species classification.

Lineage: Adapted from Project Aria's genre_cnn.py mel-spectrogram pipeline
         and genre_classifier.py MFCC extraction.

Produces fixed-size mel-spectrogram tensors from 5-second audio windows,
suitable for CNN input.  Also provides a classical feature vector path
(MFCC/spectral) for ensemble or lightweight models.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from birdclef.config import (
    SAMPLE_RATE, WINDOW_SECONDS, N_MELS, N_FFT,
    HOP_LENGTH, FMIN, FMAX,
)

logger = logging.getLogger("birdclef.features")


# ═══════════════════════════════════════════════════════════════════
# Mel-Spectrogram Path (Primary — for CNN/SED models)
# ═══════════════════════════════════════════════════════════════════

def load_audio_window(
    filepath: Path,
    offset_seconds: float = 0.0,
    duration: float = WINDOW_SECONDS,
    target_sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Load a single audio window from a file.

    Returns a 1-D float32 array of length (target_sr * duration).
    Pads with zeros if the clip is shorter than the requested duration.
    """
    import librosa

    y, sr = librosa.load(
        filepath,
        sr=target_sr,
        offset=offset_seconds,
        duration=duration,
        mono=True,
    )

    expected_len = int(target_sr * duration)
    if len(y) < expected_len:
        y = np.pad(y, (0, expected_len - len(y)), mode="constant")
    elif len(y) > expected_len:
        y = y[:expected_len]

    return y.astype(np.float32)


def audio_to_melspec(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmin: float = FMIN,
    fmax: float = FMAX,
    to_db: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert raw waveform to a mel-spectrogram (2-D numpy array).

    Returns shape (n_mels, time_frames).
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, fmin=fmin, fmax=fmax,
    )

    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)

    if normalize:
        mel_min = mel.min()
        mel_max = mel.max()
        if mel_max - mel_min > 1e-8:
            mel = (mel - mel_min) / (mel_max - mel_min)
        else:
            mel = np.zeros_like(mel)

    return mel.astype(np.float32)


def melspec_to_tensor(mel: np.ndarray) -> "torch.Tensor":
    """
    Convert a 2-D mel-spectrogram to a 3-channel image tensor
    suitable for pretrained CNN backbones (shape: 3 × 224 × 224).

    Mirrors Aria's genre_cnn.predict_audio approach.
    """
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(mel).unsqueeze(0)       # (1, n_mels, time)
    t = t.repeat(3, 1, 1)                         # (3, n_mels, time)
    t = F.interpolate(
        t.unsqueeze(0), size=(224, 224),
        mode="bilinear", align_corners=False,
    ).squeeze(0)                                   # (3, 224, 224)

    return t


def spec_augment(
    mel: np.ndarray,
    num_freq_masks: int = 1,
    freq_mask_width: int = 15,
    num_time_masks: int = 1,
    time_mask_width: int = 25,
) -> np.ndarray:
    """
    Apply SpecAugment (frequency + time masking) to a mel-spectrogram.

    Conservative defaults tuned for ~33K weakly-labeled bird audio:
      - 1 freq mask of up to 15 bins (12% of 128 mel bins)
      - 1 time mask of up to 25 frames (8% of ~313 frames)

    Args:
        mel:             2-D array of shape (n_mels, time_frames).
        num_freq_masks:  Number of frequency masks to apply.
        freq_mask_width: Maximum width of each frequency mask (in mel bins).
        num_time_masks:  Number of time masks to apply.
        time_mask_width: Maximum width of each time mask (in frames).

    Returns:
        Augmented mel-spectrogram (same shape, in-place safe).
    """
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    rng = np.random.default_rng()

    # Frequency masking
    for _ in range(num_freq_masks):
        width = rng.integers(1, min(freq_mask_width, n_mels) + 1)
        start = rng.integers(0, n_mels - width + 1)
        mel[start:start + width, :] = 0.0

    # Time masking
    for _ in range(num_time_masks):
        width = rng.integers(1, min(time_mask_width, n_frames) + 1)
        start = rng.integers(0, n_frames - width + 1)
        mel[:, start:start + width] = 0.0

    return mel


# ═══════════════════════════════════════════════════════════════════
# Classical Feature Path (Secondary — for RF / XGBoost ensembles)
# ═══════════════════════════════════════════════════════════════════

def extract_classical_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> Optional[np.ndarray]:
    """
    Extract a flat feature vector compatible with traditional ML classifiers.

    Feature set (adapted from Aria's LiveAudioAnalyzer):
        - Chroma STFT (mean, var)
        - RMS energy (mean, var)
        - Spectral centroid, bandwidth, rolloff (mean, var each)
        - Zero-crossing rate (mean, var)
        - MFCCs 1–20 (mean, var each)
        - Tempo

    Total: 51 features.
    """
    import librosa

    try:
        features = []

        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([chroma.mean(), chroma.var()])

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features.extend([rms.mean(), rms.var()])

        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([cent.mean(), cent.var()])

        # Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([bw.mean(), bw.var()])

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([rolloff.mean(), rolloff.var()])

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features.extend([zcr.mean(), zcr.var()])

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(np.atleast_1d(tempo)[0]))

        # MFCCs (20 coefficients, mean + var each)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features.extend([mfccs[i].mean(), mfccs[i].var()])

        return np.array(features, dtype=np.float32)

    except Exception as e:
        logger.warning(f"Classical feature extraction failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
# Soundscape Utilities (for slicing test files into 5-sec windows)
# ═══════════════════════════════════════════════════════════════════

def iter_soundscape_windows(
    filepath: Path,
    window_sec: float = WINDOW_SECONDS,
    sr: int = SAMPLE_RATE,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, float, float]:
    """
    Yield (audio_window, start_time_seconds, end_time_seconds) for each
    overlapping chunk in a long soundscape recording.

    Uses 50% overlap by default (hop = window_sec / 2 = 2.5s) so that
    bird vocalizations at window boundaries are captured by at least
    one fully-centered window.

    BirdCLEF submission expects predictions at 5-second boundaries;
    use pool_overlapping_predictions() to collapse overlapping windows
    back to the required grid.
    """
    import librosa

    y, _ = librosa.load(filepath, sr=sr, mono=True)
    total_samples = len(y)
    window_samples = int(sr * window_sec)
    hop_sec = window_sec * (1.0 - overlap)
    hop_samples = int(sr * hop_sec)

    start = 0
    while start < total_samples:
        end = start + window_samples
        chunk = y[start:end]

        # Pad the last chunk if needed
        if len(chunk) < window_samples:
            chunk = np.pad(chunk, (0, window_samples - len(chunk)), mode="constant")

        start_time = start / sr
        end_time = (start + window_samples) / sr
        yield chunk.astype(np.float32), start_time, end_time

        start += hop_samples


def pool_overlapping_predictions(
    start_times: list,
    end_times: list,
    predictions: np.ndarray,
    grid_sec: float = WINDOW_SECONDS,
) -> Tuple[list, np.ndarray]:
    """
    Collapse overlapping-window predictions back to a non-overlapping
    grid aligned to BirdCLEF's 5-second submission boundaries.

    For each grid cell [0–5s, 5–10s, ...], averages the sigmoid outputs
    from all overlapping windows whose center falls within that cell.

    Args:
        start_times:  List of window start times (seconds).
        end_times:    List of window end times (seconds).
        predictions:  Array of shape (num_windows, num_species).
        grid_sec:     Submission grid resolution (default 5s).

    Returns:
        (grid_end_times, pooled_predictions)
        grid_end_times: list of int end-times [5, 10, 15, ...]
        pooled_predictions: array of shape (num_grid_cells, num_species)
    """
    if len(start_times) == 0:
        return [], np.empty((0, predictions.shape[1] if predictions.ndim == 2 else 0))

    max_end = max(end_times)
    num_grid_cells = int(np.ceil(max_end / grid_sec))
    num_species = predictions.shape[1]

    accum = np.zeros((num_grid_cells, num_species), dtype=np.float64)
    counts = np.zeros(num_grid_cells, dtype=np.float64)

    for i, (s, e) in enumerate(zip(start_times, end_times)):
        center = (s + e) / 2.0
        grid_idx = int(center / grid_sec)
        # Clamp to valid range
        grid_idx = min(grid_idx, num_grid_cells - 1)
        accum[grid_idx] += predictions[i]
        counts[grid_idx] += 1.0

    # Avoid division by zero for grid cells with no overlapping windows
    mask = counts > 0
    accum[mask] /= counts[mask, np.newaxis]

    grid_end_times = [int((i + 1) * grid_sec) for i in range(num_grid_cells)]

    return grid_end_times, accum.astype(np.float32)
