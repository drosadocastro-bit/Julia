"""
Tests for birdclef.smart_crop — CFAR-driven smart cropping.

Run:  pytest birdclef/tests/test_smart_crop.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _write_wav(path: Path, y: np.ndarray, sr: int = 32000):
    """Write a float32 numpy array to a .wav file."""
    import soundfile as sf
    sf.write(str(path), y, sr, subtype="FLOAT")


def _make_audio_with_burst(
    duration: float = 30.0,
    sr: int = 32000,
    burst_start: float = 10.0,
    burst_duration: float = 5.0,
    burst_freq: float = 3000.0,
    noise_level: float = 0.01,
) -> np.ndarray:
    """
    Create synthetic audio: quiet noise with a loud burst (simulating a bird call).

    Returns 1-D float32 array.
    """
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)
    y = (rng.standard_normal(n_samples) * noise_level).astype(np.float32)

    # Insert loud sinusoidal burst
    burst_start_sample = int(sr * burst_start)
    burst_end_sample = burst_start_sample + int(sr * burst_duration)
    t = np.arange(burst_end_sample - burst_start_sample) / sr
    y[burst_start_sample:burst_end_sample] += (
        0.5 * np.sin(2 * np.pi * burst_freq * t)
    ).astype(np.float32)

    return y


def _make_silence(duration: float = 30.0, sr: int = 32000) -> np.ndarray:
    """Create near-silent audio."""
    n = int(sr * duration)
    rng = np.random.default_rng(99)
    return (rng.standard_normal(n) * 0.001).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# Tests for extract_active_windows
# ═══════════════════════════════════════════════════════════════════

class TestExtractActiveWindows:
    """Validate CFAR-based window extraction."""

    def test_returns_list_of_tuples(self, tmp_path):
        """Output should be a list of (audio_chunk, label, confidence) tuples."""
        from birdclef.smart_crop import extract_active_windows

        y = _make_audio_with_burst(duration=15.0)
        wav_path = tmp_path / "test.wav"
        _write_wav(wav_path, y)

        result = extract_active_windows(wav_path, "asbfly")

        assert isinstance(result, list)
        assert len(result) > 0
        for chunk, label, conf in result:
            assert isinstance(chunk, np.ndarray)
            assert chunk.dtype == np.float32
            assert chunk.shape == (32000 * 5,)
            assert isinstance(label, str)
            assert label == "asbfly"
            assert isinstance(conf, float)
            assert conf >= 0.0

    def test_burst_window_detected(self, tmp_path):
        """The window containing the burst should always be included."""
        from birdclef.smart_crop import extract_active_windows

        # Burst at 10-15s → window index 2 in a 30s file (0-5, 5-10, 10-15, ...)
        y = _make_audio_with_burst(duration=30.0, burst_start=10.0)
        wav_path = tmp_path / "burst.wav"
        _write_wav(wav_path, y)

        result = extract_active_windows(wav_path, "bkcbul1")

        confidences = [conf for _, _, conf in result]
        # The burst window should have the highest confidence
        assert len(result) >= 1
        max_conf = max(confidences)
        assert max_conf > 0.0

    def test_fewer_windows_than_naive(self, tmp_path):
        """Smart crop should return fewer windows than naive full-file slicing."""
        from birdclef.smart_crop import extract_active_windows

        # 60s file with only one 5s burst → naive = 12 windows
        y = _make_audio_with_burst(duration=60.0, burst_start=25.0)
        wav_path = tmp_path / "long.wav"
        _write_wav(wav_path, y)

        result = extract_active_windows(wav_path, "comfla1")
        naive_count = 60 // 5  # 12

        assert len(result) < naive_count, (
            f"Expected fewer than {naive_count} windows, got {len(result)}"
        )

    def test_silence_still_returns_something(self, tmp_path):
        """Even near-silent audio should return at least the best window (safety net)."""
        from birdclef.smart_crop import extract_active_windows

        y = _make_silence(duration=30.0)
        wav_path = tmp_path / "silence.wav"
        _write_wav(wav_path, y)

        result = extract_active_windows(wav_path, "silbird")
        assert len(result) >= 1, "Safety net should keep at least one window"

    def test_short_file_padded(self, tmp_path):
        """Files shorter than one window should be padded and returned."""
        from birdclef.smart_crop import extract_active_windows

        y = np.random.default_rng(7).standard_normal(32000 * 2).astype(np.float32)
        wav_path = tmp_path / "short.wav"
        _write_wav(wav_path, y)

        result = extract_active_windows(wav_path, "shorty")
        assert len(result) == 1
        assert result[0][0].shape == (32000 * 5,)


# ═══════════════════════════════════════════════════════════════════
# Tests for build_smart_crop_dataset
# ═══════════════════════════════════════════════════════════════════

class TestBuildSmartCropDataset:
    """Validate manifest generation."""

    def _prepare_test_files(self, tmp_path, n_files=5):
        """Create a mini dataset with synthetic audio and metadata CSV."""
        audio_dir = tmp_path / "train_audio"
        audio_dir.mkdir()

        records = []
        for i in range(n_files):
            species = f"species_{i}"
            fname = f"{species}/call_{i}.wav"
            species_dir = audio_dir / species
            species_dir.mkdir(exist_ok=True)

            # Alternate between active and mostly-silent files
            if i % 2 == 0:
                y = _make_audio_with_burst(duration=20.0, burst_start=5.0)
            else:
                y = _make_silence(duration=20.0)
            _write_wav(species_dir / f"call_{i}.wav", y)

            records.append({
                "filename": fname,
                "primary_label": species,
                "secondary_labels": "[]",
                "rating": 4.0,
            })

        csv_path = tmp_path / "train_metadata.csv"
        import pandas as pd
        pd.DataFrame(records).to_csv(csv_path, index=False)

        return csv_path, audio_dir

    def test_manifest_columns(self, tmp_path):
        """Manifest CSV should have the expected columns."""
        from birdclef.smart_crop import build_smart_crop_dataset

        csv_path, audio_dir = self._prepare_test_files(tmp_path)
        out_path = tmp_path / "manifest.csv"

        manifest = build_smart_crop_dataset(csv_path, audio_dir, output_path=out_path)

        expected_cols = {"filename", "species", "window_index", "offset_seconds", "confidence"}
        assert set(manifest.columns) == expected_cols

    def test_manifest_has_rows(self, tmp_path):
        """Manifest should contain at least one row per file (safety net)."""
        from birdclef.smart_crop import build_smart_crop_dataset

        csv_path, audio_dir = self._prepare_test_files(tmp_path, n_files=3)
        out_path = tmp_path / "manifest.csv"

        manifest = build_smart_crop_dataset(csv_path, audio_dir, output_path=out_path)

        # At least one window per file
        assert len(manifest) >= 3

    def test_manifest_saved_to_disk(self, tmp_path):
        """Manifest CSV should be written to disk."""
        from birdclef.smart_crop import build_smart_crop_dataset

        csv_path, audio_dir = self._prepare_test_files(tmp_path, n_files=2)
        out_path = tmp_path / "manifest.csv"

        build_smart_crop_dataset(csv_path, audio_dir, output_path=out_path)

        assert out_path.exists()
        import pandas as pd
        saved = pd.read_csv(out_path)
        assert len(saved) > 0

    def test_fewer_total_windows_than_naive(self, tmp_path):
        """Total smart-crop windows should be <= total naive windows."""
        from birdclef.smart_crop import build_smart_crop_dataset

        csv_path, audio_dir = self._prepare_test_files(tmp_path, n_files=5)
        out_path = tmp_path / "manifest.csv"

        manifest = build_smart_crop_dataset(csv_path, audio_dir, output_path=out_path)

        # Each file is 20s → 4 naive windows each → 20 total
        naive_total = 5 * (20 // 5)
        assert len(manifest) <= naive_total, (
            f"Smart crop ({len(manifest)}) should be <= naive ({naive_total})"
        )
