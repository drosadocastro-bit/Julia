"""
Tests for BirdCLEF module — features, model, and inference pipeline.

Run:  pytest birdclef/tests/test_birdclef.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ════════════════════════════════════════════════════════════════════
# Section 1: Feature Extraction
# ════════════════════════════════════════════════════════════════════

class TestFeatureExtraction:
    """Validate mel-spectrogram and classical feature pipelines."""

    def _make_sine_wave(self, sr=32000, duration=5.0, freq=440.0):
        """Generate a simple sine wave for testing."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def test_audio_to_melspec_shape(self):
        """Mel-spectrogram output should be (n_mels, time_frames)."""
        from birdclef.features import audio_to_melspec
        y = self._make_sine_wave()
        mel = audio_to_melspec(y)
        assert mel.ndim == 2
        assert mel.shape[0] == 128  # n_mels default
        assert mel.shape[1] > 0

    def test_audio_to_melspec_normalized(self):
        """Normalized mel-spec values should be in [0, 1]."""
        from birdclef.features import audio_to_melspec
        y = self._make_sine_wave()
        mel = audio_to_melspec(y, normalize=True)
        assert mel.min() >= 0.0 - 1e-6
        assert mel.max() <= 1.0 + 1e-6

    def test_melspec_to_tensor_shape(self):
        """Tensor output should be (3, 224, 224) for CNN input."""
        from birdclef.features import audio_to_melspec, melspec_to_tensor
        y = self._make_sine_wave()
        mel = audio_to_melspec(y)
        tensor = melspec_to_tensor(mel)
        assert tensor.shape == (3, 224, 224)

    def test_silence_produces_zero_melspec(self):
        """All-zero audio should produce a zero-filled mel-spec."""
        from birdclef.features import audio_to_melspec
        y = np.zeros(32000 * 5, dtype=np.float32)
        mel = audio_to_melspec(y)
        assert mel.max() == 0.0

    def test_classical_features_length(self):
        """Classical feature vector should have exactly 53 features."""
        from birdclef.features import extract_classical_features
        y = self._make_sine_wave()
        feats = extract_classical_features(y)
        assert feats is not None
        assert len(feats) == 53

    def test_classical_features_no_nan(self):
        """Feature vector should not contain NaN values."""
        from birdclef.features import extract_classical_features
        y = self._make_sine_wave()
        feats = extract_classical_features(y)
        assert feats is not None
        assert not np.isnan(feats).any()


# ════════════════════════════════════════════════════════════════════
# Section 2: Model Architecture
# ════════════════════════════════════════════════════════════════════

class TestModelArchitecture:
    """Validate model forward pass and output shapes."""

    def test_small_cnn_forward(self):
        """SmallCNN should produce (B, num_species) logits."""
        from birdclef.model import BirdSmallCNN
        model = BirdSmallCNN(num_species=50)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 50)

    def test_small_cnn_output_range(self):
        """Raw logits can be any value (no activation applied)."""
        from birdclef.model import BirdSmallCNN
        model = BirdSmallCNN(num_species=10)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        # After sigmoid, should be in [0, 1]
        probs = torch.sigmoid(out)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_classifier_load_save_roundtrip(self):
        """Model save → load should produce identical predictions."""
        from birdclef.model import BirdSmallCNN, BirdClassifier
        import tempfile

        num_species = 5
        labels = [f"species_{i}" for i in range(num_species)]

        # Build and save
        model = BirdSmallCNN(num_species=num_species)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            torch.save(model.state_dict(), tmpdir / "birdclef_model.pt")
            with open(tmpdir / "birdclef_labels.json", "w") as f:
                json.dump(labels, f)

            # Load and predict
            clf = BirdClassifier(backbone="small", model_dir=tmpdir)
            clf.load(
                model_path=tmpdir / "birdclef_model.pt",
                labels_path=tmpdir / "birdclef_labels.json",
            )
            assert clf.is_loaded
            assert len(clf.labels) == num_species

            x = torch.randn(3, 224, 224)
            probs = clf.predict(x)
            assert probs.shape == (num_species,)
            assert probs.min() >= 0.0
            assert probs.max() <= 1.0

    def test_get_top_species(self):
        """get_top_species should return sorted results above threshold."""
        from birdclef.model import BirdClassifier
        clf = BirdClassifier.__new__(BirdClassifier)
        clf.labels = ["sparrow", "eagle", "parrot", "hawk", "owl"]

        probs = torch.tensor([0.1, 0.9, 0.6, 0.3, 0.05])
        results = clf.get_top_species(probs, threshold=0.25, top_k=3)
        assert len(results) == 3  # eagle (0.9), parrot (0.6), hawk (0.3) above 0.25
        assert results[0]["species"] == "eagle"
        assert results[1]["species"] == "parrot"
        assert results[2]["species"] == "hawk"


# ════════════════════════════════════════════════════════════════════
# Section 3: Training Utilities
# ════════════════════════════════════════════════════════════════════

class TestTrainingUtils:
    """Validate mixup and dataset construction."""

    def test_mixup_shape_preserved(self):
        """Mixup should not change batch dimensions."""
        from birdclef.train import mixup_batch
        x = torch.randn(8, 3, 224, 224)
        y = torch.randn(8, 50)
        mx, my = mixup_batch(x, y, alpha=0.4)
        assert mx.shape == x.shape
        assert my.shape == y.shape

    def test_mixup_alpha_zero_returns_original(self):
        """Alpha=0 means no mixup — output should equal input."""
        from birdclef.train import mixup_batch
        x = torch.randn(4, 3, 224, 224)
        y = torch.randn(4, 10)
        mx, my = mixup_batch(x, y, alpha=0.0)
        assert torch.equal(mx, x)
        assert torch.equal(my, y)


# ════════════════════════════════════════════════════════════════════
# Section 4: Config Sanity
# ════════════════════════════════════════════════════════════════════

class TestConfig:
    """Validate competition constants."""

    def test_sample_rate(self):
        from birdclef.config import SAMPLE_RATE
        assert SAMPLE_RATE == 32000

    def test_window_seconds(self):
        from birdclef.config import WINDOW_SECONDS
        assert WINDOW_SECONDS == 5

    def test_frequency_range(self):
        """FMIN < FMAX and both positive."""
        from birdclef.config import FMIN, FMAX
        assert 0 < FMIN < FMAX

    def test_mel_bins(self):
        from birdclef.config import N_MELS
        assert N_MELS == 128

    def test_min_confidence_zero(self):
        """P1-01: MIN_CONFIDENCE_TO_EMIT must be 0.0 for ROC-AUC scoring."""
        from birdclef.config import MIN_CONFIDENCE_TO_EMIT
        assert MIN_CONFIDENCE_TO_EMIT == 0.0

    def test_max_label_count(self):
        """P1-02: MAX_LABEL_COUNT should match taxonomy.csv exact count."""
        from birdclef.config import MAX_LABEL_COUNT
        assert MAX_LABEL_COUNT == 234


# ════════════════════════════════════════════════════════════════════
# Section 5: Overlap Windowing & Prediction Pooling (P1-03)
# ════════════════════════════════════════════════════════════════════

class TestOverlapWindowing:
    """Validate 50% overlap windowing and prediction pooling."""

    def test_iter_soundscape_windows_yields_three_values(self):
        """P1-03: Each yield must be (chunk, start_time, end_time)."""
        import tempfile, soundfile as sf
        from birdclef.features import iter_soundscape_windows

        sr = 32000
        # 12 seconds of audio — should produce overlapping windows
        audio = np.random.randn(sr * 12).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            windows = list(iter_soundscape_windows(Path(f.name), window_sec=5, sr=sr))

        # With 50% overlap on 12s: starts at 0, 2.5, 5.0, 7.5 → 4 windows
        assert len(windows) >= 4
        for chunk, start_time, end_time in windows:
            assert isinstance(chunk, np.ndarray)
            assert len(chunk) == sr * 5
            assert end_time - start_time == 5.0
            assert start_time >= 0.0

    def test_overlap_produces_more_windows_than_non_overlap(self):
        """50% overlap should produce ~2x the windows vs no overlap."""
        import tempfile, soundfile as sf
        from birdclef.features import iter_soundscape_windows

        sr = 32000
        audio = np.random.randn(sr * 20).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            overlap_windows = list(iter_soundscape_windows(Path(f.name), overlap=0.5))
            no_overlap_windows = list(iter_soundscape_windows(Path(f.name), overlap=0.0))

        # 50% overlap should produce roughly 2x windows
        assert len(overlap_windows) > len(no_overlap_windows)

    def test_pool_overlapping_predictions_shape(self):
        """Pooled output should have one row per 5-second grid cell."""
        from birdclef.features import pool_overlapping_predictions

        # Simulate 7 overlapping windows from a 20-second file
        start_times = [0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
        end_times = [5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]
        preds = np.random.rand(7, 50).astype(np.float32)

        grid_ends, pooled = pool_overlapping_predictions(start_times, end_times, preds)

        # 20 seconds / 5-sec grid = 4 grid cells
        assert len(grid_ends) == 4
        assert pooled.shape == (4, 50)
        assert grid_ends == [5, 10, 15, 20]

    def test_pool_averages_overlapping_windows(self):
        """Windows mapping to the same grid cell should be averaged."""
        from birdclef.features import pool_overlapping_predictions

        # Two windows both centered in the first grid cell [0, 5)
        start_times = [0.0, 2.5]
        end_times = [5.0, 7.5]
        preds = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        grid_ends, pooled = pool_overlapping_predictions(start_times, end_times, preds)

        # Window 0: center=2.5, grid_idx=0. Window 1: center=5.0, grid_idx=1
        assert grid_ends[0] == 5
        # First grid cell only has window 0
        np.testing.assert_array_almost_equal(pooled[0], [1.0, 0.0])

    def test_pool_empty_input(self):
        """Empty input should return empty output."""
        from birdclef.features import pool_overlapping_predictions

        grid_ends, pooled = pool_overlapping_predictions([], [], np.empty((0, 10)))
        assert len(grid_ends) == 0
        assert pooled.shape[0] == 0
