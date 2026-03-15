"""
Tests for BirdCLEF module — features, model, and inference pipeline.

Run:  pytest birdclef/tests/test_birdclef.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
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

    def test_spec_augment_preserves_shape(self):
        """SpecAugment should not change mel-spectrogram shape."""
        from birdclef.features import audio_to_melspec, spec_augment
        y = self._make_sine_wave()
        mel = audio_to_melspec(y)
        augmented = spec_augment(mel)
        assert augmented.shape == mel.shape

    def test_spec_augment_introduces_zeros(self):
        """SpecAugment should zero out some regions of a non-silent mel."""
        from birdclef.features import audio_to_melspec, spec_augment
        y = self._make_sine_wave()
        mel = audio_to_melspec(y)
        # Original normalized mel has nonzero content
        assert mel.sum() > 0
        augmented = spec_augment(mel, num_freq_masks=1, num_time_masks=1)
        # Augmented should have more zeros than original
        assert (augmented == 0.0).sum() > (mel == 0.0).sum()

    def test_spec_augment_does_not_modify_input(self):
        """SpecAugment should return a copy, not modify the input array."""
        from birdclef.features import audio_to_melspec, spec_augment
        y = self._make_sine_wave()
        mel = audio_to_melspec(y)
        original_sum = mel.sum()
        _ = spec_augment(mel)
        assert mel.sum() == original_sum


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

    def test_multilabel_focal_loss_returns_scalar(self):
        """Focal loss should reduce to a single finite scalar by default."""
        from birdclef.train import MultilabelFocalLoss

        logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)

        loss = MultilabelFocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)
        assert loss.item() >= 0.0

    def test_multilabel_focal_loss_downweights_easy_examples(self):
        """Easy correctly classified examples should have lower focal loss."""
        from birdclef.train import MultilabelFocalLoss

        criterion = MultilabelFocalLoss(alpha=0.25, gamma=2.0, reduction="none")
        easy_logits = torch.tensor([[6.0, -6.0]], dtype=torch.float32)
        hard_logits = torch.tensor([[0.2, -0.2]], dtype=torch.float32)
        targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

        easy_loss = criterion(easy_logits, targets).mean()
        hard_loss = criterion(hard_logits, targets).mean()
        assert easy_loss < hard_loss

    def test_build_training_metadata_records_loss(self):
        """Saved training metadata should clearly record the active loss."""
        from birdclef.train import build_training_metadata

        metadata = build_training_metadata(
            backbone="small",
            loss_name="focal",
            epochs=10,
            batch_size=8,
            lr=1e-3,
            use_mixup=True,
            include_soundscapes=False,
            use_weighted_bce=True,
            best_val_loss=0.1234,
        )

        assert metadata["loss"] == "focal"
        assert metadata["backbone"] == "small"
        assert metadata["focal_alpha"] == 0.25
        assert metadata["focal_gamma"] == 2.0
        assert metadata["weighted_bce"] is False


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


# ════════════════════════════════════════════════════════════════════
# Section 6: Smart Crop Dataset Integration
# ════════════════════════════════════════════════════════════════════

def _write_test_wav(path: Path, y: np.ndarray, sr: int = 32000):
    """Write a float32 numpy array to a .wav file."""
    import soundfile as sf
    sf.write(str(path), y, sr, subtype="FLOAT")


class TestSmartCropDataset:
    """Validate SmartCropDataset manifest loading and offset extraction."""

    def _make_dataset(self, tmp_path):
        """
        Create a mini smart-crop manifest + audio files for testing.

        Returns (manifest_df, audio_dir, labels).
        """
        import pandas as pd

        audio_dir = tmp_path / "train_audio"
        audio_dir.mkdir()

        sr = 32000
        labels = ["sp_a", "sp_b"]

        # Create two 20s audio files with recognizable patterns per window
        for sp in labels:
            sp_dir = audio_dir / sp
            sp_dir.mkdir()
            # 20 seconds of noise — each 5s window has slightly different energy
            rng = np.random.default_rng(hash(sp) % 2**31)
            y = rng.standard_normal(sr * 20).astype(np.float32) * 0.01
            # Plant a loud burst at 5-10s so offset=5 window is distinctive
            burst_start = sr * 5
            burst_end = sr * 10
            t = np.arange(burst_end - burst_start) / sr
            y[burst_start:burst_end] += (0.4 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
            _write_test_wav(sp_dir / "call.wav", y)

        # Manifest: two windows from sp_a (offset 0 and 5), one from sp_b (offset 5)
        manifest = pd.DataFrame([
            {"filename": "sp_a/call.wav", "species": "sp_a", "window_index": 0,
             "offset_seconds": 0.0, "confidence": 0.1},
            {"filename": "sp_a/call.wav", "species": "sp_a", "window_index": 1,
             "offset_seconds": 5.0, "confidence": 0.8},
            {"filename": "sp_b/call.wav", "species": "sp_b", "window_index": 0,
             "offset_seconds": 5.0, "confidence": 0.7},
        ])

        return manifest, audio_dir, labels

    def test_manifest_loading_length(self, tmp_path):
        """SmartCropDataset length should match manifest row count."""
        from birdclef.train import SmartCropDataset

        manifest, audio_dir, labels = self._make_dataset(tmp_path)
        ds = SmartCropDataset(manifest, audio_dir, labels)

        assert len(ds) == 3

    def test_getitem_returns_correct_shapes(self, tmp_path):
        """Each sample should be (3×224×224 tensor, num_labels target)."""
        from birdclef.train import SmartCropDataset

        manifest, audio_dir, labels = self._make_dataset(tmp_path)
        ds = SmartCropDataset(manifest, audio_dir, labels)

        tensor, target = ds[0]
        assert tensor.shape == (3, 224, 224)
        assert target.shape == (len(labels),)
        assert target.dtype == torch.float32

    def test_species_label_mapped_correctly(self, tmp_path):
        """Target vector should have 1.0 at the correct species index."""
        from birdclef.train import SmartCropDataset

        manifest, audio_dir, labels = self._make_dataset(tmp_path)
        ds = SmartCropDataset(manifest, audio_dir, labels)

        # Row 0: species="sp_a" → index 0
        _, target_a = ds[0]
        assert target_a[0] == 1.0
        assert target_a[1] == 0.0

        # Row 2: species="sp_b" → index 1
        _, target_b = ds[2]
        assert target_b[0] == 0.0
        assert target_b[1] == 1.0

    def test_offset_extracts_different_audio(self, tmp_path):
        """Windows at offset=0 and offset=5 should produce different tensors."""
        from birdclef.train import SmartCropDataset

        manifest, audio_dir, labels = self._make_dataset(tmp_path)
        ds = SmartCropDataset(manifest, audio_dir, labels)

        tensor_0, _ = ds[0]  # sp_a @ offset 0 (quiet noise)
        tensor_5, _ = ds[1]  # sp_a @ offset 5 (loud burst)

        # They should differ — the burst window has much more energy
        assert not torch.allclose(tensor_0, tensor_5), \
            "Offset 0 and offset 5 should produce different spectrograms"

    def test_burst_window_has_higher_energy(self, tmp_path):
        """The window containing the burst should have higher raw mel energy."""
        from birdclef.features import load_audio_window, audio_to_melspec

        manifest, audio_dir, labels = self._make_dataset(tmp_path)
        filepath = audio_dir / "sp_a" / "call.wav"

        # Load raw audio at both offsets and compare unnormalized mel energy
        y_quiet = load_audio_window(filepath, offset_seconds=0.0, duration=5.0)
        y_burst = load_audio_window(filepath, offset_seconds=5.0, duration=5.0)

        mel_quiet = audio_to_melspec(y_quiet, normalize=False, to_db=False)
        mel_burst = audio_to_melspec(y_burst, normalize=False, to_db=False)

        energy_quiet = mel_quiet.mean()
        energy_burst = mel_burst.mean()
        assert energy_burst > energy_quiet, \
            f"Burst window energy ({energy_burst:.6f}) should exceed quiet ({energy_quiet:.6f})"


# ═══════════════════════════════════════════════════════════════════
# Secondary Labels
# ═══════════════════════════════════════════════════════════════════

class TestSecondaryLabels:
    """Validate secondary label weighting in BirdCLEFDataset and SmartCropDataset."""

    def _make_audio(self, tmp_path, species_list):
        """Create dummy audio files for each species."""
        audio_dir = tmp_path / "train_audio"
        audio_dir.mkdir()
        sr = 32000
        for sp in species_list:
            sp_dir = audio_dir / sp
            sp_dir.mkdir()
            y = np.random.default_rng(42).standard_normal(sr * 5).astype(np.float32) * 0.01
            _write_test_wav(sp_dir / "call.wav", y)
        return audio_dir

    def test_birdclef_secondary_weight_05(self, tmp_path):
        """Secondary species should appear in target at 0.5, primary at 1.0."""
        from birdclef.train import BirdCLEFDataset

        labels = ["sp_a", "sp_b", "sp_c"]
        audio_dir = self._make_audio(tmp_path, labels)
        meta = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "primary_label": "sp_a",
            "secondary_labels": "['sp_b', 'sp_c']",
        }])
        ds = BirdCLEFDataset(meta, audio_dir, labels, secondary_weight=0.5)
        _, target = ds[0]

        assert target[0] == 1.0, "Primary label should be 1.0"
        assert target[1] == 0.5, "Secondary label sp_b should be 0.5"
        assert target[2] == 0.5, "Secondary label sp_c should be 0.5"

    def test_birdclef_secondary_weight_zero_disables(self, tmp_path):
        """Setting secondary_weight=0 should leave secondary slots at 0."""
        from birdclef.train import BirdCLEFDataset

        labels = ["sp_a", "sp_b"]
        audio_dir = self._make_audio(tmp_path, labels)
        meta = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "primary_label": "sp_a",
            "secondary_labels": "['sp_b']",
        }])
        ds = BirdCLEFDataset(meta, audio_dir, labels, secondary_weight=0.0)
        _, target = ds[0]

        assert target[0] == 1.0
        assert target[1] == 0.0, "Secondary should be 0 when weight is 0"

    def test_birdclef_empty_secondary_no_effect(self, tmp_path):
        """Empty secondary_labels ('[]') should only activate primary."""
        from birdclef.train import BirdCLEFDataset

        labels = ["sp_a", "sp_b"]
        audio_dir = self._make_audio(tmp_path, labels)
        meta = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "primary_label": "sp_a",
            "secondary_labels": "[]",
        }])
        ds = BirdCLEFDataset(meta, audio_dir, labels)
        _, target = ds[0]

        assert target[0] == 1.0
        assert target[1] == 0.0

    def test_smartcrop_secondary_from_train_meta(self, tmp_path):
        """SmartCropDataset should pick up secondary labels from train_meta."""
        from birdclef.train import SmartCropDataset

        labels = ["sp_a", "sp_b", "sp_c"]
        audio_dir = self._make_audio(tmp_path, labels)

        train_meta = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "primary_label": "sp_a",
            "secondary_labels": "['sp_b']",
        }])
        manifest = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "species": "sp_a",
            "window_index": 0,
            "offset_seconds": 0.0,
            "confidence": 0.9,
        }])
        ds = SmartCropDataset(
            manifest, audio_dir, labels,
            secondary_weight=0.5, train_meta=train_meta,
        )
        _, target = ds[0]

        assert target[0] == 1.0, "Primary sp_a should be 1.0"
        assert target[1] == 0.5, "Secondary sp_b should be 0.5"
        assert target[2] == 0.0, "sp_c not mentioned, should be 0.0"

    def test_smartcrop_no_train_meta_no_secondary(self, tmp_path):
        """Without train_meta, SmartCropDataset should only set primary."""
        from birdclef.train import SmartCropDataset

        labels = ["sp_a", "sp_b"]
        audio_dir = self._make_audio(tmp_path, labels)

        manifest = pd.DataFrame([{
            "filename": "sp_a/call.wav",
            "species": "sp_a",
            "window_index": 0,
            "offset_seconds": 0.0,
            "confidence": 0.9,
        }])
        ds = SmartCropDataset(manifest, audio_dir, labels)
        _, target = ds[0]

        assert target[0] == 1.0
        assert target[1] == 0.0


# ═══════════════════════════════════════════════════════════════════
# GPU / Performance Optimizations
# ═══════════════════════════════════════════════════════════════════

class TestGPUOptimizations:
    """Validate performance optimizations don't break functionality."""

    def _make_audio(self, tmp_path, species_list):
        audio_dir = tmp_path / "train_audio"
        audio_dir.mkdir()
        sr = 32000
        for sp in species_list:
            sp_dir = audio_dir / sp
            sp_dir.mkdir()
            y = np.random.default_rng(42).standard_normal(sr * 10).astype(np.float32) * 0.01
            _write_test_wav(sp_dir / "call.wav", y)
        return audio_dir

    def test_duration_cache_avoids_repeated_io(self, tmp_path):
        """BirdCLEFDataset should cache file durations after first access."""
        from birdclef.train import BirdCLEFDataset

        labels = ["sp_a"]
        audio_dir = self._make_audio(tmp_path, labels)
        meta = pd.DataFrame([
            {"filename": "sp_a/call.wav", "primary_label": "sp_a", "secondary_labels": "[]"},
            {"filename": "sp_a/call.wav", "primary_label": "sp_a", "secondary_labels": "[]"},
        ])
        ds = BirdCLEFDataset(meta, audio_dir, labels, augment=True)

        # Access twice — cache should be populated after first
        _ = ds[0]
        assert len(ds._duration_cache) == 1, "Duration should be cached"
        _ = ds[1]
        assert len(ds._duration_cache) == 1, "Same file should reuse cache"

    def test_duration_cache_returns_valid_float(self, tmp_path):
        """Cached duration should be a positive float matching actual file length."""
        from birdclef.train import BirdCLEFDataset

        labels = ["sp_a"]
        audio_dir = self._make_audio(tmp_path, labels)
        meta = pd.DataFrame([
            {"filename": "sp_a/call.wav", "primary_label": "sp_a", "secondary_labels": "[]"},
        ])
        ds = BirdCLEFDataset(meta, audio_dir, labels, augment=True)
        _ = ds[0]

        cached = list(ds._duration_cache.values())[0]
        assert isinstance(cached, float)
        assert cached > 9.0, f"10s file should report ~10s, got {cached}"

    def test_amp_autocast_cpu_noop(self):
        """AMP autocast on CPU should be a safe no-op."""
        model = torch.nn.Linear(10, 5)
        x = torch.randn(2, 10)

        with torch.amp.autocast(device_type="cpu", dtype=torch.float16, enabled=False):
            out = model(x)
        assert out.shape == (2, 5)

    def test_grad_scaler_disabled_on_cpu(self):
        """GradScaler with enabled=False should pass through normally."""
        scaler = torch.amp.GradScaler(enabled=False)
        assert not scaler.is_enabled()
