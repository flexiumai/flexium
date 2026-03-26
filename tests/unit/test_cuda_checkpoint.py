"""Tests for cuda_checkpoint utility module."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import requires_nvidia_driver

from flexium.utils.cuda_checkpoint import (
    find_cuda_checkpoint,
    get_bundled_path,
    get_driver_version,
    get_cuda_checkpoint_version,
    get_file_hash,
    verify_cuda_checkpoint,
    ensure_cuda_checkpoint,
    download_cuda_checkpoint,
    supports_migration,
    get_capabilities,
    CudaCheckpointError,
    CUDA_CHECKPOINT_URL,
    MIN_DRIVER_VERSION,
    MIGRATION_DRIVER_VERSION,
)


class TestGetDriverVersion:
    """Tests for get_driver_version()."""

    def test_returns_integer(self):
        """Driver version should be an integer."""
        version = get_driver_version()
        # May be None if no NVIDIA driver, but if present should be int
        if version is not None:
            assert isinstance(version, int)
            assert version > 0

    def test_fallback_logic_exists(self):
        """Verify fallback logic is implemented."""
        # Just verify the function handles both pynvml and nvidia-smi paths
        # by checking it doesn't crash
        version = get_driver_version()
        # Should return int or None, not raise
        assert version is None or isinstance(version, int)


class TestFindCudaCheckpoint:
    """Tests for find_cuda_checkpoint()."""

    def test_finds_in_path(self):
        """Should find cuda-checkpoint if in PATH."""
        result = find_cuda_checkpoint()
        # May be None if not installed
        if result is not None:
            assert isinstance(result, Path)
            assert result.exists()
            assert os.access(result, os.X_OK)

    def test_returns_path_object(self):
        """Should return Path object or None."""
        result = find_cuda_checkpoint()
        assert result is None or isinstance(result, Path)


class TestVerifyCudaCheckpoint:
    """Tests for verify_cuda_checkpoint()."""

    @requires_nvidia_driver
    def test_verify_existing(self):
        """Should verify existing cuda-checkpoint."""
        path = find_cuda_checkpoint()
        if path is not None:
            assert verify_cuda_checkpoint(path) is True

    def test_verify_nonexistent(self):
        """Should return False for nonexistent path."""
        assert verify_cuda_checkpoint(Path("/nonexistent/cuda-checkpoint")) is False


class TestGetCudaCheckpointVersion:
    """Tests for get_cuda_checkpoint_version()."""

    @requires_nvidia_driver
    def test_get_version_existing(self):
        """Should get version from existing cuda-checkpoint."""
        path = find_cuda_checkpoint()
        if path is not None:
            version = get_cuda_checkpoint_version(path)
            assert version is not None
            # Version format like "590.48.01"
            parts = version.split(".")
            assert len(parts) >= 2
            assert parts[0].isdigit()

    def test_get_version_nonexistent(self):
        """Should return None for nonexistent path."""
        assert get_cuda_checkpoint_version(Path("/nonexistent/cuda-checkpoint")) is None


class TestEnsureCudaCheckpoint:
    """Tests for ensure_cuda_checkpoint()."""

    def test_returns_path_when_available(self):
        """Should return path when cuda-checkpoint is available."""
        # Skip if driver not available
        driver_version = get_driver_version()
        if driver_version is None or driver_version < MIN_DRIVER_VERSION:
            pytest.skip("NVIDIA driver 580+ required")

        path = ensure_cuda_checkpoint()
        assert isinstance(path, Path)
        assert path.exists()

    @patch("flexium.utils.cuda_checkpoint.get_driver_version")
    def test_raises_on_missing_driver(self, mock_driver):
        """Should raise error if driver not detected."""
        mock_driver.return_value = None

        # Clear cached path to force re-check
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None

        with pytest.raises(CudaCheckpointError, match="Could not detect"):
            ensure_cuda_checkpoint()

    @patch("flexium.utils.cuda_checkpoint.get_driver_version")
    def test_raises_on_old_driver(self, mock_driver):
        """Should raise error if driver too old (below 550)."""
        mock_driver.return_value = 540  # Below MIN_DRIVER_VERSION (550)

        # Clear cached path to force re-check
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None

        with pytest.raises(CudaCheckpointError, match="too old"):
            ensure_cuda_checkpoint()


class TestDownloadCudaCheckpoint:
    """Tests for download_cuda_checkpoint()."""

    def test_url_is_valid(self):
        """CUDA_CHECKPOINT_URL should be a valid URL."""
        assert CUDA_CHECKPOINT_URL.startswith("https://")
        assert "github" in CUDA_CHECKPOINT_URL
        assert "cuda-checkpoint" in CUDA_CHECKPOINT_URL

    def test_download_to_custom_dir(self, tmp_path):
        """Should download to specified directory."""
        # This test actually downloads - skip in CI without network
        try:
            path = download_cuda_checkpoint(install_dir=tmp_path)
            assert path.exists()
            assert path.parent == tmp_path
            assert os.access(path, os.X_OK)
        except CudaCheckpointError as e:
            if "internet connection" in str(e).lower():
                pytest.skip("No internet connection")
            raise


class TestBundledBinary:
    """Tests for bundled cuda-checkpoint binary."""

    def test_bundled_path_exists(self):
        """Bundled binary should exist in package."""
        path = get_bundled_path()
        assert path.exists(), f"Bundled binary not found at {path}"

    def test_bundled_is_executable(self):
        """Bundled binary should be executable."""
        path = get_bundled_path()
        assert os.access(path, os.X_OK), f"Bundled binary not executable: {path}"

    @requires_nvidia_driver
    def test_bundled_works(self):
        """Bundled binary should run successfully."""
        path = get_bundled_path()
        assert verify_cuda_checkpoint(path)

    def test_bundled_has_license(self):
        """NVIDIA license should be bundled."""
        license_path = get_bundled_path().parent / "NVIDIA_LICENSE"
        assert license_path.exists(), "NVIDIA_LICENSE not found"


class TestFileHash:
    """Tests for get_file_hash()."""

    def test_hash_existing_file(self):
        """Should return hash for existing file."""
        path = get_bundled_path()
        hash_val = get_file_hash(path)
        assert hash_val is not None
        assert len(hash_val) == 32  # MD5 hex length

    def test_hash_nonexistent_file(self):
        """Should return None for nonexistent file."""
        assert get_file_hash(Path("/nonexistent/file")) is None


class TestSupportsMigration:
    """Tests for supports_migration() and get_capabilities()."""

    def test_supports_migration_returns_bool(self):
        """supports_migration() should return a boolean."""
        result = supports_migration()
        assert isinstance(result, bool)

    def test_get_capabilities_returns_dict(self):
        """get_capabilities() should return expected structure."""
        caps = get_capabilities()
        assert isinstance(caps, dict)
        assert "driver_version" in caps
        assert "pause_resume" in caps
        assert "migration" in caps

    def test_capabilities_consistent_with_supports_migration(self):
        """get_capabilities()['migration'] should match supports_migration()."""
        caps = get_capabilities()
        assert caps["migration"] == supports_migration()

    @patch("flexium.utils.cuda_checkpoint.get_driver_version")
    def test_supports_migration_true_for_580(self, mock_driver):
        """Driver 580+ should support migration."""
        mock_driver.return_value = 580
        assert supports_migration() is True

    @patch("flexium.utils.cuda_checkpoint.get_driver_version")
    def test_supports_migration_false_for_550(self, mock_driver):
        """Driver 550-579 should not support migration."""
        mock_driver.return_value = 550
        assert supports_migration() is False

    @patch("flexium.utils.cuda_checkpoint.get_driver_version")
    def test_capabilities_for_550(self, mock_driver):
        """Driver 550 should support pause/resume but not migration."""
        mock_driver.return_value = 550
        caps = get_capabilities()
        assert caps["pause_resume"] is True
        assert caps["migration"] is False


class TestConstants:
    """Tests for module constants."""

    def test_min_driver_version(self):
        """MIN_DRIVER_VERSION should be 550 (pause/resume)."""
        assert MIN_DRIVER_VERSION == 550

    def test_migration_driver_version(self):
        """MIGRATION_DRIVER_VERSION should be 580."""
        assert MIGRATION_DRIVER_VERSION == 580

    def test_url_points_to_nvidia(self):
        """URL should point to NVIDIA's official repo."""
        assert "NVIDIA" in CUDA_CHECKPOINT_URL or "nvidia" in CUDA_CHECKPOINT_URL.lower()
