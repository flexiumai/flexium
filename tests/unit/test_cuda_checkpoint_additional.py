"""Additional tests for cuda_checkpoint module to improve coverage."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from flexium.utils.cuda_checkpoint import (
    get_driver_version,
    get_file_hash,
    get_bundled_path,
    supports_migration,
    get_capabilities,
    MIN_DRIVER_VERSION,
    MIGRATION_DRIVER_VERSION,
)


class TestGetDriverVersionFallback:
    """Tests for get_driver_version fallback paths."""

    def test_driver_version_pynvml_exception(self) -> None:
        """Test fallback when pynvml fails."""
        with patch("pynvml.nvmlInit", side_effect=Exception("pynvml failed")):
            # Should fall through to nvidia-smi fallback
            result = get_driver_version()
            # Result could be int from nvidia-smi or None if that also fails
            assert result is None or isinstance(result, int)

    def test_driver_version_both_fail(self) -> None:
        """Test return value when both pynvml and nvidia-smi fail."""
        with patch("pynvml.nvmlInit", side_effect=Exception("pynvml failed")):
            with patch("subprocess.run", side_effect=Exception("nvidia-smi failed")):
                result = get_driver_version()
                assert result is None


class TestGetFileHash:
    """Tests for get_file_hash function."""

    def test_file_hash_existing_file(self, tmp_path: Path) -> None:
        """Test hash of existing file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = get_file_hash(test_file)

        assert result is not None
        assert len(result) == 32  # MD5 hash length

    def test_file_hash_nonexistent_file(self) -> None:
        """Test hash of nonexistent file returns None."""
        result = get_file_hash(Path("/nonexistent/file.txt"))
        assert result is None

    def test_file_hash_permission_error(self, tmp_path: Path) -> None:
        """Test hash when file read fails."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        with patch.object(Path, "read_bytes", side_effect=PermissionError("denied")):
            result = get_file_hash(test_file)
            assert result is None


class TestGetBundledPath:
    """Tests for get_bundled_path function."""

    def test_bundled_path_returns_path(self) -> None:
        """Test bundled path returns Path object."""
        result = get_bundled_path()
        assert isinstance(result, Path)


class TestSupportsMigration:
    """Tests for supports_migration function."""

    def test_supports_migration_high_driver(self) -> None:
        """Test migration support with high driver version."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=600):
            result = supports_migration()
            assert result is True

    def test_supports_migration_low_driver(self) -> None:
        """Test migration not supported with low driver version."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=500):
            result = supports_migration()
            assert result is False

    def test_supports_migration_no_driver(self) -> None:
        """Test migration not supported without driver."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=None):
            result = supports_migration()
            assert result is False

    def test_supports_migration_exact_version(self) -> None:
        """Test migration at exact required version."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=MIGRATION_DRIVER_VERSION):
            result = supports_migration()
            assert result is True


class TestGetCapabilities:
    """Tests for get_capabilities function."""

    def test_capabilities_with_high_driver(self) -> None:
        """Test capabilities with high driver version."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=600):
            with patch("flexium.utils.cuda_checkpoint.find_cuda_checkpoint", return_value=Path("/usr/bin/cuda-checkpoint")):
                caps = get_capabilities()

                assert isinstance(caps, dict)
                assert "driver_version" in caps
                assert caps["driver_version"] == 600
                assert "pause_resume" in caps
                assert "migration" in caps

    def test_capabilities_with_no_driver(self) -> None:
        """Test capabilities when driver not available."""
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=None):
            caps = get_capabilities()

            assert isinstance(caps, dict)
            assert caps["driver_version"] is None
            assert caps["pause_resume"] is False
            assert caps["migration"] is False

    def test_capabilities_pause_resume_only(self) -> None:
        """Test capabilities with driver supporting pause but not migration."""
        # Driver 550+ supports pause/resume, 580+ supports migration
        with patch("flexium.utils.cuda_checkpoint.get_driver_version", return_value=560):
            caps = get_capabilities()

            assert caps["pause_resume"] is True
            assert caps["migration"] is False


class TestMinDriverVersions:
    """Tests for driver version constants."""

    def test_min_driver_version_reasonable(self) -> None:
        """Test minimum driver version is reasonable."""
        assert MIN_DRIVER_VERSION >= 500  # Should require modern driver

    def test_migration_driver_version_higher(self) -> None:
        """Test migration requires higher version than pause."""
        assert MIGRATION_DRIVER_VERSION >= MIN_DRIVER_VERSION
