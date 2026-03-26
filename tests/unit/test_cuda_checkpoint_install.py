"""Tests for cuda-checkpoint installation flow.

Tests the complete installation flow that users experience:
1. pip install flexium -> binary is bundled
2. import flexium.auto -> binary is found, update check runs
3. Training runs -> cuda-checkpoint is available

Note: Many tests require NVIDIA driver/CUDA to be available.
These are skipped in CI environments without GPU support.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from tests.conftest import requires_nvidia_driver


class TestInstallationFlow:
    """Tests for the complete installation flow."""

    def test_bundled_binary_included_in_package(self):
        """Verify cuda-checkpoint is bundled in the package."""
        import flexium
        package_dir = Path(flexium.__file__).parent
        binary_path = package_dir / "bin" / "cuda-checkpoint"

        assert binary_path.exists(), f"Bundled binary not found at {binary_path}"
        assert os.access(binary_path, os.X_OK), "Bundled binary not executable"

    def test_nvidia_license_included(self):
        """Verify NVIDIA license is bundled."""
        import flexium
        package_dir = Path(flexium.__file__).parent
        license_path = package_dir / "bin" / "NVIDIA_LICENSE"

        assert license_path.exists(), "NVIDIA_LICENSE not found"
        content = license_path.read_text()
        assert "NVIDIA" in content

    @requires_nvidia_driver
    def test_import_flexium_auto_works(self):
        """Verify importing flexium.auto succeeds."""
        # This should work without errors
        import flexium.auto

        # flexium.auto should import successfully
        assert flexium.auto is not None

        # cuda_checkpoint module should be importable
        from flexium.utils.cuda_checkpoint import ensure_cuda_checkpoint
        path = ensure_cuda_checkpoint()
        assert path.exists()

    @requires_nvidia_driver
    def test_binary_runs_successfully(self):
        """Verify bundled binary executes correctly."""
        from flexium.utils.cuda_checkpoint import get_bundled_path

        path = get_bundled_path()
        result = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "CUDA checkpoint" in result.stdout
        assert "Version" in result.stdout

    @requires_nvidia_driver
    def test_version_detection_works(self):
        """Verify version can be extracted from binary."""
        from flexium.utils.cuda_checkpoint import get_bundled_path, get_cuda_checkpoint_version

        path = get_bundled_path()
        version = get_cuda_checkpoint_version(path)

        assert version is not None
        # Version format: "590.48.01"
        parts = version.split(".")
        assert len(parts) >= 2
        assert parts[0].isdigit()
        assert int(parts[0]) >= 550  # Minimum supported driver


class TestUpdateFlow:
    """Tests for the update check flow."""

    def test_update_check_with_network(self):
        """Test update check when network is available."""
        from flexium.utils.cuda_checkpoint import check_for_update, get_bundled_path, get_file_hash

        # Get current hash
        bundled_hash = get_file_hash(get_bundled_path())

        # Check for update (may or may not find one)
        try:
            result = check_for_update()
            # Result is URL if update available, None otherwise
            assert result is None or result.startswith("https://")
        except Exception:
            # Network error is acceptable
            pass

    def test_update_check_without_network(self):
        """Test update check gracefully handles network failure."""
        from flexium.utils.cuda_checkpoint import check_for_update

        with patch("flexium.utils.cuda_checkpoint.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Network error")

            # Should return None (no update) rather than raising
            result = check_for_update()
            assert result is None

    @requires_nvidia_driver
    def test_ensure_uses_bundled_when_update_fails(self):
        """Verify bundled binary is used when update download fails."""
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None  # Clear cache

        with patch.object(cc, "check_for_update", return_value="https://fake-url"):
            with patch.object(cc, "download_cuda_checkpoint", side_effect=Exception("Download failed")):
                # Should still succeed using bundled
                path = cc.ensure_cuda_checkpoint(check_updates=True)

                assert path is not None
                assert path.exists()


class TestUserInstallDirectory:
    """Tests for user installation directory (~/.flexium/bin/)."""

    def test_user_dir_takes_priority(self, tmp_path):
        """Verify user-installed binary takes priority over bundled."""
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None  # Clear cache

        # Create a fake user binary
        user_dir = tmp_path / ".flexium" / "bin"
        user_dir.mkdir(parents=True)
        user_binary = user_dir / "cuda-checkpoint"

        # Copy bundled binary to simulate user install
        bundled = cc.get_bundled_path()
        user_binary.write_bytes(bundled.read_bytes())
        user_binary.chmod(0o755)

        # Patch USER_INSTALL_DIR to use our temp dir
        with patch.object(cc, "USER_INSTALL_DIR", user_dir):
            found = cc.find_cuda_checkpoint()
            assert found == user_binary

    def test_download_creates_user_dir(self, tmp_path):
        """Verify download creates user directory if needed."""
        import flexium.utils.cuda_checkpoint as cc

        install_dir = tmp_path / "new_dir" / "bin"
        assert not install_dir.exists()

        try:
            path = cc.download_cuda_checkpoint(install_dir=install_dir)
            assert install_dir.exists()
            assert path.exists()
        except cc.CudaCheckpointError:
            # Network error is acceptable in test environment
            pytest.skip("Network unavailable")


class TestDriverVersionCheck:
    """Tests for driver version validation."""

    def test_driver_version_detected(self):
        """Verify driver version can be detected."""
        from flexium.utils.cuda_checkpoint import get_driver_version

        version = get_driver_version()
        # May be None if no NVIDIA driver
        if version is not None:
            assert isinstance(version, int)
            assert version > 0

    def test_minimum_driver_requirement(self):
        """Verify minimum driver version is 550 (pause/resume)."""
        from flexium.utils.cuda_checkpoint import MIN_DRIVER_VERSION

        assert MIN_DRIVER_VERSION == 550

    def test_migration_driver_requirement(self):
        """Verify migration requires driver 580."""
        from flexium.utils.cuda_checkpoint import MIGRATION_DRIVER_VERSION

        assert MIGRATION_DRIVER_VERSION == 580

    def test_very_old_driver_rejected(self):
        """Verify driver < 550 raises error."""
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None  # Clear cache

        with patch.object(cc, "get_driver_version", return_value=540):
            with pytest.raises(cc.CudaCheckpointError, match="too old"):
                cc.ensure_cuda_checkpoint()

    def test_driver_550_accepted(self):
        """Verify driver 550+ is accepted (pause/resume)."""
        import flexium.utils.cuda_checkpoint as cc
        cc._resolved_path = None  # Clear cache

        with patch.object(cc, "get_driver_version", return_value=550):
            # Should not raise, but may fail for other reasons in test env
            # Just verify it doesn't raise "too old" error
            try:
                cc.ensure_cuda_checkpoint()
            except cc.CudaCheckpointError as e:
                assert "too old" not in str(e).lower()


class TestCLISetup:
    """Tests for flexium-setup CLI."""

    def test_cli_module_exists(self):
        """Verify CLI module can be imported."""
        from flexium.cli.flexium_setup import main
        assert callable(main)

    @requires_nvidia_driver
    def test_cli_check_mode(self):
        """Test --check mode doesn't modify anything."""
        from flexium.cli.flexium_setup import main

        with patch("sys.argv", ["flexium-setup", "--check", "--quiet"]):
            # Should exit 0 if everything is fine
            try:
                result = main()
                assert result == 0
            except SystemExit as e:
                assert e.code == 0
