"""Tests for the driver interface module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestDriverAvailability:
    """Tests for driver availability checking."""

    def test_disable_interface(self) -> None:
        """Test disable_interface sets flags correctly."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            _driver.disable_interface()
            assert _driver._interface_disabled is True
            assert _driver._interface_available is False
        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available

    def test_enable_interface(self) -> None:
        """Test enable_interface resets flags."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            _driver.disable_interface()
            _driver.enable_interface()
            assert _driver._interface_disabled is False
            assert _driver._interface_available is None  # Forces re-check
        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available

    def test_is_available_when_disabled(self) -> None:
        """Test is_available returns False when disabled."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            _driver._interface_disabled = True
            _driver._interface_available = None  # Reset cache
            result = _driver.is_available()
            assert result is False
        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available

    def test_is_available_uses_cache(self) -> None:
        """Test is_available uses cached result."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            _driver._interface_disabled = False
            _driver._interface_available = True  # Set cache
            result = _driver.is_available()
            assert result is True
        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available

    def test_is_available_checks_driver_version(self) -> None:
        """Test is_available checks driver version."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            _driver._interface_disabled = False
            _driver._interface_available = None  # Force check

            # Mock driver version check to fail
            with patch.object(_driver, "_check_driver_version", return_value=False):
                result = _driver.is_available()
                assert result is False
        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available


class TestCheckDriverVersion:
    """Tests for _check_driver_version function."""

    def test_check_driver_version_success(self) -> None:
        """Test _check_driver_version with valid version."""
        from flexium import _driver

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "580.100.00\n"

        with patch("subprocess.run", return_value=mock_result):
            result = _driver._check_driver_version()
            assert result is True

    def test_check_driver_version_too_old(self) -> None:
        """Test _check_driver_version with old driver."""
        from flexium import _driver

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "470.50.00\n"

        with patch("subprocess.run", return_value=mock_result):
            result = _driver._check_driver_version()
            assert result is False

    def test_check_driver_version_nvidia_smi_fails(self) -> None:
        """Test _check_driver_version when nvidia-smi fails."""
        from flexium import _driver

        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("subprocess.run", return_value=mock_result):
            result = _driver._check_driver_version()
            assert result is False

    def test_check_driver_version_exception(self) -> None:
        """Test _check_driver_version handles exceptions."""
        from flexium import _driver

        with patch("subprocess.run", side_effect=Exception("command not found")):
            result = _driver._check_driver_version()
            assert result is False

    def test_check_driver_version_timeout(self) -> None:
        """Test _check_driver_version handles timeout."""
        from flexium import _driver

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = _driver._check_driver_version()
            assert result is False


class TestGetSearchPaths:
    """Tests for _get_search_paths function."""

    def test_get_search_paths_returns_list(self) -> None:
        """Test _get_search_paths returns a list of paths."""
        from flexium import _driver

        paths = _driver._get_search_paths()
        assert isinstance(paths, list)
        assert all(isinstance(p, Path) for p in paths)

    def test_get_search_paths_includes_home_bin(self) -> None:
        """Test _get_search_paths includes ~/bin."""
        from flexium import _driver

        paths = _driver._get_search_paths()
        home_bin = Path.home() / "bin"
        assert any(str(p).startswith(str(home_bin)) for p in paths)


class TestGetInterfacePath:
    """Tests for get_interface_path function."""

    def test_get_interface_path_when_not_available(self) -> None:
        """Test get_interface_path returns None when not available."""
        from flexium import _driver

        original_available = _driver._interface_available

        try:
            _driver._interface_available = False
            result = _driver.get_interface_path()
            assert result is None
        finally:
            _driver._interface_available = original_available

    def test_get_interface_path_when_available(self) -> None:
        """Test get_interface_path returns path when available."""
        from flexium import _driver

        original_available = _driver._interface_available
        original_path = _driver._interface_path

        try:
            _driver._interface_available = True
            _driver._interface_path = Path("/usr/bin/test-tool")
            result = _driver.get_interface_path()
            assert result == Path("/usr/bin/test-tool")
        finally:
            _driver._interface_available = original_available
            _driver._interface_path = original_path


class TestCaptureLock:
    """Tests for capture_lock function."""

    def test_capture_lock_no_path(self) -> None:
        """Test capture_lock returns False when no interface path."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = None
            result = _driver.capture_lock(12345)
            assert result is False
        finally:
            _driver._interface_path = original_path

    def test_capture_lock_success(self) -> None:
        """Test capture_lock returns True on success."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_lock(12345)
                assert result is True
        finally:
            _driver._interface_path = original_path

    def test_capture_lock_failure(self) -> None:
        """Test capture_lock returns False on failure."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 1

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_lock(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path

    def test_capture_lock_exception(self) -> None:
        """Test capture_lock handles exceptions."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            with patch("subprocess.run", side_effect=Exception("error")):
                result = _driver.capture_lock(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path


class TestCaptureState:
    """Tests for capture_state function."""

    def test_capture_state_no_path(self) -> None:
        """Test capture_state returns False when no interface path."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = None
            result = _driver.capture_state(12345)
            assert result is False
        finally:
            _driver._interface_path = original_path

    def test_capture_state_success(self) -> None:
        """Test capture_state returns True on success."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_state(12345)
                assert result is True
        finally:
            _driver._interface_path = original_path

    def test_capture_state_failure(self) -> None:
        """Test capture_state returns False on failure."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 1

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_state(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path


class TestRestoreState:
    """Tests for restore_state function."""

    def test_restore_state_no_path(self) -> None:
        """Test restore_state returns False when no interface path."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = None
            result = _driver.restore_state(12345)
            assert result is False
        finally:
            _driver._interface_path = original_path

    def test_restore_state_success(self) -> None:
        """Test restore_state returns True on success."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.restore_state(12345)
                assert result is True
        finally:
            _driver._interface_path = original_path

    def test_restore_state_with_device_map(self) -> None:
        """Test restore_state passes device_map argument."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result) as mock_run:
                result = _driver.restore_state(12345, device_map="GPU-A=GPU-B")
                assert result is True
                # Verify device_map was passed
                call_args = mock_run.call_args
                assert "--device-map" in call_args[0][0]


        finally:
            _driver._interface_path = original_path


class TestCaptureUnlock:
    """Tests for capture_unlock function."""

    def test_capture_unlock_no_path(self) -> None:
        """Test capture_unlock returns False when no interface path."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = None
            result = _driver.capture_unlock(12345)
            assert result is False
        finally:
            _driver._interface_path = original_path

    def test_capture_unlock_success(self) -> None:
        """Test capture_unlock returns True on success."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 0

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_unlock(12345)
                assert result is True
        finally:
            _driver._interface_path = original_path

    def test_capture_unlock_failure(self) -> None:
        """Test capture_unlock returns False on failure."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 1

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.capture_unlock(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path

    def test_capture_unlock_exception(self) -> None:
        """Test capture_unlock handles exceptions."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            with patch("subprocess.run", side_effect=Exception("error")):
                result = _driver.capture_unlock(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path


class TestCaptureStateException:
    """Tests for capture_state exception handling."""

    def test_capture_state_exception(self) -> None:
        """Test capture_state handles exceptions."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            with patch("subprocess.run", side_effect=Exception("checkpoint error")):
                result = _driver.capture_state(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path


class TestRestoreStateException:
    """Tests for restore_state exception handling."""

    def test_restore_state_exception(self) -> None:
        """Test restore_state handles exceptions."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            with patch("subprocess.run", side_effect=Exception("restore error")):
                result = _driver.restore_state(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path

    def test_restore_state_failure_stderr(self) -> None:
        """Test restore_state logs stderr on failure."""
        from flexium import _driver

        original_path = _driver._interface_path

        try:
            _driver._interface_path = Path("/usr/bin/test-tool")

            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stderr = "Error: device not found"
            mock_result.stdout = ""

            with patch("subprocess.run", return_value=mock_result):
                result = _driver.restore_state(12345)
                assert result is False
        finally:
            _driver._interface_path = original_path


class TestIsAvailableSearchPaths:
    """Tests for is_available searching paths."""

    def test_is_available_finds_tool_in_path(self) -> None:
        """Test is_available finds tool via shutil.which."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available
        original_path = _driver._interface_path

        try:
            _driver._interface_disabled = False
            _driver._interface_available = None  # Force check
            _driver._interface_path = None

            # Mock driver version check to pass
            with patch.object(_driver, "_check_driver_version", return_value=True):
                # Mock shutil.which to return a path
                with patch("shutil.which", return_value="/usr/bin/test-tool"):
                    result = _driver.is_available()
                    assert result is True

        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available
            _driver._interface_path = original_path

    def test_is_available_searches_paths(self) -> None:
        """Test is_available searches common paths when which fails."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available
        original_path = _driver._interface_path

        try:
            _driver._interface_disabled = False
            _driver._interface_available = None  # Force check
            _driver._interface_path = None

            # Mock driver version check to pass
            with patch.object(_driver, "_check_driver_version", return_value=True):
                # Mock shutil.which to return None (not in PATH)
                with patch("shutil.which", return_value=None):
                    # Mock _get_search_paths to return non-existent paths
                    mock_paths = [Path("/nonexistent/path/tool")]
                    with patch.object(_driver, "_get_search_paths", return_value=mock_paths):
                        result = _driver.is_available()
                        # Should return False since paths don't exist
                        assert result is False

        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available
            _driver._interface_path = original_path


class TestSupportsMigration:
    """Tests for supports_migration function."""

    def test_supports_migration_when_not_available(self) -> None:
        """Test supports_migration returns False when driver not available."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=False):
            assert _driver.supports_migration() is False

    def test_supports_migration_with_driver_580(self) -> None:
        """Test supports_migration returns True with driver 580+."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=True):
            with patch.object(_driver, "_get_driver_version", return_value=580):
                assert _driver.supports_migration() is True

    def test_supports_migration_with_driver_590(self) -> None:
        """Test supports_migration returns True with driver 590."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=True):
            with patch.object(_driver, "_get_driver_version", return_value=590):
                assert _driver.supports_migration() is True

    def test_supports_migration_with_driver_575(self) -> None:
        """Test supports_migration returns False with driver 575 (pause only)."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=True):
            with patch.object(_driver, "_get_driver_version", return_value=575):
                assert _driver.supports_migration() is False

    def test_supports_migration_with_driver_550(self) -> None:
        """Test supports_migration returns False with driver 550 (pause only)."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=True):
            with patch.object(_driver, "_get_driver_version", return_value=550):
                assert _driver.supports_migration() is False

    def test_supports_migration_driver_version_none(self) -> None:
        """Test supports_migration returns False when version unavailable."""
        from flexium import _driver

        with patch.object(_driver, "is_available", return_value=True):
            with patch.object(_driver, "_get_driver_version", return_value=None):
                assert _driver.supports_migration() is False
