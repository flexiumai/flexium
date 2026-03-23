"""Tests for auto module using MockGPU for GPU simulation."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Dict, List, Optional

import pytest


class TestAutoDeviceMapBuilding:
    """Tests for device map building in auto module."""

    def test_build_device_map_from_uuids_basic(self) -> None:
        """Test _build_device_map_from_uuids with basic swap."""
        from flexium.auto import _build_device_map_from_uuids

        uuids = ["GPU-UUID-0", "GPU-UUID-1", "GPU-UUID-2"]

        result = _build_device_map_from_uuids(0, 1, uuids)

        assert result is not None
        # Should swap 0 and 1, keep 2 as-is
        assert "GPU-UUID-0=GPU-UUID-1" in result
        assert "GPU-UUID-1=GPU-UUID-0" in result
        assert "GPU-UUID-2=GPU-UUID-2" in result

    def test_build_device_map_from_uuids_same_device(self) -> None:
        """Test _build_device_map_from_uuids when source==target."""
        from flexium.auto import _build_device_map_from_uuids

        uuids = ["GPU-UUID-0", "GPU-UUID-1"]

        result = _build_device_map_from_uuids(0, 0, uuids)

        # Should be identity mapping
        assert result is not None
        assert "GPU-UUID-0=GPU-UUID-0" in result
        assert "GPU-UUID-1=GPU-UUID-1" in result

    def test_build_device_map_from_uuids_empty_list(self) -> None:
        """Test _build_device_map_from_uuids with empty UUID list."""
        from flexium.auto import _build_device_map_from_uuids

        result = _build_device_map_from_uuids(0, 1, [])

        assert result is None

    def test_build_device_map_from_uuids_none(self) -> None:
        """Test _build_device_map_from_uuids with None UUID list."""
        from flexium.auto import _build_device_map_from_uuids

        result = _build_device_map_from_uuids(0, 1, None)

        assert result is None

    def test_build_device_map_from_uuids_invalid_source_index(self) -> None:
        """Test _build_device_map_from_uuids with invalid source index."""
        from flexium.auto import _build_device_map_from_uuids

        uuids = ["GPU-UUID-0", "GPU-UUID-1"]

        result = _build_device_map_from_uuids(5, 1, uuids)

        assert result is None

    def test_build_device_map_from_uuids_invalid_target_index(self) -> None:
        """Test _build_device_map_from_uuids with invalid target index."""
        from flexium.auto import _build_device_map_from_uuids

        uuids = ["GPU-UUID-0", "GPU-UUID-1"]

        result = _build_device_map_from_uuids(0, 10, uuids)

        assert result is None

    def test_build_device_map_uses_cache(self) -> None:
        """Test _build_device_map uses cached UUIDs when available."""
        import flexium.auto as auto

        original_cache = auto._gpu_index_to_uuid.copy()

        try:
            # Set up cached UUIDs
            auto._gpu_index_to_uuid = {
                0: "CACHED-GPU-UUID-0",
                1: "CACHED-GPU-UUID-1",
            }

            result = auto._build_device_map(0, 1)

            # Should use cached UUIDs
            assert result is not None
            assert "CACHED-GPU-UUID-0" in result
            assert "CACHED-GPU-UUID-1" in result

        finally:
            auto._gpu_index_to_uuid = original_cache


class TestAutoMigrationValidation:
    """Tests for migration validation in auto module."""

    def test_do_migration_rejects_cpu_target_at_runtime(self) -> None:
        """Test _do_migration returns False for CPU target."""
        import flexium.auto as auto

        original_device = auto._current_device
        original_in_progress = auto._migration_in_progress
        original_enabled = auto._migration_enabled

        try:
            auto._current_device = "cuda:0"
            auto._migration_in_progress = False
            auto._migration_enabled = True

            result = auto._do_migration("cpu")

            assert result is False

        finally:
            auto._current_device = original_device
            auto._migration_in_progress = original_in_progress
            auto._migration_enabled = original_enabled

    def test_do_migration_rejects_when_not_on_gpu(self) -> None:
        """Test _do_migration returns False when not currently on GPU."""
        import flexium.auto as auto

        original_device = auto._current_device
        original_in_progress = auto._migration_in_progress
        original_enabled = auto._migration_enabled

        try:
            auto._current_device = "cpu"
            auto._migration_in_progress = False
            auto._migration_enabled = True

            result = auto._do_migration("cuda:1")

            assert result is False

        finally:
            auto._current_device = original_device
            auto._migration_in_progress = original_in_progress
            auto._migration_enabled = original_enabled

    def test_do_migration_rejects_when_disabled(self) -> None:
        """Test _do_migration returns False when migration disabled."""
        import flexium.auto as auto

        original_device = auto._current_device
        original_in_progress = auto._migration_in_progress
        original_enabled = auto._migration_enabled

        try:
            auto._current_device = "cuda:0"
            auto._migration_in_progress = False
            auto._migration_enabled = False  # Disabled

            result = auto._do_migration("cuda:1")

            assert result is False

        finally:
            auto._current_device = original_device
            auto._migration_in_progress = original_in_progress
            auto._migration_enabled = original_enabled


class TestAutoPauseValidation:
    """Tests for pause validation in auto module."""

    def test_do_pause_rejects_when_not_on_gpu(self) -> None:
        """Test _do_pause returns early when not on GPU."""
        import flexium.auto as auto

        original_device = auto._current_device
        original_enabled = auto._migration_enabled
        original_pause = auto._pause_in_progress

        try:
            auto._current_device = "cpu"
            auto._migration_enabled = True
            auto._pause_in_progress = False

            # Should return without doing anything since not on GPU
            auto._do_pause()

            # Should have cleared the pause flag
            assert auto._pause_in_progress is False

        finally:
            auto._current_device = original_device
            auto._migration_enabled = original_enabled
            auto._pause_in_progress = original_pause

    def test_do_pause_rejects_when_disabled(self) -> None:
        """Test _do_pause returns early when migration disabled."""
        import flexium.auto as auto

        original_device = auto._current_device
        original_enabled = auto._migration_enabled
        original_pause = auto._pause_in_progress

        try:
            auto._current_device = "cuda:0"
            auto._migration_enabled = False  # Disabled
            auto._pause_in_progress = False

            auto._do_pause()

            assert auto._pause_in_progress is False

        finally:
            auto._current_device = original_device
            auto._migration_enabled = original_enabled
            auto._pause_in_progress = original_pause


class TestAutoResumeFromCheckpoint:
    """Tests for resume from checkpoint functionality."""

    def test_do_resume_same_device(self) -> None:
        """Test _do_resume_from_checkpoint to same device."""
        import flexium.auto as auto

        original_physical = auto._physical_device
        original_current = auto._current_device

        # Mock driver functions
        with patch.object(auto, "_driver_restore", return_value=True):
            with patch.object(auto, "_driver_unlock", return_value=True):
                with patch.object(auto, "_orchestrator_client", None):
                    try:
                        auto._physical_device = "cuda:0"
                        auto._current_device = "cuda:0"

                        result = auto._do_resume_from_checkpoint("cuda:0", "cuda:0")

                        assert result is True
                        assert auto._physical_device == "cuda:0"

                    finally:
                        auto._physical_device = original_physical
                        auto._current_device = original_current

    def test_do_resume_restore_fails(self) -> None:
        """Test _do_resume_from_checkpoint when restore fails."""
        import flexium.auto as auto

        original_physical = auto._physical_device
        original_current = auto._current_device

        with patch.object(auto, "_driver_restore", return_value=False):
            with patch.object(auto, "_driver_unlock", return_value=True):
                try:
                    auto._physical_device = "cuda:0"
                    auto._current_device = "cuda:0"

                    result = auto._do_resume_from_checkpoint("cuda:0", "cuda:0")

                    assert result is False

                finally:
                    auto._physical_device = original_physical
                    auto._current_device = original_current

    def test_do_resume_different_device_restore_fails(self) -> None:
        """Test _do_resume_from_checkpoint to different device when restore fails."""
        import flexium.auto as auto

        original_physical = auto._physical_device
        original_current = auto._current_device

        with patch.object(auto, "_driver_restore", return_value=False):
            with patch.object(auto, "_driver_unlock", return_value=True):
                try:
                    auto._physical_device = "cuda:0"
                    auto._current_device = "cuda:0"

                    result = auto._do_resume_from_checkpoint("cuda:0", "cuda:1")

                    assert result is False

                finally:
                    auto._physical_device = original_physical
                    auto._current_device = original_current

    def test_do_resume_different_device_migration_fails(self) -> None:
        """Test _do_resume_from_checkpoint to different device when migration fails."""
        import flexium.auto as auto

        original_physical = auto._physical_device
        original_current = auto._current_device

        with patch.object(auto, "_driver_restore", return_value=True):
            with patch.object(auto, "_driver_unlock", return_value=True):
                with patch.object(auto, "_do_migration", return_value=False):
                    try:
                        auto._physical_device = "cuda:0"
                        auto._current_device = "cuda:0"

                        result = auto._do_resume_from_checkpoint("cuda:0", "cuda:1")

                        assert result is False

                    finally:
                        auto._physical_device = original_physical
                        auto._current_device = original_current


class TestAutoWithMockGPU:
    """Tests using MockGPU for GPU simulation."""

    def test_mock_gpu_provides_device_info(self) -> None:
        """Test MockGPU can be used to provide device info for testing."""
        from flexium.gpu.mock import MockGPU

        mock_gpu = MockGPU(num_devices=4, memory_per_device=16 * 1024**3)

        # Simulate allocation on device 0
        mock_gpu.set_memory_allocated(0, 4 * 1024**3)

        info = mock_gpu.get_device_info(0)
        assert info is not None
        assert info.memory_used == 4 * 1024**3
        assert info.memory_free == 12 * 1024**3

    def test_mock_gpu_reports(self) -> None:
        """Test MockGPU generates device reports."""
        from flexium.gpu.mock import MockGPU

        mock_gpu = MockGPU(num_devices=2)
        mock_gpu.set_memory_allocated(0, 1000)

        reports = mock_gpu.get_all_device_reports("testhost")

        assert len(reports) == 2
        assert reports[0].process_count == 1  # Has allocation
        assert reports[1].process_count == 0  # No allocation


class TestAutoVerifyEnvironment:
    """Tests for environment verification."""

    def test_verify_environment_no_cuda(self) -> None:
        """Test _verify_environment when CUDA unavailable."""
        import flexium.auto as auto

        original_enabled = auto._migration_enabled

        # Mock torch.cuda.is_available to return False
        with patch.dict("sys.modules", {"torch": MagicMock()}):
            import sys
            sys.modules["torch"].cuda.is_available.return_value = False

            try:
                result = auto._verify_environment()

                assert result is False
                assert auto._migration_enabled is False

            finally:
                auto._migration_enabled = original_enabled

    def test_verify_environment_no_driver(self) -> None:
        """Test _verify_environment when driver interface unavailable."""
        import flexium.auto as auto
        from flexium import _driver

        original_enabled = auto._migration_enabled
        original_available = _driver._interface_available
        original_disabled = _driver._interface_disabled

        try:
            # Force driver to report unavailable
            _driver._interface_available = None  # Reset cache
            _driver._interface_disabled = True  # Force disabled

            result = auto._verify_environment()

            assert result is False
            assert auto._migration_enabled is False

        finally:
            auto._migration_enabled = original_enabled
            _driver._interface_available = original_available
            _driver._interface_disabled = original_disabled


class TestAutoHeartbeatLoop:
    """Tests for heartbeat loop functionality."""

    def test_heartbeat_loop_exits_when_stop_set(self) -> None:
        """Test _heartbeat_loop exits when stop event is set."""
        import flexium.auto as auto

        original_stop = auto._stop_heartbeat

        try:
            auto._stop_heartbeat = threading.Event()
            auto._stop_heartbeat.set()  # Already set - should exit immediately

            with patch.object(auto, "_send_heartbeat") as mock_send:
                auto._heartbeat_loop()

                # Should have exited quickly, may have called once
                assert mock_send.call_count <= 1

        finally:
            auto._stop_heartbeat = original_stop

    def test_send_heartbeat_handles_exception(self) -> None:
        """Test _send_heartbeat handles exceptions gracefully."""
        import flexium.auto as auto

        original_client = auto._orchestrator_client

        try:
            # Mock client that raises
            mock_client = MagicMock()
            mock_client.heartbeat.side_effect = Exception("Connection failed")
            auto._orchestrator_client = mock_client

            # Should not raise
            auto._send_heartbeat()

        finally:
            auto._orchestrator_client = original_client


class TestAutoMigrationWithDriverMocked:
    """Tests for migration with driver interface mocked."""

    def test_do_migration_with_driver_already_in_progress(self) -> None:
        """Test _do_migration_with_driver returns False when already migrating."""
        import flexium.auto as auto

        original_in_progress = auto._migration_in_progress

        try:
            auto._migration_in_progress = True

            result = auto._do_migration_with_driver("cuda:1")

            assert result is False

        finally:
            auto._migration_in_progress = original_in_progress

    def test_do_migration_with_driver_lock_fails(self) -> None:
        """Test _do_migration_with_driver returns False when lock fails."""
        import flexium.auto as auto

        original_in_progress = auto._migration_in_progress
        original_device = auto._current_device
        original_physical = auto._physical_device
        original_physical_idx = auto._physical_gpu_index
        original_cache = auto._gpu_index_to_uuid.copy()

        with patch.object(auto, "_driver_lock", return_value=False):
            try:
                auto._migration_in_progress = False
                auto._current_device = "cuda:0"
                auto._physical_device = "cuda:0"
                auto._physical_gpu_index = -1
                auto._gpu_index_to_uuid = {0: "GPU-0", 1: "GPU-1"}

                result = auto._do_migration_with_driver("cuda:1")

                assert result is False
                assert auto._migration_in_progress is False

            finally:
                auto._migration_in_progress = original_in_progress
                auto._current_device = original_device
                auto._physical_device = original_physical
                auto._physical_gpu_index = original_physical_idx
                auto._gpu_index_to_uuid = original_cache

    def test_do_migration_with_driver_capture_fails(self) -> None:
        """Test _do_migration_with_driver returns False when capture fails."""
        import flexium.auto as auto

        original_in_progress = auto._migration_in_progress
        original_device = auto._current_device
        original_physical = auto._physical_device
        original_physical_idx = auto._physical_gpu_index
        original_cache = auto._gpu_index_to_uuid.copy()

        with patch.object(auto, "_driver_lock", return_value=True):
            with patch.object(auto, "_driver_capture", return_value=False):
                with patch.object(auto, "_driver_unlock", return_value=True):
                    try:
                        auto._migration_in_progress = False
                        auto._current_device = "cuda:0"
                        auto._physical_device = "cuda:0"
                        auto._physical_gpu_index = -1
                        auto._gpu_index_to_uuid = {0: "GPU-0", 1: "GPU-1"}

                        result = auto._do_migration_with_driver("cuda:1")

                        assert result is False
                        assert auto._migration_in_progress is False

                    finally:
                        auto._migration_in_progress = original_in_progress
                        auto._current_device = original_device
                        auto._physical_device = original_physical
                        auto._physical_gpu_index = original_physical_idx
                        auto._gpu_index_to_uuid = original_cache

    def test_do_migration_with_driver_restore_fails(self) -> None:
        """Test _do_migration_with_driver handles restore failure."""
        import flexium.auto as auto

        original_in_progress = auto._migration_in_progress
        original_device = auto._current_device
        original_physical = auto._physical_device
        original_physical_idx = auto._physical_gpu_index
        original_cache = auto._gpu_index_to_uuid.copy()

        with patch.object(auto, "_driver_lock", return_value=True):
            with patch.object(auto, "_driver_capture", return_value=True):
                with patch.object(auto, "_driver_restore", side_effect=[False, True]):  # First fails, recovery succeeds
                    with patch.object(auto, "_driver_unlock", return_value=True):
                        try:
                            auto._migration_in_progress = False
                            auto._current_device = "cuda:0"
                            auto._physical_device = "cuda:0"
                            auto._physical_gpu_index = -1
                            auto._gpu_index_to_uuid = {0: "GPU-0", 1: "GPU-1"}

                            result = auto._do_migration_with_driver("cuda:1")

                            # Should fail and attempt recovery
                            assert result is False
                            assert auto._migration_in_progress is False

                        finally:
                            auto._migration_in_progress = original_in_progress
                            auto._current_device = original_device
                            auto._physical_device = original_physical
                            auto._physical_gpu_index = original_physical_idx
                            auto._gpu_index_to_uuid = original_cache


class TestAutoGetAllGpuUuids:
    """Tests for _get_all_gpu_uuids function."""

    def test_get_all_gpu_uuids_pynvml_exception(self) -> None:
        """Test _get_all_gpu_uuids handles pynvml exceptions."""
        from flexium.auto import _get_all_gpu_uuids

        with patch("pynvml.nvmlInit", side_effect=Exception("pynvml failed")):
            result = _get_all_gpu_uuids()

            assert result == []

    def test_get_all_gpu_uuids_with_mock_pynvml(self) -> None:
        """Test _get_all_gpu_uuids with mocked pynvml."""
        from flexium.auto import _get_all_gpu_uuids

        mock_handle = MagicMock()

        with patch("pynvml.nvmlInit"):
            with patch("pynvml.nvmlDeviceGetCount", return_value=2):
                with patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=mock_handle):
                    with patch("pynvml.nvmlDeviceGetUUID", side_effect=["UUID-0", "UUID-1"]):
                        result = _get_all_gpu_uuids()

                        assert result == ["UUID-0", "UUID-1"]


class TestAutoInitialization:
    """Tests for auto module initialization."""

    def test_is_active_returns_false_initially(self) -> None:
        """Test is_active returns False when not initialized."""
        import flexium.auto as auto

        original_client = auto._orchestrator_client
        original_process_id = auto._process_id

        try:
            auto._orchestrator_client = None
            auto._process_id = ""

            assert auto.is_active() is False

        finally:
            auto._orchestrator_client = original_client
            auto._process_id = original_process_id

    def test_is_active_returns_true_with_process_id(self) -> None:
        """Test is_active returns True when process_id is set."""
        import flexium.auto as auto

        original_client = auto._orchestrator_client
        original_process_id = auto._process_id

        try:
            auto._orchestrator_client = None
            auto._process_id = "gpu-test123"

            assert auto.is_active() is True

        finally:
            auto._orchestrator_client = original_client
            auto._process_id = original_process_id

    def test_is_active_returns_true_with_client(self) -> None:
        """Test is_active returns True when client exists."""
        import flexium.auto as auto

        original_client = auto._orchestrator_client
        original_process_id = auto._process_id

        try:
            auto._orchestrator_client = MagicMock()
            auto._process_id = ""

            assert auto.is_active() is True

        finally:
            auto._orchestrator_client = original_client
            auto._process_id = original_process_id


class TestAutoGlobalState:
    """Tests for auto module global state management."""

    def test_get_physical_device(self) -> None:
        """Test get_physical_device returns physical device."""
        import flexium.auto as auto

        original = auto._physical_device

        try:
            auto._physical_device = "cuda:2"
            assert auto.get_physical_device() == "cuda:2"

        finally:
            auto._physical_device = original

    def test_is_migration_in_progress(self) -> None:
        """Test is_migration_in_progress returns correct state."""
        import flexium.auto as auto

        original = auto._migration_in_progress

        try:
            auto._migration_in_progress = True
            assert auto.is_migration_in_progress() is True

            auto._migration_in_progress = False
            assert auto.is_migration_in_progress() is False

        finally:
            auto._migration_in_progress = original

    def test_get_process_id(self) -> None:
        """Test get_process_id returns process ID."""
        import flexium.auto as auto

        original = auto._process_id

        try:
            auto._process_id = "gpu-test12345"
            assert auto.get_process_id() == "gpu-test12345"

        finally:
            auto._process_id = original

    def test_is_migration_enabled(self) -> None:
        """Test is_migration_enabled returns correct state."""
        import flexium.auto as auto

        original = auto._migration_enabled

        try:
            auto._migration_enabled = True
            assert auto.is_migration_enabled() is True

            auto._migration_enabled = False
            assert auto.is_migration_enabled() is False

        finally:
            auto._migration_enabled = original
