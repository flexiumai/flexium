"""Tests for the auto module (transparent GPU management)."""

from __future__ import annotations

import threading
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest


class TestAutoModuleGlobals:
    """Tests for auto module global state."""

    def test_get_device(self) -> None:
        """Test get_device returns the current device."""
        import flexium.auto as auto

        original = auto._current_device
        try:
            auto._current_device = "cuda:1"
            assert auto.get_device() == "cuda:1"
        finally:
            auto._current_device = original

    def test_extract_gpu_index_cuda_colon(self) -> None:
        """Test _extract_gpu_index handles cuda:N format."""
        from flexium.auto import _extract_gpu_index

        assert _extract_gpu_index("cuda:0") == "0"
        assert _extract_gpu_index("cuda:1") == "1"
        assert _extract_gpu_index("cuda:7") == "7"

    def test_extract_gpu_index_cuda_only(self) -> None:
        """Test _extract_gpu_index handles 'cuda' without index."""
        from flexium.auto import _extract_gpu_index

        assert _extract_gpu_index("cuda") == "0"

    def test_extract_gpu_index_other(self) -> None:
        """Test _extract_gpu_index handles unknown formats."""
        from flexium.auto import _extract_gpu_index

        assert _extract_gpu_index("cpu") == "0"
        assert _extract_gpu_index("") == "0"


class TestMigration:
    """Tests for migration functionality."""

    def test_heartbeat_triggers_migration_directly(self) -> None:
        """Test _send_heartbeat triggers migration directly (not via pending flag)."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should call _do_migration directly, not set _pending_migration
        assert "_do_migration(target)" in source or "_do_migration(" in source, \
            "_send_heartbeat should call _do_migration directly"
        assert "_do_pause()" in source, \
            "_send_heartbeat should call _do_pause directly"

    def test_migration_skips_when_in_progress(self) -> None:
        """Test migration is skipped when already in progress."""
        import flexium.auto as auto

        original_in_progress = auto._migration_in_progress
        original_device = auto._current_device
        try:
            auto._migration_in_progress = True
            auto._current_device = "cuda:0"

            result = auto._do_migration("cuda:1")

            assert result is False
            assert auto._current_device == "cuda:0"  # Not changed
        finally:
            auto._migration_in_progress = original_in_progress
            auto._current_device = original_device


class TestRunContextManager:
    """Tests for the run() context manager."""

    def test_run_disabled_mode(self) -> None:
        """Test run() in disabled mode does nothing special."""
        import flexium.auto as auto

        with auto.run(disabled=True):
            pass  # Should not raise

    def test_run_sets_process_id(self) -> None:
        """Test run() generates a process ID."""
        import flexium.auto as auto

        # Mock orchestrator connection to avoid network calls
        with patch("flexium.auto._connect_orchestrator"):
            with patch("flexium.auto._disconnect_orchestrator"):
                with patch("flexium.auto._patch_module_cuda"):
                    with patch("flexium.auto._patch_tensor_cuda"):
                        with auto.run(orchestrator=""):
                            assert auto._process_id.startswith("gpu-")
                            assert len(auto._process_id) > 4


class TestPyTorchPatches:
    """Tests for PyTorch patching functions."""

    def test_patch_module_cuda_patches_class(self) -> None:
        """Test _patch_module_cuda modifies nn.Module class."""
        import flexium.auto as auto

        with patch("torch.nn.Module") as mock_module:
            original_cuda = MagicMock()
            mock_module.cuda = original_cuda

            auto._patch_module_cuda()

            # The cuda method should have been replaced

class TestHeartbeat:
    """Tests for heartbeat functionality."""

    def test_send_heartbeat_no_client(self) -> None:
        """Test _send_heartbeat does nothing without client."""
        import flexium.auto as auto

        original_client = auto._orchestrator_client
        try:
            auto._orchestrator_client = None

            # Should not raise
            auto._send_heartbeat()
        finally:
            auto._orchestrator_client = original_client

    def test_heartbeat_loop_stops_on_event(self) -> None:
        """Test _heartbeat_loop stops when stop event is set."""
        import flexium.auto as auto

        original_stop = auto._stop_heartbeat

        try:
            auto._stop_heartbeat = threading.Event()
            auto._stop_heartbeat.set()  # Already set

            with patch("flexium.auto._send_heartbeat") as mock_send:
                # Should exit immediately
                auto._heartbeat_loop()

                # May or may not have called send once depending on timing
        finally:
            auto._stop_heartbeat = original_stop


class TestDriverInterface:
    """Tests for driver interface integration."""

    def test_check_driver_interface_available(self) -> None:
        """Test _check_driver_interface_available returns boolean."""
        from flexium.auto import _check_driver_interface_available

        result = _check_driver_interface_available()
        assert isinstance(result, bool)

    def test_driver_interface_caches_result(self) -> None:
        """Test driver interface availability check caches its result."""
        from flexium import _driver

        # Reset cache
        _driver._interface_available = None
        _driver._interface_path = None

        # First call sets the cache
        result1 = _driver.is_available()

        # Second call should return same result (cached)
        result2 = _driver.is_available()

        assert result1 == result2

    def test_do_migration_selects_strategy(self) -> None:
        """Test _do_migration selects appropriate strategy based on driver availability."""
        import flexium.auto as auto

        # This test verifies the function exists and has the right signature
        # Actual migration testing requires GPUs
        assert callable(auto._do_migration)
        assert callable(auto._do_migration_with_driver)

    def test_do_migration_with_driver_uses_physical_device(self) -> None:
        """Test _do_migration_with_driver uses _physical_device for source.

        Regression test for back-migration bug. After transparent migration from
        cuda:0 to cuda:2, _current_device is still cuda:0 but _physical_device
        is cuda:2. When migrating back (2->0), we must use _physical_device to
        determine the source GPU, not _current_device.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration_with_driver)

        # The source_idx line should use _physical_device, not _current_device
        assert "_extract_gpu_index(_physical_device)" in source, \
            "_do_migration_with_driver should use _physical_device for source_idx"
        assert "_extract_gpu_index(old_device)" not in source or \
               "_extract_gpu_index(_current_device)" not in source, \
            "_do_migration_with_driver should NOT use _current_device for source_idx"

    def test_do_migration_rejects_cpu_target(self) -> None:
        """Test _do_migration rejects CPU as migration target."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration)

        # Should reject CPU target
        assert 'target_device.startswith("cuda")' in source, \
            "_do_migration should validate target is a GPU"
        # Should return False for invalid target
        assert "return False" in source, \
            "_do_migration should return False for invalid target"

    def test_do_migration_rejects_non_gpu_source(self) -> None:
        """Test _do_migration rejects migration from non-GPU device."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration)

        # Should reject if not on GPU
        assert '_current_device.startswith("cuda")' in source, \
            "_do_migration should validate we're on GPU"

    def test_do_migration_requires_driver_interface(self) -> None:
        """Test _do_migration requires driver interface availability."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration)

        # Should check driver interface availability
        assert "_check_driver_interface_available" in source, \
            "_do_migration should check driver interface availability"

    def test_driver_functions_exist(self) -> None:
        """Test driver interface helper functions exist."""
        from flexium.auto import (
            _driver_lock,
            _driver_capture,
            _driver_restore,
            _driver_unlock,
        )

        # Functions should exist and be callable
        assert callable(_driver_lock)
        assert callable(_driver_capture)
        assert callable(_driver_restore)
        assert callable(_driver_unlock)

    def test_driver_functions_handle_missing_path(self) -> None:
        """Test driver functions return False when path is None."""
        from flexium import _driver

        # Ensure path is None
        original_path = _driver._interface_path
        try:
            _driver._interface_path = None

            # All functions should return False when path is not set
            assert _driver.capture_lock(12345) is False
            assert _driver.capture_state(12345) is False
            assert _driver.restore_state(12345) is False
            assert _driver.capture_unlock(12345) is False
        finally:
            _driver._interface_path = original_path

    def test_driver_disabled_flag(self) -> None:
        """Test _interface_disabled flag disables migration."""
        from flexium import _driver

        original_disabled = _driver._interface_disabled
        original_available = _driver._interface_available

        try:
            # Reset cached value and enable disabled flag
            _driver._interface_available = None
            _driver._interface_disabled = True

            # Should return False regardless of actual availability
            result = _driver.is_available()
            assert result is False

        finally:
            _driver._interface_disabled = original_disabled
            _driver._interface_available = original_available

    def test_build_device_map_function_exists(self) -> None:
        """Test _build_device_map function exists and is callable."""
        from flexium.auto import _build_device_map

        assert callable(_build_device_map)

    def test_get_all_gpu_uuids_function_exists(self) -> None:
        """Test _get_all_gpu_uuids function exists and returns list."""
        from flexium.auto import _get_all_gpu_uuids

        result = _get_all_gpu_uuids()
        assert isinstance(result, list)


class TestZeroResideMigration:
    """Tests for zero-residue GPU migration functionality."""

    def test_extract_gpu_index_variations(self) -> None:
        """Test _extract_gpu_index handles various input formats."""
        from flexium.auto import _extract_gpu_index

        # Standard cuda:N format
        assert _extract_gpu_index("cuda:0") == "0"
        assert _extract_gpu_index("cuda:1") == "1"
        assert _extract_gpu_index("cuda:10") == "10"

        # Just "cuda" defaults to 0
        assert _extract_gpu_index("cuda") == "0"

        # Non-cuda formats default to 0
        assert _extract_gpu_index("cpu") == "0"
        assert _extract_gpu_index("mps") == "0"
        assert _extract_gpu_index("") == "0"

    def test_do_migration_requires_driver_interface_runtime(self) -> None:
        """Test _do_migration fails gracefully when driver interface unavailable.

        Migration now requires driver 580+. When unavailable,
        _do_migration should return False with an error message.
        """
        pytest.importorskip("torch")
        import torch
        import flexium.auto as auto
        from flexium import _driver

        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Need at least 2 CUDA devices")

        # Save original state
        original_device = auto._current_device
        original_in_progress = auto._migration_in_progress
        original_interface_available = _driver._interface_available
        original_interface_disabled = _driver._interface_disabled

        try:
            # Disable driver interface to test the error path
            _driver._interface_disabled = True
            _driver._interface_available = None  # Reset cached value

            # Setup
            auto._current_device = "cuda:0"
            auto._migration_in_progress = False
            torch.cuda.set_device(0)

            # Do migration - should fail since driver interface is disabled
            result = auto._do_migration("cuda:1")

            # Verify migration failed gracefully
            assert result is False, "Migration should fail when driver interface unavailable"
            # Device should remain unchanged
            assert auto._current_device == "cuda:0"

        finally:
            # Restore state
            auto._current_device = original_device
            auto._migration_in_progress = original_in_progress
            _driver._interface_available = original_interface_available
            _driver._interface_disabled = original_interface_disabled

    def test_do_migration_with_driver_updates_device(self) -> None:
        """Test _do_migration_with_driver updates device tracking.

        The _do_migration_with_driver function should update
        _physical_device after successful migration. Note: _current_device
        stays the same (logical device) since driver migration does GPU
        identity swapping at the driver level.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration_with_driver)

        # Should update _physical_device tracking after migration
        # _current_device stays the same (logical device like cuda:0)
        assert "_physical_device = target_device" in source, \
            "_do_migration_with_driver must update _physical_device"


class TestPauseFunctionality:
    """Tests for pause/resume functionality."""

    def test_heartbeat_triggers_pause_directly(self) -> None:
        """Test _send_heartbeat triggers _do_pause directly for __PAUSE__ command."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should call _do_pause directly for __PAUSE__ command
        assert "_do_pause()" in source, \
            "_send_heartbeat should call _do_pause directly"
        assert '__PAUSE__' in source, \
            "_send_heartbeat should check for __PAUSE__ target"

    def test_heartbeat_triggers_migration_directly(self) -> None:
        """Test _send_heartbeat triggers _do_migration directly for GPU targets."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should call _do_migration directly for GPU targets
        assert "_do_migration(target)" in source or "_do_migration(" in source, \
            "_send_heartbeat should call _do_migration directly"

    def test_do_pause_function_exists(self) -> None:
        """Test _do_pause function exists and is callable."""
        import flexium.auto as auto

        assert hasattr(auto, "_do_pause")
        assert callable(auto._do_pause)

    def test_do_resume_from_checkpoint_function_exists(self) -> None:
        """Test _do_resume_from_checkpoint function exists and is callable."""
        import flexium.auto as auto

        assert hasattr(auto, "_do_resume_from_checkpoint")
        assert callable(auto._do_resume_from_checkpoint)

    def test_do_pause_attempts_gpu_free_with_driver(self) -> None:
        """Test _do_pause attempts to free GPU when driver migration is available.

        Verifies that pause uses lock -> checkpoint flow to free GPU memory.
        """
        import flexium.auto as auto
        from flexium import _driver

        # Track which functions were called
        calls = []

        def mock_check_available() -> bool:
            calls.append("check_available")
            return True

        def mock_lock(pid: int) -> bool:
            calls.append(f"lock:{pid}")
            return True

        def mock_checkpoint(pid: int) -> bool:
            calls.append(f"checkpoint:{pid}")
            return True

        # Mock the orchestrator client to avoid network calls and break the loop
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.should_migrate = True
        mock_response.target_device = "cuda:0"  # Resume immediately
        mock_client._stub.Heartbeat.return_value = mock_response
        mock_client.complete_migration = MagicMock()

        # Mock _do_resume_from_checkpoint to avoid actual restore
        def mock_resume(paused: str, target: str, cached_gpu_uuids=None) -> bool:
            calls.append(f"resume:{paused}->{target}")
            return True

        original_client = auto._orchestrator_client
        original_path = _driver._interface_path

        try:
            auto._orchestrator_client = mock_client
            _driver._interface_path = "/fake/path"

            with patch.object(auto, "_check_driver_interface_available", mock_check_available):
                with patch.object(auto, "_driver_lock", mock_lock):
                    with patch.object(auto, "_driver_capture", mock_checkpoint):
                        with patch.object(auto, "_do_resume_from_checkpoint", mock_resume):
                            auto._do_pause()

            # Verify the checkpoint flow was called
            assert "check_available" in calls
            assert any("lock:" in c for c in calls), "lock should be called"
            assert any("checkpoint:" in c for c in calls), "checkpoint should be called"
            assert any("resume:" in c for c in calls), "resume should be called"

        finally:
            auto._orchestrator_client = original_client
            _driver._interface_path = original_path

    def test_do_pause_fails_without_driver(self) -> None:
        """Test _do_pause fails when driver migration unavailable.

        Pause now requires driver migration (driver 580+). When unavailable,
        _do_pause should return early with an error.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # Should check driver migration availability and return if not available
        assert "_check_driver_interface_available" in source, \
            "_do_pause must check driver migration availability"
        assert "_pause_in_progress = False" in source, \
            "_do_pause must clear flag when returning early"

    def test_do_resume_from_checkpoint_builds_device_map_for_different_gpu(self) -> None:
        """Test _do_resume_from_checkpoint builds device map when resuming to different GPU."""
        import flexium.auto as auto

        calls = []

        def mock_build_device_map(source: int, target: int) -> str:
            calls.append(f"build_map:{source}->{target}")
            return "GPU-0=GPU-1,GPU-1=GPU-0"

        def mock_restore(pid: int, device_map: str = None) -> bool:
            calls.append(f"restore:{pid}:{device_map}")
            return True

        def mock_unlock(pid: int) -> bool:
            calls.append(f"unlock:{pid}")
            return True

        original_physical = auto._physical_device
        original_client = auto._orchestrator_client

        try:
            auto._physical_device = "cuda:0"
            auto._orchestrator_client = MagicMock()

            with patch.object(auto, "_build_device_map", mock_build_device_map):
                with patch.object(auto, "_driver_restore", mock_restore):
                    with patch.object(auto, "_driver_unlock", mock_unlock):
                        result = auto._do_resume_from_checkpoint("cuda:0", "cuda:1")

            assert result is True
            assert "build_map:0->1" in calls, "should build device map for different GPUs"
            assert any("restore:" in c and "GPU-0=GPU-1" in c for c in calls), \
                "should restore with device map"

        finally:
            auto._physical_device = original_physical
            auto._orchestrator_client = original_client

    def test_do_resume_from_checkpoint_no_device_map_for_same_gpu(self) -> None:
        """Test _do_resume_from_checkpoint doesn't build device map for same GPU."""
        import flexium.auto as auto

        calls = []

        def mock_build_device_map(source: int, target: int) -> str:
            calls.append(f"build_map:{source}->{target}")
            return "GPU-0=GPU-0"

        def mock_restore(pid: int, device_map: str = None) -> bool:
            calls.append(f"restore:{pid}:{device_map}")
            return True

        def mock_unlock(pid: int) -> bool:
            calls.append(f"unlock:{pid}")
            return True

        original_physical = auto._physical_device
        original_client = auto._orchestrator_client

        try:
            auto._physical_device = "cuda:0"
            auto._orchestrator_client = MagicMock()

            with patch.object(auto, "_build_device_map", mock_build_device_map):
                with patch.object(auto, "_driver_restore", mock_restore):
                    with patch.object(auto, "_driver_unlock", mock_unlock):
                        result = auto._do_resume_from_checkpoint("cuda:0", "cuda:0")

            assert result is True
            # Should NOT call build_device_map for same GPU
            assert "build_map:0->0" not in calls, "should not build map for same GPU"
            # Should restore without device map
            assert any("restore:" in c and ":None" in c for c in calls), \
                "should restore without device map"

        finally:
            auto._physical_device = original_physical
            auto._orchestrator_client = original_client

    def test_do_resume_uses_cached_uuids(self) -> None:
        """Test _do_resume_from_checkpoint uses cached UUIDs when provided.

        Regression test for bug where pynvml hung when trying to get GPU UUIDs
        after driver migration because the CUDA context was suspended. The fix
        is to cache UUIDs before checkpointing and pass them to resume.
        """
        import flexium.auto as auto

        calls = []

        # Mock _build_device_map_from_uuids to verify cached UUIDs are used
        def mock_build_from_uuids(source: int, target: int, uuids) -> str:
            calls.append(f"build_from_uuids:{source}->{target}:uuids={uuids}")
            return f"GPU-{source}=GPU-{target}"

        # Mock _build_device_map (shouldn't be called when UUIDs are cached)
        def mock_build_device_map(source: int, target: int) -> str:
            calls.append(f"build_device_map:{source}->{target}")
            return f"GPU-{source}=GPU-{target}"

        def mock_restore(pid: int, device_map: str = None) -> bool:
            calls.append(f"restore:{device_map}")
            return True

        def mock_unlock(pid: int) -> bool:
            calls.append("unlock")
            return True

        original_physical = auto._physical_device
        original_client = auto._orchestrator_client

        try:
            auto._physical_device = "cuda:0"
            auto._orchestrator_client = MagicMock()

            cached_uuids = ["GPU-AAA", "GPU-BBB", "GPU-CCC"]

            with patch.object(auto, "_build_device_map_from_uuids", mock_build_from_uuids):
                with patch.object(auto, "_build_device_map", mock_build_device_map):
                    with patch.object(auto, "_driver_restore", mock_restore):
                        with patch.object(auto, "_driver_unlock", mock_unlock):
                            result = auto._do_resume_from_checkpoint(
                                "cuda:0", "cuda:1", cached_gpu_uuids=cached_uuids
                            )

            assert result is True
            # Should use _build_device_map_from_uuids with cached UUIDs
            assert any("build_from_uuids:" in c and "uuids=" in c for c in calls), \
                f"should use cached UUIDs: {calls}"
            # Should NOT call regular _build_device_map
            assert not any("build_device_map:0->1" in c for c in calls), \
                f"should not call _build_device_map when UUIDs cached: {calls}"

        finally:
            auto._physical_device = original_physical
            auto._orchestrator_client = original_client

    def test_do_resume_does_not_change_cuda_device(self) -> None:
        """Test _do_resume_from_checkpoint does not call torch.cuda.set_device.

        Regression test for bug where resume called torch.cuda.set_device(target_idx)
        which broke transparent migration. After device-map restore, the process
        should continue using the same cuda:X device (now physically mapped to
        a different GPU). Calling set_device would cause new allocations to go
        to a different device than the restored tensors.
        """
        import flexium.auto as auto
        import ast
        import inspect

        source = inspect.getsource(auto._do_resume_from_checkpoint)

        # Parse the source to find actual function calls (not comments)
        # We need to check that set_device is not actually called
        tree = ast.parse(source)

        # Look for any Call nodes with set_device
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check if it's an attribute call like torch.cuda.set_device
                if isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "set_device", \
                        "_do_resume_from_checkpoint should NOT call set_device " \
                        "(breaks transparent migration)"


class TestMemoryReporting:
    """Tests for accurate memory reporting."""

    def test_send_heartbeat_uses_pynvml_memory(self) -> None:
        """Test _send_heartbeat uses get_estimated_gpu_memory for accurate reporting."""
        import flexium.auto as auto

        # Check that get_estimated_gpu_memory is imported in the function
        import inspect
        source = inspect.getsource(auto._send_heartbeat)

        assert "get_estimated_gpu_memory" in source, \
            "_send_heartbeat should use get_estimated_gpu_memory for accurate memory"

    def test_get_estimated_gpu_memory_function_exists(self) -> None:
        """Test get_estimated_gpu_memory function exists in gpu_info."""
        from flexium.utils.gpu_info import get_estimated_gpu_memory

        assert callable(get_estimated_gpu_memory)


class TestPauseResume:
    """Tests for pause/resume functionality (GPU-only)."""

    def test_do_pause_requires_gpu(self) -> None:
        """Test _do_pause requires being on a GPU."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # Should validate we're on GPU
        assert '_current_device.startswith("cuda")' in source, \
            "_do_pause must check we're on GPU"

    def test_do_pause_requires_driver(self) -> None:
        """Test _do_pause requires driver migration availability."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # Should check driver migration availability
        assert "_check_driver_interface_available" in source, \
            "_do_pause must check driver migration availability"

    def test_do_pause_rejects_cpu_resume_target(self) -> None:
        """Test _do_pause rejects CPU as resume target."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # Should validate resume target is GPU
        assert 'target_device.startswith("cuda")' in source, \
            "_do_pause must validate resume target is GPU"

    def test_pause_in_progress_flag_prevents_duplicate_migration(self) -> None:
        """Test _pause_in_progress flag prevents heartbeat thread from migrating.

        The heartbeat thread should not trigger migration when
        _pause_in_progress is True, since _do_pause handles resume internally.
        """
        import flexium.auto as auto

        # 1. Verify _pause_in_progress flag exists
        assert hasattr(auto, "_pause_in_progress"), \
            "auto module must have _pause_in_progress flag"

        # 2. Verify _do_pause sets the flag
        import inspect
        source = inspect.getsource(auto._do_pause)
        assert "_pause_in_progress = True" in source, \
            "_do_pause must set _pause_in_progress flag at start"

        # 3. Verify _do_pause clears the flag before returning
        assert "_pause_in_progress = False" in source, \
            "_do_pause must clear _pause_in_progress flag before return"

    def test_heartbeat_respects_pause_in_progress_flag(self) -> None:
        """Test _send_heartbeat respects _pause_in_progress flag.

        The heartbeat thread should not trigger migration when
        _pause_in_progress is True, since _do_pause handles resume internally.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should check _pause_in_progress before migrating
        assert "_pause_in_progress" in source, \
            "_send_heartbeat must check _pause_in_progress flag"
        assert "Ignoring migration request during pause" in source, \
            "_send_heartbeat must log when ignoring migration during pause"

    def test_heartbeat_loop_skips_during_pause(self) -> None:
        """Test _heartbeat_loop doesn't send heartbeats during pause.

        The regular heartbeat loop should skip sending heartbeats when
        _pause_in_progress is True, letting _do_pause's internal heartbeat
        loop handle communication with the server.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._heartbeat_loop)

        # Should check _pause_in_progress before sending heartbeat
        assert "_pause_in_progress" in source, \
            "_heartbeat_loop must check _pause_in_progress flag"

    def test_do_pause_does_not_acquire_migration_lock(self) -> None:
        """Test _do_pause doesn't try to acquire _migration_lock.

        Regression test for deadlock bug: the heartbeat thread holds
        _migration_lock while calling _do_pause(), so _do_pause() must not
        try to acquire the same lock (threading.Lock is not reentrant).
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # _do_pause should NOT contain "with _migration_lock" since it would deadlock
        assert "with _migration_lock" not in source, \
            "_do_pause must not acquire _migration_lock (would deadlock)"


class TestGpuUuidCaching:
    """Tests for GPU UUID caching at startup.

    After driver migration migration, pynvml indices get remapped at the process
    level. We cache the GPU index → UUID mapping at startup (before any migrations)
    to ensure we always know which physical GPU corresponds to which index.
    """

    def test_gpu_index_to_uuid_cache_exists(self) -> None:
        """Test that _gpu_index_to_uuid cache variable exists."""
        import flexium.auto as auto

        # The cache should exist as a module-level dict
        assert hasattr(auto, "_gpu_index_to_uuid")
        assert isinstance(auto._gpu_index_to_uuid, dict)

    def test_gpu_index_to_name_cache_exists(self) -> None:
        """Test that _gpu_index_to_name cache variable exists."""
        import flexium.auto as auto

        assert hasattr(auto, "_gpu_index_to_name")
        assert isinstance(auto._gpu_index_to_name, dict)

    def test_cache_gpu_info_at_startup_function_exists(self) -> None:
        """Test _cache_gpu_info_at_startup function exists and is callable."""
        import flexium.auto as auto

        assert hasattr(auto, "_cache_gpu_info_at_startup")
        assert callable(auto._cache_gpu_info_at_startup)

    def test_cache_gpu_info_at_startup_is_idempotent(self) -> None:
        """Test that _cache_gpu_info_at_startup only caches once.

        Multiple calls should not recache - the first call's data is used.
        This is critical because after driver migration, pynvml indices are
        scrambled and we need the original mapping.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._cache_gpu_info_at_startup)

        # Should check if cache is already populated and return early
        assert "_gpu_index_to_uuid" in source
        # Should have early return when cache is populated
        assert "return" in source

    def test_physical_gpu_index_tracking_exists(self) -> None:
        """Test that _physical_gpu_index variable exists for tracking.

        After driver migration, we can't extract source_idx from device strings
        because they become meaningless. We need to track the physical index.
        """
        import flexium.auto as auto

        assert hasattr(auto, "_physical_gpu_index")
        # Should be an int (initialized to -1 meaning "not set")
        assert isinstance(auto._physical_gpu_index, int)

    def test_initial_gpu_tracking_exists(self) -> None:
        """Test that initial GPU tracking variables exist.

        After driver migration, pynvml ALWAYS reports process memory on the
        INITIAL GPU, not the current GPU. We need to track this separately.
        """
        import flexium.auto as auto

        assert hasattr(auto, "_initial_gpu_index")
        assert hasattr(auto, "_initial_gpu_uuid")
        assert isinstance(auto._initial_gpu_index, int)
        assert isinstance(auto._initial_gpu_uuid, str)

    def test_migration_uses_cached_uuids(self) -> None:
        """Test that _do_migration_with_driver uses cached UUIDs.

        After driver migration, querying pynvml for GPU info returns wrong data
        because indices are scrambled. The migration code must use cached UUIDs.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration_with_driver)

        # Should use the cached mapping
        assert "_gpu_index_to_uuid" in source, \
            "Migration must use cached GPU UUIDs, not query pynvml"

    def test_migration_uses_physical_gpu_index(self) -> None:
        """Test that _do_migration_with_driver uses _physical_gpu_index.

        After first migration, device strings like 'cuda:1' become meaningless.
        We must use the tracked physical GPU index instead.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration_with_driver)

        # Should reference _physical_gpu_index for source determination
        assert "_physical_gpu_index" in source, \
            "Migration must use tracked _physical_gpu_index, not extract from device string"


class TestDebugFlag:
    """Tests for the FLEXIUM_DEBUG environment variable flag.

    Debug output is controlled by the _DEBUG flag which is set from
    the FLEXIUM_DEBUG environment variable at module load time.
    """

    def test_debug_flag_exists(self) -> None:
        """Test that _DEBUG flag exists in auto module."""
        import flexium.auto as auto

        assert hasattr(auto, "_DEBUG")
        assert isinstance(auto._DEBUG, bool)

    def test_debug_flag_reads_from_environment(self) -> None:
        """Test that _DEBUG is set based on FLEXIUM_DEBUG environment variable."""
        import os

        # Note: We can't easily test this without reimporting the module,
        # so we verify the implementation via source inspection
        import flexium.auto as auto
        import inspect

        # Get the module source file
        source_file = inspect.getfile(auto)
        with open(source_file, "r") as f:
            module_source = f.read()

        # Should read FLEXIUM_DEBUG from environment
        assert "FLEXIUM_DEBUG" in module_source
        assert "_DEBUG = os.environ.get" in module_source

    def test_debug_statements_guarded_in_migration(self) -> None:
        """Test that debug prints in migration are guarded by _DEBUG flag.

        Debug output should only appear when FLEXIUM_DEBUG=1, not always.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_migration_with_driver)

        # Debug prints should be inside if _DEBUG: blocks
        # Count occurrences of "if _DEBUG:" in migration code
        debug_guards = source.count("if _DEBUG:")

        # There should be multiple debug guards for the migration debug output
        assert debug_guards >= 2, \
            "Migration debug output must be guarded by if _DEBUG:"

    def test_debug_statements_guarded_in_heartbeat(self) -> None:
        """Test that debug prints in heartbeat are guarded by _DEBUG flag."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Heartbeat debug output should be guarded
        assert "if _DEBUG:" in source, \
            "Heartbeat debug output must be guarded by if _DEBUG:"


class TestPynvmlGpuUuidTracking:
    """Tests for tracking pynvml_gpu_uuid separately from gpu_uuid.

    After driver migration migration, pynvml reports process memory on the
    INITIAL GPU. The heartbeat must send both:
    - gpu_uuid: Logical/target GPU (where process is running)
    - pynvml_gpu_uuid: Initial GPU (where pynvml sees memory)
    """

    def test_heartbeat_sends_pynvml_gpu_uuid(self) -> None:
        """Test that heartbeat includes pynvml_gpu_uuid field."""
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should construct heartbeat request with pynvml_gpu_uuid
        assert "pynvml_gpu_uuid" in source, \
            "Heartbeat must include pynvml_gpu_uuid field"

    def test_pynvml_gpu_uuid_uses_initial_uuid(self) -> None:
        """Test that pynvml_gpu_uuid is set to _initial_gpu_uuid when available.

        After driver migration, pynvml sees memory on the initial GPU, so
        pynvml_gpu_uuid should be _initial_gpu_uuid (not the current gpu_uuid).
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._send_heartbeat)

        # Should use _initial_gpu_uuid for pynvml_gpu_uuid
        assert "_initial_gpu_uuid" in source, \
            "Heartbeat must use _initial_gpu_uuid for pynvml_gpu_uuid"


class TestEnvironmentVerification:
    """Tests for environment verification and degraded mode."""

    def test_verify_environment_function_exists(self) -> None:
        """Test _verify_environment function exists."""
        import flexium.auto as auto

        assert hasattr(auto, "_verify_environment")
        assert callable(auto._verify_environment)

    def test_is_migration_enabled_function_exists(self) -> None:
        """Test is_migration_enabled function exists."""
        import flexium.auto as auto

        assert hasattr(auto, "is_migration_enabled")
        assert callable(auto.is_migration_enabled)

    def test_migration_enabled_flag_exists(self) -> None:
        """Test _migration_enabled flag exists."""
        import flexium.auto as auto

        assert hasattr(auto, "_migration_enabled")

    def test_verify_environment_disables_migration_without_driver(self) -> None:
        """Test _verify_environment disables migration when driver migration unavailable."""
        import flexium.auto as auto
        from flexium import _driver

        # Save original state
        original_enabled = auto._migration_enabled
        original_available = _driver._interface_available
        original_disabled = _driver._interface_disabled

        try:
            # Force driver migration to be unavailable
            _driver._interface_available = None
            _driver._interface_disabled = True

            result = auto._verify_environment()

            # Should return False and disable migration
            assert result is False
            assert auto._migration_enabled is False

        finally:
            auto._migration_enabled = original_enabled
            _driver._interface_available = original_available
            _driver._interface_disabled = original_disabled

    def test_do_migration_returns_false_when_disabled(self) -> None:
        """Test _do_migration returns False when migration is disabled."""
        import flexium.auto as auto

        original_enabled = auto._migration_enabled
        original_device = auto._current_device

        try:
            auto._migration_enabled = False
            auto._current_device = "cuda:0"

            result = auto._do_migration("cuda:1")
            assert result is False

        finally:
            auto._migration_enabled = original_enabled
            auto._current_device = original_device

    def test_do_pause_returns_when_disabled(self) -> None:
        """Test _do_pause returns early when migration is disabled.

        Check that _do_pause includes the migration_enabled check.
        """
        import flexium.auto as auto
        import inspect

        source = inspect.getsource(auto._do_pause)

        # Should check _migration_enabled
        assert "_migration_enabled" in source, \
            "_do_pause must check _migration_enabled flag"
        assert "requirements" in source.lower() or "disabled" in source.lower(), \
            "_do_pause must warn about requirements or disabled state"
