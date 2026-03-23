"""Transparent GPU control with automatic PyTorch patching.

This module provides a near-zero-code-change experience for GPU migration.
Just add one import and wrap your training in `run()`:

    import flexium.auto

    with flexium.auto.run():
        model = nn.Linear(784, 10).cuda()  # Transparently managed
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(100):
            for batch in dataloader:
                data, target = batch[0].cuda(), batch[1].cuda()
                loss = model(data).sum()
                loss.backward()
                optimizer.step()

Configuration priority:
1. Inline parameters to run()
2. Environment variables (FLEXIUM_SERVER, GPU_DEVICE)
3. Config file (~/.flexiumrc or ./.flexiumrc)
4. Defaults (local mode with warning)

Migration is truly transparent - training continues in the same process,
same loop iteration, just on a different GPU.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set

from flexium.config import (
    FlexiumConfig,
    load_config,
    print_no_orchestrator_warning,
)
from flexium.utils.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Debug flag - set via FLEXIUM_DEBUG=1 environment variable
_DEBUG = os.environ.get("FLEXIUM_DEBUG", "").lower() in ("1", "true", "yes")

# Global state
_current_device: str = "cuda:0"
_physical_device: str = "cuda:0"
_physical_gpu_uuid: str = ""  # Cached GPU UUID (updated after migration) - the TARGET/logical GPU
_physical_gpu_name: str = ""  # Cached GPU name (updated after migration)
_physical_gpu_index: int = -1  # Cached physical GPU index (updated after migration)
_physical_gpu_pid: int = 0  # Cached GPU PID (updated after migration)
# =============================================================================
# GPU STATE MANAGEMENT NOTES
# =============================================================================
# After driver-level migration, TWO things get remapped at the process level:
#
# 1. PYNVML INDICES: nvmlDeviceGetHandleByIndex(i) returns different GPUs after
#    migration. The process's view of GPU indices is scrambled.
#
# 2. MEMORY REPORTING: pynvml always reports our process's memory on the INITIAL
#    GPU (the first GPU we started on), regardless of subsequent migrations.
#
# SOLUTION: We cache the GPU index→UUID mapping at startup (before any migrations)
# and use this cache for all subsequent operations. This ensures we always know
# which physical GPU corresponds to which index.
#
# TODO(multi-gpu): For multi-GPU process support, consider:
#   - Track _initial_gpu_index/_initial_gpu_uuid per GPU (Dict[int, str])
#   - Track memory per GPU using torch.cuda.memory_reserved(device) instead of pynvml
#   - The cached _gpu_index_to_uuid mapping will still work for multi-GPU
#   - Migration may need to handle multiple device mappings simultaneously
#   - Consider whether to migrate all GPUs together or allow partial migration
# =============================================================================
_initial_gpu_index: int = -1  # Initial GPU index (pynvml reports process memory here)
_initial_gpu_uuid: str = ""  # Initial GPU UUID (where pynvml sees our memory)

# Cache GPU index -> UUID mapping at startup (before any driver-level remapping)
# CRITICAL: After migration, pynvml indices get remapped at process level!
# This cache preserves the true physical index → UUID mapping.
# TODO(multi-gpu): This cache works for multi-GPU - each index maps to correct UUID
_gpu_index_to_uuid: Dict[int, str] = {}  # Physical index -> UUID (cached at startup)
_gpu_index_to_name: Dict[int, str] = {}  # Physical index -> name (cached at startup)
_process_id: str = ""
_orchestrator_client: Optional[Any] = None
_heartbeat_thread: Optional[threading.Thread] = None
_stop_heartbeat = threading.Event()
_migration_lock = threading.Lock()
_migration_in_progress = False
_pause_in_progress = False  # Set during _do_pause() to prevent heartbeat thread conflicts
_migration_enabled = True  # Set to False if environment requirements not met
_cached_visible_devices: List[Dict[str, Any]] = []  # Cached device reports for reconnect during pause
_failed_gpus: Set[str] = set()  # GPUs that have failed (for standalone recovery)
_start_time: float = 0.0  # Unix timestamp when process started (for runtime tracking)
_cached_memory_allocated: int = 0  # Cached memory for reconnect
_cached_memory_reserved: int = 0  # Cached memory for reconnect




def get_device() -> str:
    """Get the current device string.

    Returns:
        Current device (e.g., "cuda:0", "cuda:1").
    """
    return _current_device


def get_physical_device() -> str:
    """Get the physical device string.

    After migration, this reflects the actual GPU we're running on.

    Returns:
        Physical device (e.g., "cuda:0", "cuda:1").
    """
    return _physical_device


def is_active() -> bool:
    """Check if Flexium auto-management is currently active.

    Returns:
        True if inside a flexium.auto.run() context, False otherwise.
    """
    return _orchestrator_client is not None or _process_id != ""


def is_migration_in_progress() -> bool:
    """Check if a migration is currently in progress.

    Returns:
        True if migration is happening, False otherwise.
    """
    return _migration_in_progress


def get_process_id() -> str:
    """Get the Flexium process ID.

    Returns:
        Process ID string (e.g., "gpu-abc12345").
    """
    return _process_id


def _extract_gpu_index(device: str) -> str:
    """Extract GPU index from device string."""
    if device.startswith("cuda:"):
        return device.split(":")[1]
    elif device == "cuda":
        return "0"
    return "0"


def _verify_environment() -> bool:
    """Verify that the environment meets Flexium requirements for migration.

    Checks:
    - CUDA is available
    - Driver interface is available (requires driver 580+)

    Returns:
        True if all requirements are met, False otherwise.
        When False, migration and pause are disabled but training continues.
    """
    global _migration_enabled

    from flexium import _driver

    issues = []

    # Check CUDA availability
    try:
        import torch
        if not torch.cuda.is_available():
            issues.append("CUDA is not available")
    except ImportError:
        issues.append("PyTorch is not installed")

    # Check driver interface availability (550+ for pause, 580+ for migration)
    if not _driver.is_available():
        issues.append("NVIDIA driver 550+ required for pause/resume (580+ for migration)")

    if issues:
        _migration_enabled = False
        for issue in issues:
            logger.warning(
                "Requirements for Flexium migration are not met: %s. "
                "Training will continue but migration and pause are disabled.",
                issue
            )
        return False

    _migration_enabled = True
    logger.debug("Environment verification passed - migration enabled")
    return True


def is_migration_enabled() -> bool:
    """Check if migration/pause functionality is enabled.

    Returns:
        True if environment requirements are met and migration is possible.
    """
    return _migration_enabled


# ============================================================================
# Zero-residue migration support (NVIDIA driver 580+)
# ============================================================================
#
# When supported by the driver, Flexium can perform migrations that leave
# absolutely no memory on the source GPU (zero VRAM residue). This is
# achieved through driver-level capabilities (driver 580+).
#
# Requirements:
# - NVIDIA driver 580 or higher
#
# ============================================================================


def _check_driver_interface_available() -> bool:
    """Check if zero-residue migration is available.

    Zero-residue migration requires driver 580+ and provides:
    - Complete GPU memory release on source device
    - Seamless state preservation during migration

    Returns:
        True if zero-residue migration is available.
    """
    from flexium import _driver
    return _driver.is_available()


def _driver_lock(pid: int) -> bool:
    """Lock GPU state for migration."""
    from flexium import _driver
    return _driver.capture_lock(pid)


def _driver_capture(pid: int) -> bool:
    """Capture GPU state for migration."""
    from flexium import _driver
    return _driver.capture_state(pid)


def _driver_restore(pid: int, device_map: Optional[str] = None) -> bool:
    """Restore GPU state after capture."""
    from flexium import _driver
    return _driver.restore_state(pid, device_map)


def _driver_unlock(pid: int) -> bool:
    """Unlock GPU state after migration."""
    from flexium import _driver
    return _driver.capture_unlock(pid)


def _get_all_gpu_uuids() -> List[str]:
    """Get identifiers for all available GPUs.

    Returns:
        List of GPU identifiers.
    """
    uuids = []
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            uuids.append(uuid)
    except Exception as e:
        logger.debug(f"Failed to get GPU UUIDs: {e}")
    return uuids


def _build_device_map(source_idx: int, target_idx: int) -> Optional[str]:
    """Build device mapping for zero-residue migration.

    Parameters:
        source_idx: Source GPU index.
        target_idx: Target GPU index.

    Returns:
        Device mapping string or None on error.
    """
    # Use cached UUIDs if available (after driver migration, pynvml indices are remapped)
    if _gpu_index_to_uuid:
        uuids = [_gpu_index_to_uuid.get(i, "") for i in range(len(_gpu_index_to_uuid))]
    else:
        uuids = _get_all_gpu_uuids()
    return _build_device_map_from_uuids(source_idx, target_idx, uuids)


def _build_device_map_from_uuids(
    source_idx: int,
    target_idx: int,
    uuids: Optional[List[str]],
) -> Optional[str]:
    """Build device mapping from pre-fetched GPU UUIDs.

    This variant accepts pre-cached UUIDs, which is necessary when the process
    is in a checkpointed state (pynvml calls may hang with suspended CUDA context).

    Parameters:
        source_idx: Source GPU index.
        target_idx: Target GPU index.
        uuids: List of GPU UUIDs (pre-cached).

    Returns:
        Device mapping string or None on error.
    """
    try:
        if not uuids:
            logger.error("_build_device_map_from_uuids: No GPU UUIDs provided")
            return None

        if source_idx >= len(uuids) or target_idx >= len(uuids):
            logger.error(f"Invalid GPU index: source={source_idx}, target={target_idx}, available={len(uuids)}")
            return None

        # Build mapping: swap source and target, keep others as identity
        mappings = []
        for i, uuid in enumerate(uuids):
            if i == source_idx:
                # Source GPU maps to target GPU
                new_uuid = uuids[target_idx]
            elif i == target_idx:
                # Target GPU maps to source GPU (swap for back-migration)
                new_uuid = uuids[source_idx]
            else:
                # Other GPUs map to themselves
                new_uuid = uuid
            mappings.append(f"{uuid}={new_uuid}")

        return ",".join(mappings)
    except Exception as e:
        logger.error(f"_build_device_map_from_uuids failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _do_migration_with_driver(target_device: str) -> bool:
    """Perform zero-residue GPU migration.

    Uses advanced driver capabilities (driver 580+) to migrate with
    complete GPU memory release on the source device.

    Parameters:
        target_device: Target device string (e.g., "cuda:1").

    Returns:
        True if migration succeeded, False otherwise.
    """
    global _current_device, _physical_device, _migration_in_progress, _physical_gpu_uuid, _physical_gpu_name, _physical_gpu_index, _physical_gpu_pid, _initial_gpu_index, _initial_gpu_uuid

    if _migration_in_progress:
        return False

    _migration_in_progress = True
    old_device = _current_device
    pid = os.getpid()

    try:
        import torch

        # IMPORTANT: After driver migration, device strings like "cuda:1" don't correspond
        # to physical GPU indices anymore! We must use tracked physical indices.
        #
        # - source: Use _physical_gpu_index (tracked physical index of current GPU)
        #           Falls back to extracting from _physical_device only on first migration
        # - target: The target_device string (e.g., "cuda:2") refers to the PHYSICAL GPU
        #           the user wants to migrate to, so we extract the index directly.
        #
        # TODO(multi-gpu): For multi-GPU, track physical index per device
        if _physical_gpu_index >= 0:
            # After first migration, use tracked physical index
            source_idx = _physical_gpu_index
        else:
            # First migration - extract from device string (still valid)
            source_idx = int(_extract_gpu_index(_physical_device))

        # Target is always the physical GPU index the user requested
        target_idx = int(_extract_gpu_index(target_device))

        print(f"\n[flexium] === ZERO-RESIDUE MIGRATION: GPU {source_idx} -> GPU {target_idx} ===")
        sys.stdout.flush()

        # Get target GPU info using CACHED UUID mapping (NOT pynvml!)
        # After driver migration remapping, pynvml indices get remapped at the process level,
        # so we use the cached mapping from startup when indices were correct.
        from flexium.utils.gpu_info import discover_gpu_pid, GPUInfo

        if _DEBUG:
            print(f"[flexium] DEBUG MIGRATION: _physical_device={_physical_device}, target_device={target_device}")
            print(f"[flexium] DEBUG MIGRATION: _physical_gpu_index={_physical_gpu_index} (tracked)")
            print(f"[flexium] DEBUG MIGRATION: source_idx={source_idx}, target_idx={target_idx}")
            print(f"[flexium] DEBUG MIGRATION: Current state: _physical_gpu_uuid={_physical_gpu_uuid[:16] if _physical_gpu_uuid else 'NOT SET'}...")
            print(f"[flexium] DEBUG MIGRATION: Current state: _initial_gpu_uuid={_initial_gpu_uuid[:16] if _initial_gpu_uuid else 'NOT SET'}...")
            print(f"[flexium] DEBUG MIGRATION: Current state: _initial_gpu_index={_initial_gpu_index}")
            print(f"[flexium] DEBUG MIGRATION: Cached GPU map: {{{', '.join(f'{k}:{v[:16]}...' for k,v in _gpu_index_to_uuid.items())}}}")

        # Use CACHED UUID mapping instead of querying pynvml (which gets remapped after driver migration)
        target_uuid = _gpu_index_to_uuid.get(target_idx)
        target_name = _gpu_index_to_name.get(target_idx, "Unknown GPU")
        if _DEBUG:
            print(f"[flexium] DEBUG MIGRATION: target from cache (idx={target_idx}): uuid={target_uuid[:16] if target_uuid else 'NONE'}...")
        if target_uuid:
            target_gpu_info = GPUInfo(
                logical_index=target_idx,
                physical_index=target_idx,
                uuid=target_uuid,
                name=target_name,
                memory_total=0,  # We don't need memory_total for migration
            )
        else:
            target_gpu_info = None

        # Source GPU info from cache
        source_uuid = _gpu_index_to_uuid.get(source_idx)
        source_name = _gpu_index_to_name.get(source_idx, "Unknown GPU")
        if _DEBUG:
            print(f"[flexium] DEBUG MIGRATION: source from cache (idx={source_idx}): uuid={source_uuid[:16] if source_uuid else 'NONE'}...")
        if source_uuid:
            source_gpu_info = GPUInfo(
                logical_index=source_idx,
                physical_index=source_idx,
                uuid=source_uuid,
                name=source_name,
                memory_total=0,
            )
        else:
            source_gpu_info = None

        # Discover GPU PID now (will be the same PID after migration, just on different GPU)
        current_gpu_pid = discover_gpu_pid(_physical_device)

        # Build device mapping
        device_map = _build_device_map(source_idx, target_idx)
        if not device_map:
            logger.error("Failed to build device map for migration")
            print("[flexium] ERROR: Failed to build device map")
            _migration_in_progress = False
            return False

        # Step 1: Prepare migration
        print("[flexium] Preparing migration...")
        if not _driver_lock(pid):
            logger.error("Driver lock failed")
            print("[flexium] ERROR: Driver lock failed")
            _migration_in_progress = False
            return False

        try:
            # Step 2: Save GPU state
            print("[flexium] Saving GPU state...")
            if not _driver_capture(pid):
                logger.error("Driver capture failed")
                print("[flexium] ERROR: Driver capture failed")
                _driver_unlock(pid)
                _migration_in_progress = False
                return False

            print("[flexium] GPU resources released from source")

            # Step 3: Restore to target GPU
            print("[flexium] Restoring to target GPU...")

            if not _driver_restore(pid, device_map=device_map):
                logger.error("Migration restore failed!")
                logger.info("Attempting recovery...")
                if not _driver_restore(pid):
                    logger.error("Recovery failed!")
                _driver_unlock(pid)
                _migration_in_progress = False
                return False

            # Update tracking - physical device changed
            # Cache GPU UUID/name/index/pid for heartbeat (get_gpu_info doesn't work after migration
            # because device mapping is remapped at process level but pynvml sees system view)
            # IMPORTANT: After driver migration, pynvml ALWAYS sees our process on the INITIAL GPU index
            # (the GPU we first started on), regardless of how many migrations happen.
            _physical_device = target_device

            # Debug: Log GPU info before updating
            if _DEBUG:
                print(f"[flexium] DEBUG: source_idx={source_idx}, target_idx={target_idx}")
                print(f"[flexium] DEBUG: source_gpu_info={source_gpu_info.uuid[:16] if source_gpu_info else 'None'}...")
                print(f"[flexium] DEBUG: target_gpu_info={target_gpu_info.uuid[:16] if target_gpu_info else 'None'}...")

            # Only set _initial_gpu_index/_initial_gpu_uuid on FIRST migration (they never change after that)
            # These track WHERE pynvml sees our memory, not where we logically are
            if _initial_gpu_index < 0:
                _initial_gpu_index = source_idx
                if source_gpu_info:
                    _initial_gpu_uuid = source_gpu_info.uuid
                if _DEBUG:
                    print(f"[flexium] DEBUG: Set _initial_gpu_index={_initial_gpu_index}, _initial_gpu_uuid={_initial_gpu_uuid[:16]}...")
            if target_gpu_info:
                _physical_gpu_uuid = target_gpu_info.uuid
                _physical_gpu_name = target_gpu_info.name
                _physical_gpu_index = target_gpu_info.physical_index
                if _DEBUG:
                    print(f"[flexium] DEBUG: Set _physical_gpu_uuid={_physical_gpu_uuid[:16]}..., _physical_gpu_index={_physical_gpu_index}")
            if current_gpu_pid:
                _physical_gpu_pid = current_gpu_pid

            # Log memory status
            mem_allocated = torch.cuda.memory_allocated(source_idx)
            mem_reserved = torch.cuda.memory_reserved(source_idx)
            print(f"[flexium] Memory: {mem_allocated/1e6:.1f} MB allocated, {mem_reserved/1e6:.1f} MB reserved")

            print("[flexium] === MIGRATION COMPLETE (zero VRAM residue) ===\n")
            sys.stdout.flush()

            # Notify orchestrator with the PHYSICAL device we're now on
            # Use target_gpu_info captured BEFORE driver migration (pynvml can hang after restore)
            if _orchestrator_client:
                try:
                    _orchestrator_client.complete_migration(
                        _process_id,
                        target_device,
                        gpu_uuid=target_gpu_info.uuid if target_gpu_info else "",
                        gpu_name=target_gpu_info.name if target_gpu_info else "",
                        memory_reserved=mem_reserved,
                    )
                except Exception as e:
                    logger.warning(f"Failed to notify orchestrator: {e}")

            return True

        finally:
            # Always unlock
            _driver_unlock(pid)

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()

        _current_device = old_device
        _physical_device = old_device
        return False

    finally:
        _migration_in_progress = False


def _do_migration(target_device: str) -> bool:
    """Perform GPU-to-GPU migration using driver-level state management.

    Only GPU → GPU migrations are supported. CPU is not a valid migration target.
    Use pause/resume for freeing GPU memory temporarily.

    Requires NVIDIA driver 580+ for zero-residue migration.

    Parameters:
        target_device: Target GPU device string (e.g., "cuda:0", "cuda:1").

    Returns:
        True if migration succeeded, False otherwise.
    """
    print(f"[flexium] _do_migration: current={_current_device}, target={target_device}")
    sys.stdout.flush()

    # Check if migration is enabled (environment requirements met)
    if not _migration_enabled:
        logger.warning("Migration requested but disabled (requirements not met)")
        print("[flexium] WARNING: Migration disabled - requirements for Flexium migration are not met.")
        sys.stdout.flush()
        return False

    # Validate target is a CUDA device
    if not target_device.startswith("cuda"):
        logger.error(f"Invalid migration target: {target_device}. Only GPU targets supported.")
        print(f"[flexium] ERROR: Invalid target '{target_device}'. Only GPU-to-GPU migration supported.")
        print("[flexium] Use pause/resume to free GPU memory temporarily.")
        sys.stdout.flush()
        return False

    # Validate we're currently on a GPU
    if not _current_device.startswith("cuda"):
        logger.error(f"Cannot migrate from {_current_device}. Must be on a GPU.")
        print(f"[flexium] ERROR: Current device is '{_current_device}'. Must be on GPU to migrate.")
        sys.stdout.flush()
        return False

    # Check driver interface availability (redundant but explicit)
    if not _check_driver_interface_available():
        logger.error("Driver interface not available. Driver 580+ required for migration.")
        print("[flexium] ERROR: Driver interface not available.")
        print("[flexium] GPU migration requires NVIDIA driver 580+.")
        sys.stdout.flush()
        return False

    return _do_migration_with_driver(target_device)


def _do_pause() -> None:
    """Pause training and wait for resume command.

    Uses driver-level state management to checkpoint GPU state (freeing GPU memory
    to 0 MB), then waits for a resume command to restore to a GPU.

    Requires NVIDIA driver 580+ for zero-residue pause.
    Only GPU pause/resume is supported (no CPU).
    """
    global _physical_device, _current_device, _pause_in_progress
    global _cached_memory_allocated, _cached_memory_reserved, _cached_visible_devices

    # Cache memory and device info BEFORE pausing (pynvml may hang after checkpoint)
    try:
        import torch
        from flexium.utils.gpu_info import get_all_device_reports, get_estimated_gpu_memory
        import socket

        if _physical_device.startswith("cuda") and torch.cuda.is_available():
            _cached_memory_allocated = get_estimated_gpu_memory(_physical_device)
            _cached_memory_reserved = _cached_memory_allocated
        _cached_visible_devices = get_all_device_reports(socket.gethostname())
    except Exception as e:
        logger.warning(f"Failed to cache memory/devices before pause: {e}")

    # Set flag to prevent heartbeat thread from triggering another migration
    # while we're handling the pause/resume internally
    _pause_in_progress = True

    # Mark as paused for reconnection handling (with cached memory for display)
    if _orchestrator_client:
        _orchestrator_client.set_paused(True, memory_reserved=_cached_memory_reserved)

    print("\n[flexium] === PAUSING ===")
    print(f"[flexium] _current_device={_current_device}, _physical_device={_physical_device}")
    sys.stdout.flush()

    # Check if migration/pause is enabled (environment requirements met)
    if not _migration_enabled:
        logger.warning("Pause requested but disabled (requirements not met)")
        print("[flexium] WARNING: Pause disabled - requirements for Flexium migration are not met.")
        sys.stdout.flush()
        _pause_in_progress = False
        return

    # Validate we're on a GPU
    if not _current_device.startswith("cuda"):
        logger.error(f"Cannot pause from {_current_device}. Must be on GPU.")
        print(f"[flexium] ERROR: Cannot pause from '{_current_device}'. Must be on GPU.")
        sys.stdout.flush()
        _pause_in_progress = False
        return

    # Check driver interface availability (redundant but explicit)
    if not _check_driver_interface_available():
        logger.error("Driver interface not available. Driver 580+ required for pause.")
        print("[flexium] ERROR: Driver interface not available.")
        print("[flexium] Pause requires NVIDIA driver 580+.")
        sys.stdout.flush()
        _pause_in_progress = False
        return

    paused_device = _physical_device
    gpu_uuids = None

    # Get GPU UUIDs BEFORE checkpointing - we need these for device mapping on resume
    # After checkpoint, pynvml calls may hang because the CUDA context is suspended
    gpu_uuids = _get_all_gpu_uuids()
    if gpu_uuids:
        print(f"[flexium] Cached {len(gpu_uuids)} GPU UUIDs for resume")

    # Checkpoint GPU state using driver interface
    pid = os.getpid()
    print("[flexium] Checkpointing GPU state (zero-residue)...")
    sys.stdout.flush()

    try:
        # Lock -> Checkpoint (this frees the GPU)
        if not _driver_lock(pid):
            logger.error("Lock failed, cannot pause")
            print("[flexium] ERROR: Driver lock failed")
            sys.stdout.flush()
            _pause_in_progress = False
            return

        if not _driver_capture(pid):
            logger.error("Checkpoint failed, cannot pause")
            print("[flexium] ERROR: Driver capture failed")
            _driver_unlock(pid)
            sys.stdout.flush()
            _pause_in_progress = False
            return

        print("[flexium] GPU memory freed (0 MB used)")

    except Exception as e:
        logger.error(f"Failed to checkpoint for pause: {e}")
        print(f"[flexium] ERROR: Checkpoint failed: {e}")
        sys.stdout.flush()
        _pause_in_progress = False
        return

    print("[flexium] Waiting for resume command...")
    sys.stdout.flush()

    # Notify orchestrator we're paused (preserve memory info for dashboard display)
    if _orchestrator_client:
        try:
            _orchestrator_client.complete_migration(
                _process_id,
                "__PAUSED__",
                gpu_uuid=_physical_gpu_uuid,
                gpu_name=_physical_gpu_name,
                memory_reserved=_cached_memory_reserved,
            )
        except Exception as e:
            logger.warning(f"Failed to notify orchestrator of pause: {e}")

    # Wait for resume (poll for migration target)
    while True:
        time.sleep(1.0)

        # Send heartbeat to check for resume
        if _orchestrator_client:
            try:
                response = _orchestrator_client.heartbeat(
                    memory_allocated=0,
                    memory_reserved=0,
                    device="__PAUSED__",
                    visible_devices=_cached_visible_devices,
                )

                if response and response.get("should_migrate") and response.get("target_device"):
                    target_device = response["target_device"]
                    if target_device != "__PAUSE__":
                        # Validate target is a GPU
                        if not target_device.startswith("cuda"):
                            logger.warning(f"Invalid resume target: {target_device}. Only GPU supported.")
                            print(f"[flexium] WARNING: Cannot resume to '{target_device}'. Only GPU targets supported.")
                            sys.stdout.flush()
                            continue

                        # Resume to the specified GPU
                        print(f"\n[flexium] === RESUMING to {target_device} ===")
                        print(f"[flexium] Paused from: {paused_device}")
                        sys.stdout.flush()

                        # Restore from checkpoint to target GPU
                        success = _do_resume_from_checkpoint(
                            paused_device, target_device, cached_gpu_uuids=gpu_uuids
                        )

                        if not success:
                            print("[flexium] WARNING: Resume failed!")
                        else:
                            print("[flexium] === RESUMED ===")
                            # Mark as no longer paused
                            if _orchestrator_client:
                                _orchestrator_client.set_paused(False)

                        sys.stdout.flush()

                        # Note: We're called from _send_heartbeat() which
                        # already holds _migration_lock, so we can't acquire it again.
                        _pause_in_progress = False
                        return

            except Exception as e:
                # Connection error - attempt reconnection
                logger.warning(f"Pause heartbeat error: {e}")
                print("[flexium] Lost connection to orchestrator during pause")
                sys.stdout.flush()

                # Try to reconnect with timeout, then auto-resume if orchestrator stays down
                reconnect_interval = 5.0  # seconds between attempts
                max_reconnect_time = 300.0  # 5 minutes max before auto-resume
                reconnect_start = time.time()
                reconnected = False

                while True:
                    elapsed = time.time() - reconnect_start
                    remaining = max_reconnect_time - elapsed

                    if remaining <= 0:
                        print(f"[flexium] Orchestrator reconnect timeout ({max_reconnect_time:.0f}s)")
                        print(f"[flexium] Auto-resuming on last device: {paused_device}")
                        sys.stdout.flush()
                        logger.warning(f"Orchestrator timeout - auto-resuming on {paused_device}")

                        # Resume on the device we were paused from
                        success = _do_resume_from_checkpoint(
                            paused_device, paused_device, cached_gpu_uuids=gpu_uuids
                        )

                        if success:
                            print("[flexium] === AUTO-RESUMED (local mode) ===")
                            print("[flexium] Running without orchestrator. Will reconnect when available.")
                            # Mark as no longer paused
                            if _orchestrator_client:
                                _orchestrator_client.set_paused(False)
                        else:
                            print("[flexium] WARNING: Auto-resume failed!")

                        sys.stdout.flush()
                        _pause_in_progress = False
                        return

                    print(f"[flexium] Attempting to reconnect... ({remaining:.0f}s remaining)")
                    sys.stdout.flush()

                    if _attempt_reconnect():
                        print("[flexium] Reconnected to orchestrator!")
                        sys.stdout.flush()
                        reconnected = True
                        break
                    else:
                        print(f"[flexium] Reconnect failed, retrying in {reconnect_interval}s...")
                        sys.stdout.flush()
                        time.sleep(reconnect_interval)

                if reconnected:
                    # Continue the pause loop - server will send resume command
                    continue

    # Should not reach here, but clear flag just in case
    _pause_in_progress = False


def _do_resume_from_checkpoint(
    paused_device: str,
    target_device: str,
    cached_gpu_uuids: Optional[List[str]] = None,
) -> bool:
    """Resume from a checkpointed (paused) state.

    Restores GPU state from checkpoint to the target device.

    Parameters:
        paused_device: Device the process was on when paused.
        target_device: Device to resume on.
        cached_gpu_uuids: Pre-cached GPU UUIDs (needed because pynvml may hang
                         after checkpoint since CUDA context is suspended).

    Returns:
        True if resume succeeded, False otherwise.
    """
    global _current_device, _physical_device

    pid = os.getpid()
    source_idx = int(_extract_gpu_index(paused_device))
    target_idx = int(_extract_gpu_index(target_device))

    print("[flexium] Restoring GPU state from checkpoint...")
    print(f"[flexium] Source device: {paused_device} (index {source_idx})")
    print(f"[flexium] Target device: {target_device} (index {target_idx})")
    print(f"[flexium] Current _current_device: {_current_device}")
    print(f"[flexium] Current _physical_device: {_physical_device}")
    sys.stdout.flush()

    try:
        # For pause/resume, we use a two-step approach when changing devices:
        # 1. First restore to the ORIGINAL device (no device-map needed)
        # 2. Then do a standard migration to the target device
        #
        # This is more reliable than trying to use device-map for resume,
        # especially after previous transparent migrations.

        if source_idx != target_idx:
            print("[flexium] Resume to different device: will restore then migrate")
            print(f"[flexium] Step 1: Restoring to original device (cuda:{source_idx})...")
            sys.stdout.flush()

            # Restore to original device first
            if not _driver_restore(pid):
                logger.error("Restore failed")
                print("[flexium] ERROR: Driver restore failed")
                _driver_unlock(pid)
                return False

            # Unlock after restore
            if not _driver_unlock(pid):
                logger.warning("Unlock failed but restore succeeded")

            # Update tracking to reflect we're back on original device
            _physical_device = paused_device

            print(f"[flexium] Step 2: Migrating from cuda:{source_idx} to cuda:{target_idx}...")
            sys.stdout.flush()

            # Now do a standard migration to the target device
            # This will use the regular migration path which handles device changes properly
            success = _do_migration(target_device)
            if not success:
                logger.error("Migration after restore failed")
                print("[flexium] ERROR: Migration to target device failed")
                return False

            # _do_migration already updates _physical_device, so we're done
            print(f"[flexium] Successfully resumed to {target_device}")
            sys.stdout.flush()

            # Notify orchestrator (if _do_migration didn't already)
            # Actually _do_migration calls complete_migration, so skip here
            return True

        else:
            # Same device - just restore, no migration needed
            print("[flexium] Restoring to same device (no migration needed)")
            if not _driver_restore(pid):
                logger.error("Restore failed")
                print("[flexium] ERROR: Driver restore failed")
                _driver_unlock(pid)
                return False

            # Unlock
            if not _driver_unlock(pid):
                logger.warning("Unlock failed but restore succeeded")

            # Update tracking - we're back on the paused device
            _physical_device = paused_device

            print(f"[flexium] Restored to {paused_device}")
            sys.stdout.flush()

            # Notify orchestrator
            if _orchestrator_client:
                try:
                    from flexium.utils.gpu_info import get_gpu_info, get_estimated_gpu_memory
                    gpu_info = get_gpu_info(paused_device)
                    mem_reserved = get_estimated_gpu_memory(paused_device)
                    _orchestrator_client.complete_migration(
                        _process_id,
                        paused_device,
                        gpu_uuid=gpu_info.uuid if gpu_info else "",
                        gpu_name=gpu_info.name if gpu_info else "",
                        memory_reserved=mem_reserved,
                    )
                except Exception as e:
                    logger.warning(f"Failed to notify orchestrator: {e}")

            return True

    except Exception as e:
        logger.error(f"Resume from checkpoint failed: {e}")
        import traceback
        traceback.print_exc()
        return False




def _send_heartbeat() -> None:
    """Send heartbeat to orchestrator with visible device reports."""
    if _orchestrator_client is None:
        return

    try:
        import torch
        from flexium.utils.gpu_info import (
            discover_gpu_pid,
            get_estimated_gpu_memory,
            get_gpu_info,
        )

        # Get memory info - use pynvml for accurate nvidia-smi matching memory
        memory_allocated = 0
        memory_reserved = 0
        gpu_uuid = ""
        gpu_name = ""
        gpu_pid = 0
        container_pid = os.getpid()

        if _physical_device.startswith("cuda") and torch.cuda.is_available():
            # After driver migration, device mapping is remapped at process level
            # but pynvml sees system-level view. Use cached values if available.
            if _physical_gpu_index >= 0:
                # After driver migration: pynvml ALWAYS sees our process on the INITIAL GPU
                # (the first GPU we started on), regardless of how many migrations happen.
                from flexium.utils.gpu_info import get_gpu_memory_by_physical_index
                query_idx = _initial_gpu_index if _initial_gpu_index >= 0 else _physical_gpu_index
                memory_allocated = get_gpu_memory_by_physical_index(
                    query_idx, _physical_gpu_pid if _physical_gpu_pid > 0 else None
                )
                memory_reserved = memory_allocated
                gpu_uuid = _physical_gpu_uuid
                gpu_name = _physical_gpu_name
            else:
                # No migration yet - use pynvml query (matches nvidia-smi)
                memory_allocated = get_estimated_gpu_memory(_physical_device)
                memory_reserved = memory_allocated
                gpu_info = get_gpu_info(_physical_device)
                if gpu_info:
                    gpu_uuid = gpu_info.uuid
                    gpu_name = gpu_info.name

            # Get GPU PID (host PID as seen by nvidia-smi)
            discovered_pid = discover_gpu_pid(_physical_device)
            if discovered_pid is not None:
                gpu_pid = discovered_pid
        # When on CPU, don't query any GPU - just report 0

        # Cache memory for reconnect during pause (when pynvml might hang)
        global _cached_memory_allocated, _cached_memory_reserved
        _cached_memory_allocated = memory_allocated
        _cached_memory_reserved = memory_reserved

        # Determine pynvml_gpu_uuid - where pynvml actually sees our memory
        # After driver migration, this is the INITIAL GPU, not the target GPU
        pynvml_gpu_uuid = _initial_gpu_uuid if _initial_gpu_uuid else gpu_uuid

        # Debug: Log heartbeat data periodically (only when _DEBUG is enabled)
        if _DEBUG:
            global _heartbeat_debug_counter
            try:
                _heartbeat_debug_counter += 1
            except NameError:
                _heartbeat_debug_counter = 1
            if _heartbeat_debug_counter % 5 == 1:  # Every 5th heartbeat
                print(f"[flexium] DEBUG HEARTBEAT: device={_physical_device}, gpu_uuid={gpu_uuid[:16] if gpu_uuid else 'NONE'}..., pynvml_gpu_uuid={pynvml_gpu_uuid[:16] if pynvml_gpu_uuid else 'NONE'}..., memory={memory_reserved/1e6:.1f}MB")
                print(f"[flexium] DEBUG HEARTBEAT: _physical_gpu_uuid={_physical_gpu_uuid[:16] if _physical_gpu_uuid else 'NOT SET'}..., _initial_gpu_uuid={_initial_gpu_uuid[:16] if _initial_gpu_uuid else 'NOT SET'}...")
                sys.stdout.flush()

        # Get CUDA device count (respects CUDA_VISIBLE_DEVICES)
        # This is used to filter migration dropdown options in the dashboard
        cuda_device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Get all device reports to populate device registry in dashboard
        from flexium.utils.gpu_info import get_all_device_reports
        visible_devices = get_all_device_reports(socket.gethostname())

        # Send heartbeat via WebSocket client
        response = _orchestrator_client.heartbeat(
            memory_allocated=memory_allocated,
            memory_reserved=memory_reserved,
            gpu_pid=gpu_pid,
            container_pid=container_pid,
            pynvml_gpu_uuid=pynvml_gpu_uuid,
            cuda_device_count=cuda_device_count,
            device=_physical_device,
            gpu_uuid=gpu_uuid,
            gpu_name=gpu_name,
            visible_devices=visible_devices,
        )

        if response and response.get("should_migrate") and response.get("target_device"):
            # Don't migrate if we're inside _do_pause()
            # The pause heartbeat loop handles resume internally
            if not _pause_in_progress:
                target = response["target_device"]
                logger.info(f"Migration requested: {target}")

                # Execute migration directly from heartbeat thread.
                # This is safe because driver migration atomically freezes the process
                # during migration - no concurrent access issues.
                with _migration_lock:
                    if target == "__PAUSE__":
                        _do_pause()
                    else:
                        _do_migration(target)
            else:
                logger.debug(
                    f"Ignoring migration request during pause: {response.get('target_device')}"
                )

    except Exception as e:
        # Connection error - the client handles reconnection internally
        logger.debug(f"Heartbeat error: {e}")


def _heartbeat_loop() -> None:
    """Background heartbeat loop."""
    while not _stop_heartbeat.is_set():
        # Don't send heartbeats during migration or pause
        # During pause, _do_pause() has its own heartbeat loop
        if not _migration_in_progress and not _pause_in_progress:
            _send_heartbeat()
        _stop_heartbeat.wait(timeout=3.0)


def _attempt_reconnect() -> bool:
    """Attempt to reconnect to orchestrator after connection loss.

    Re-registers with the orchestrator using stored credentials.

    Returns:
        True if reconnection succeeded, False otherwise.
    """
    global _orchestrator_client

    if _orchestrator_client is None:
        return False

    try:
        # Use cached GPU info (don't call pynvml - it may hang if paused)
        gpu_uuid = ""
        gpu_name = ""
        if _physical_device:
            # Extract GPU index from device string (e.g., "cuda:0" -> 0)
            try:
                gpu_idx = int(_physical_device.split(":")[-1])
                gpu_uuid = _gpu_index_to_uuid.get(gpu_idx, "")
                gpu_name = _gpu_index_to_name.get(gpu_idx, "")
            except (ValueError, IndexError):
                pass

        # Re-connect via WebSocket client
        if not _orchestrator_client.connect():
            print("[flexium] Reconnection failed: could not connect")
            sys.stdout.flush()
            return False

        print("[flexium] Connected, sending re-registration...")
        sys.stdout.flush()

        # Re-register with stored settings
        result = _orchestrator_client.register(
            process_id=_process_id,
            device=_physical_device,
            metadata=_orchestrator_client._metadata or {},
            min_gpus=getattr(_orchestrator_client, '_min_gpus', 1),
            max_gpus=getattr(_orchestrator_client, '_max_gpus', 1),
            max_vram=getattr(_orchestrator_client, '_max_vram', 0),
            can_share=getattr(_orchestrator_client, '_can_share', True),
            priority=getattr(_orchestrator_client, '_priority', 50),
            preemptible=getattr(_orchestrator_client, '_preemptible', True),
            migratable=getattr(_orchestrator_client, '_migratable', True),
            start_time=_start_time,
        )

        print(f"[flexium] Registration result: {result}")
        sys.stdout.flush()

        if result:
            # Send a heartbeat with cached device info so server knows about GPUs
            # (Can't call pynvml during pause - it may hang)
            if _cached_visible_devices:
                try:
                    _orchestrator_client.heartbeat(
                        memory_allocated=_cached_memory_allocated,
                        memory_reserved=_cached_memory_reserved,
                        device=_physical_device,
                        gpu_uuid=gpu_uuid,
                        gpu_name=gpu_name,
                    )
                    logger.debug(f"Sent reconnect heartbeat with cached device info")
                except Exception as e:
                    logger.warning(f"Failed to send reconnect heartbeat: {e}")

            # If we're paused, notify server about paused state
            if _pause_in_progress:
                try:
                    _orchestrator_client.complete_migration(
                        _process_id,
                        "__PAUSED__",
                        gpu_uuid=gpu_uuid,
                        gpu_name=gpu_name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to notify paused state: {e}")

            # Don't print here - callers will print their own success message
            return True
        else:
            print("[flexium] Reconnection rejected by server")
            sys.stdout.flush()
            return False

    except Exception as e:
        print(f"[flexium] Reconnection failed: {e}")
        sys.stdout.flush()
        return False


def _cache_gpu_info_at_startup() -> None:
    """Cache GPU UUID/name mapping at startup before any driver migration remapping.

    After driver migration migration, pynvml indices get remapped at the process level,
    so we can't rely on pynvml to give us correct UUIDs by index. We cache the mapping
    at startup when we know the indices are correct.

    This cache maps physical GPU index (0, 1, 2, ...) to GPU UUID and name.
    It must be called BEFORE any driver migration operations.

    TODO(multi-gpu): This function already supports multi-GPU - it caches all GPUs.
    For multi-GPU processes, ensure this is called before any GPU is used.
    """
    global _gpu_index_to_uuid, _gpu_index_to_name

    if _gpu_index_to_uuid:
        # Already cached
        return

    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            uuid = pynvml.nvmlDeviceGetUUID(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            _gpu_index_to_uuid[i] = uuid
            _gpu_index_to_name[i] = name
        logger.debug(f"Cached GPU info: {_gpu_index_to_uuid}")
    except Exception as e:
        logger.debug(f"Failed to cache GPU info: {e}")


def _connect_orchestrator(config: FlexiumConfig) -> None:
    """Connect to orchestrator server."""
    global _orchestrator_client

    # Cache GPU info BEFORE any driver migration migrations
    _cache_gpu_info_at_startup()

    if not config.orchestrator:
        return

    try:
        from flexium.orchestrator.client import OrchestratorClient
        from flexium import _driver

        _orchestrator_client = OrchestratorClient(
            config.orchestrator,
            heartbeat_interval=config.heartbeat_interval,
        )

        # Check if migration is supported (driver 580+)
        # If config.migratable is False, respect that (user disabled it)
        # If config.migratable is True, check driver capability
        can_migrate = config.migratable and _driver.supports_migration()

        result = _orchestrator_client.register(
            process_id=_process_id,
            device=_current_device,
            min_gpus=config.min_gpus,
            max_gpus=config.max_gpus,
            max_vram=config.max_vram,
            can_share=config.can_share,
            priority=config.priority,
            preemptible=config.preemptible,
            migratable=can_migrate,
            start_time=_start_time,
        )

        if result:
            print("[flexium] Registered with orchestrator")
        else:
            # Keep client alive for reconnection attempts
            print("[flexium] WARNING: Registration failed, will retry periodically")

    except Exception as e:
        logger.error(f"Failed to connect to orchestrator: {e}")
        # Keep client alive for reconnection attempts if it was created
        # _orchestrator_client stays as-is (may be None or valid)


def _disconnect_orchestrator() -> None:
    """Disconnect from orchestrator."""
    global _orchestrator_client, _heartbeat_thread

    _stop_heartbeat.set()
    if _heartbeat_thread:
        _heartbeat_thread.join(timeout=2.0)
        _heartbeat_thread = None

    if _orchestrator_client:
        try:
            _orchestrator_client.unregister(_process_id)
            print("[flexium] Unregistered from orchestrator")
        except Exception:
            pass
        _orchestrator_client = None


@contextmanager
def run(
    orchestrator: Optional[str] = None,
    device: Optional[str] = None,
    disabled: bool = False,
) -> Iterator[None]:
    """Run training with transparent GPU management.

    This context manager enables automatic migration support.
    Migrations are triggered by the heartbeat thread and happen
    transparently - no changes needed to user code.

    Parameters:
        orchestrator: Orchestrator address (host:port).
        device: Initial device to use.
        disabled: If True, bypass flexium entirely.

    Yields:
        None

    Example:
        with flexium.auto.run():
            model = Net().cuda()  # Routed to managed device
            optimizer = Adam(model.parameters())

            for epoch in range(100):
                for batch in dataloader:
                    # Training code - unchanged
                    # Migration happens via heartbeat thread (transparent)
                    loss.backward()
                    optimizer.step()
    """
    global _current_device, _physical_device, _process_id, _heartbeat_thread, _start_time

    if disabled:
        print("[flexium] Running in DISABLED mode")
        yield
        return

    # Load config
    config = load_config(orchestrator=orchestrator, device=device)

    _current_device = config.device
    _physical_device = config.device
    _process_id = f"gpu-{uuid.uuid4().hex[:8]}"
    _start_time = time.time()  # Track when process started for runtime display

    # Verify environment requirements for migration
    # This sets _migration_enabled = False if requirements not met
    env_ok = _verify_environment()

    # Show config
    print(f"[flexium] Process: {_process_id}")
    print(f"[flexium] Device:  {_current_device}")
    # Show capability status based on driver version
    if _migration_enabled:
        from flexium import _driver
        if _driver.supports_migration():
            print("[flexium] Capabilities: pause, resume, migrate (driver 580+)")
        else:
            print("[flexium] Capabilities: pause, resume (driver 550+, migration requires 580+)")
    else:
        print("[flexium] Capabilities: DISABLED (requirements not met)")
    if config.orchestrator:
        print(f"[flexium] Orchestrator: {config.orchestrator}")
    else:
        print_no_orchestrator_warning()
    print("-" * 50)
    sys.stdout.flush()

    # Connect to orchestrator
    _connect_orchestrator(config)

    # Start heartbeat
    if _orchestrator_client:
        _stop_heartbeat.clear()
        _heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
        _heartbeat_thread.start()

    try:
        yield
    finally:
        _disconnect_orchestrator()


# =============================================================================
# GPU Error Recovery
# =============================================================================

# Error types that can be recovered via migration
_RECOVERABLE_CUDA_ERRORS = {
    "OOM": ["out of memory", "CUDA out of memory"],
    "ECC": ["uncorrectable ECC error", "ECC error"],
    "DEVICE_ASSERT": ["device-side assert", "CUDA error: device assert"],
    "ILLEGAL_ACCESS": ["illegal memory access", "an illegal memory access was encountered"],
    "LAUNCH_FAILURE": ["launch failure", "unspecified launch failure"],
}


def _classify_cuda_error(error: BaseException) -> tuple[str, str]:
    """Classify a CUDA error by type.

    Parameters:
        error: The exception to classify.

    Returns:
        Tuple of (error_type, error_message). error_type is one of:
        "OOM", "ECC", "DEVICE_ASSERT", "ILLEGAL_ACCESS", "LAUNCH_FAILURE", "UNKNOWN"
    """
    import torch

    error_msg = str(error).lower()

    # Check for OutOfMemoryError specifically
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return "OOM", str(error)

    # Check for RuntimeError with CUDA error messages
    if isinstance(error, RuntimeError):
        for error_type, patterns in _RECOVERABLE_CUDA_ERRORS.items():
            for pattern in patterns:
                if pattern.lower() in error_msg:
                    return error_type, str(error)

    return "UNKNOWN", str(error)


def _estimate_memory_needed(error_msg: str) -> int:
    """Estimate memory needed from OOM error message.

    Parameters:
        error_msg: The OOM error message.

    Returns:
        Estimated memory needed in bytes, or 0 if cannot parse.
    """
    import re

    # Try to parse "Tried to allocate X.XX GiB"
    match = re.search(r"Tried to allocate ([\d.]+)\s*(GiB|MiB|GB|MB)", error_msg)
    if match:
        value = float(match.group(1))
        unit = match.group(2).upper()
        if "GI" in unit or "G" in unit:
            return int(value * 1024 * 1024 * 1024)
        else:
            return int(value * 1024 * 1024)

    return 0


def _clear_cuda_error_state() -> None:
    """Clear CUDA error state after an error.

    This attempts to recover CUDA to a usable state after an error.
    """
    try:
        import torch

        # Synchronize to ensure all pending operations complete/fail
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Clear cached memory
        torch.cuda.empty_cache()

        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()

        logger.debug("CUDA error state cleared")
    except Exception as e:
        logger.warning(f"Failed to clear CUDA error state: {e}")


def _request_recovery_target_local(error_type: str, memory_needed: int = 0) -> Optional[str]:
    """Find a recovery target GPU locally (standalone mode).

    Scans available GPUs and picks the best candidate for migration,
    avoiding the current (failed) GPU.

    Parameters:
        error_type: Type of error ("OOM", "ECC", etc.)
        memory_needed: For OOM, estimated bytes needed.

    Returns:
        Target device string (e.g., "cuda:1") or None if no target available.
    """
    global _current_device, _physical_gpu_uuid, _failed_gpus
    import torch

    try:
        device_count = torch.cuda.device_count()
        if device_count <= 1:
            logger.warning("No alternative GPUs available for recovery")
            return None

        # Get current logical device index
        current_idx = 0
        if _current_device and _current_device.startswith("cuda:"):
            try:
                current_idx = int(_current_device.split(":")[1])
            except (ValueError, IndexError):
                pass

        # Try to get memory info for all GPUs
        candidates = []
        for idx in range(device_count):
            if idx == current_idx:
                continue  # Skip current (failed) GPU

            # Check if this GPU was previously marked as failed
            device_str = f"cuda:{idx}"
            if device_str in _failed_gpus:
                logger.debug(f"Skipping {device_str}: previously failed")
                continue

            # Get GPU info including free memory
            try:
                import pynvml
                pynvml.nvmlInit()

                # Get physical index for this logical device
                from flexium.utils.gpu_info import _get_visible_device_indices
                visible_indices = _get_visible_device_indices()
                if idx >= len(visible_indices):
                    continue
                physical_idx = visible_indices[idx]

                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_free = mem_info.free
                memory_total = mem_info.total
                pynvml.nvmlShutdown()

                # For OOM errors, check if this GPU has enough memory
                if error_type == "OOM" and memory_needed > 0:
                    if memory_free < memory_needed:
                        logger.debug(
                            f"Skipping cuda:{idx}: insufficient memory "
                            f"({memory_free / 1e9:.2f} GB free, need {memory_needed / 1e9:.2f} GB)"
                        )
                        continue

                candidates.append({
                    "device": device_str,
                    "memory_free": memory_free,
                    "memory_total": memory_total,
                })

            except Exception as e:
                logger.debug(f"Could not get info for cuda:{idx}: {e}")
                # Still add as candidate with unknown memory
                candidates.append({
                    "device": device_str,
                    "memory_free": 0,
                    "memory_total": 0,
                })

        if not candidates:
            logger.warning("No suitable GPU found for recovery")
            return None

        # Sort by free memory (descending) - prefer GPU with most free memory
        candidates.sort(key=lambda x: x["memory_free"], reverse=True)

        target = candidates[0]["device"]
        logger.info(
            f"Local recovery: selected {target} "
            f"({candidates[0]['memory_free'] / 1e9:.2f} GB free)"
        )
        return target

    except Exception as e:
        logger.error(f"Failed to find local recovery target: {e}")
        return None


def _request_recovery_target(error_type: str, memory_needed: int = 0) -> Optional[str]:
    """Request a recovery target GPU from orchestrator or find one locally.

    Parameters:
        error_type: Type of error ("OOM", "ECC", etc.)
        memory_needed: For OOM, estimated bytes needed.

    Returns:
        Target device string (e.g., "cuda:1") or None if no target available.
    """
    global _orchestrator_client, _process_id, _current_device, _physical_gpu_uuid

    # Try orchestrator first if connected
    if _orchestrator_client is not None and not _orchestrator_client.connection_manager.is_local_mode:
        result = _orchestrator_client.request_error_recovery(
            process_id=_process_id,
            error_type=error_type,
            current_device=_current_device,
            memory_needed=memory_needed,
            current_gpu_uuid=_physical_gpu_uuid,
        )

        if result and result.get("target_device"):
            return result["target_device"]

        # Orchestrator didn't find a target, fall through to local search
        logger.info("Orchestrator found no target, trying local GPU search")

    # Standalone mode or orchestrator didn't find target - search locally
    return _request_recovery_target_local(error_type, memory_needed)


class _RecoverableAttempt:
    """Context manager for a single attempt within recoverable().

    This is yielded by the recoverable() iterator and handles exception
    suppression for retry logic.
    """

    def __init__(self, parent: "recoverable"):
        self._parent = parent
        self._success = False

    def __enter__(self) -> "_RecoverableAttempt":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        import torch

        # No exception - success!
        if exc_type is None:
            self._success = True
            return False

        # Check if it's a recoverable CUDA error
        if not isinstance(exc_val, (torch.cuda.OutOfMemoryError, RuntimeError)):
            return False  # Re-raise non-CUDA errors

        # Classify the error
        error_type, error_msg = _classify_cuda_error(exc_val)

        if error_type == "UNKNOWN":
            return False  # Re-raise unknown errors

        # Store error info for parent to handle
        self._parent._last_error = exc_val
        self._parent._last_error_type = error_type
        self._parent._last_error_msg = error_msg
        self._success = False

        # Suppress the exception - parent iterator will handle recovery
        return True


class recoverable:
    """Automatic GPU error recovery - migrate and optionally retry on CUDA errors.

    When a GPU error occurs (OOM, ECC, device assert, etc.), this will:
    1. Clear the CUDA error state
    2. Request a suitable GPU from the orchestrator
    3. Migrate to the new GPU

    There are THREE ways to use this, from simplest to most control:

    **Option 1: Simple context manager (LOSES the failed operation)**

        The simplest approach. If an error occurs, we migrate to a new GPU
        and continue. THE CURRENT OPERATION IS LOST - if you were in a
        training loop, that batch is skipped.

        ```python
        with flexium.auto.run():
            for batch in dataloader:
                with flexium.auto.recoverable():
                    output = model(batch.cuda())
                    loss.backward()
                    optimizer.step()
                # If OOM happened above, we're now on a new GPU
                # but that batch was lost - training continues with next batch
        ```

    **Option 2: Decorator (REPLAYS the failed operation)**

        Wrap your training step function. If an error occurs, we migrate
        and RETRY the function call with the same arguments.

        ```python
        @flexium.auto.recoverable(retries=3)
        def train_step(model, batch, optimizer):
            output = model(batch.cuda())
            loss = output.sum()
            loss.backward()
            optimizer.step()

        with flexium.auto.run():
            for batch in dataloader:
                train_step(model, batch, optimizer)  # Auto-retries on error
        ```

    **Option 3: Iterator pattern (REPLAYS, advanced)**

        Most control, but most verbose. You write the retry loop structure.

        ```python
        with flexium.auto.run():
            for batch in dataloader:
                for attempt in flexium.auto.recoverable(retries=3):
                    with attempt:
                        output = model(batch.cuda())
                        loss.backward()
                        optimizer.step()
        ```

    Parameters:
        retries: Maximum retry attempts for decorator/iterator patterns (default: 3).
                 Ignored for simple context manager (Option 1).

    Raises:
        RuntimeError: If recovery fails after max retries (decorator/iterator only).
        The original exception: If the error is not a recoverable CUDA error.
    """

    def __init__(self, retries_or_func: Any = None, *, retries: int = 3):
        """Initialize recoverable.

        Supports multiple calling conventions:
        - recoverable() - default 3 retries
        - recoverable(retries=5) - custom retries
        - @recoverable - decorator without parens (retries_or_func is the function)
        - @recoverable(retries=5) - decorator with config

        Parameters:
            retries_or_func: Either the wrapped function (callable) when used as
                           @recoverable without parens, or None.
            retries: Maximum retry attempts for decorator/iterator patterns (default: 3).
        """
        # Handle @recoverable (no parens) - first arg is the function
        if callable(retries_or_func):
            self._func: Optional[Callable] = retries_or_func
            self.max_retries = retries
        else:
            self._func = None
            self.max_retries = retries

        self._attempts = 0
        self._last_error: Optional[BaseException] = None
        self._last_error_type: str = ""
        self._last_error_msg: str = ""

    def __call__(self, *args, **kwargs):
        """Support decorator and direct call patterns."""
        # Case 1: @recoverable(retries=N) - we're being used as decorator factory
        # The first call after __init__ will have a function as the only arg
        if self._func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            self._func = args[0]
            return self

        # Case 2: Actually calling the wrapped function
        if self._func is not None:
            return self._call_with_retry(*args, **kwargs)

        # Case 3: Something unexpected
        raise TypeError("recoverable() missing required function to wrap")

    def _call_with_retry(self, *args, **kwargs):
        """Call the wrapped function with retry logic."""
        self._attempts = 0

        while self._attempts <= self.max_retries:
            self._attempts += 1
            try:
                return self._func(*args, **kwargs)
            except BaseException as e:
                import torch

                # Check if recoverable CUDA error
                if not isinstance(e, (torch.cuda.OutOfMemoryError, RuntimeError)):
                    raise

                error_type, error_msg = _classify_cuda_error(e)
                if error_type == "UNKNOWN":
                    raise

                self._last_error = e
                self._last_error_type = error_type
                self._last_error_msg = error_msg

                if self._attempts > self.max_retries:
                    raise RuntimeError(
                        f"GPU error recovery failed after {self.max_retries} retries. "
                        f"Original error: {e}"
                    ) from e

                # Handle recovery (migrate to new GPU)
                self._handle_recovery()

        # Should not reach here
        raise RuntimeError("Unexpected state in recoverable")

    def __iter__(self) -> Iterator[_RecoverableAttempt]:
        """Iterate over recovery attempts."""
        while self._attempts <= self.max_retries:
            self._attempts += 1
            attempt = _RecoverableAttempt(self)

            yield attempt

            # Check if attempt succeeded
            if attempt._success:
                return  # Done - exit iterator

            # Attempt failed - handle recovery
            self._handle_recovery()

        # Exhausted all retries
        raise RuntimeError(
            f"GPU error recovery failed after {self.max_retries} retries. "
            f"Original error: {self._last_error}"
        )

    def __enter__(self) -> "recoverable":
        """Enter context manager for simple usage (no retry, operation is lost on error).

        WARNING: If a GPU error occurs, the current operation is LOST.
        We migrate to a new GPU, but cannot replay what was inside the `with` block.

        For automatic retry, use the decorator or iterator pattern instead.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> bool:
        """Exit context manager - handle errors (migrate but DON'T retry).

        If a recoverable CUDA error occurred:
        1. Clear error state
        2. Migrate to new GPU
        3. Suppress the exception (operation is lost, training continues)

        Non-CUDA errors are re-raised.
        """
        import torch

        # No exception - success
        if exc_type is None:
            return False

        # Check if it's a recoverable CUDA error
        if not isinstance(exc_val, (torch.cuda.OutOfMemoryError, RuntimeError)):
            return False  # Re-raise non-CUDA errors

        # Classify the error
        error_type, error_msg = _classify_cuda_error(exc_val)

        if error_type == "UNKNOWN":
            return False  # Re-raise unknown errors

        # Store error info
        self._last_error = exc_val
        self._last_error_type = error_type
        self._last_error_msg = error_msg

        # Log the error
        logger.warning(f"GPU error detected: {error_type}")
        print(f"[flexium] GPU error: {error_type}")
        print("[flexium] WARNING: The current operation is LOST. Migrating to new GPU...")
        sys.stdout.flush()

        # Handle recovery (migrate to new GPU)
        try:
            self._handle_recovery_simple()
            print("[flexium] Migration complete. Training continues (current batch was lost).")
            sys.stdout.flush()
            return True  # Suppress exception, training continues
        except RuntimeError as e:
            # Migration failed - re-raise original error
            logger.error(f"Migration failed: {e}")
            print("[flexium] ERROR: Migration failed. Original error will be raised.")
            sys.stdout.flush()
            return False

    def _handle_recovery_simple(self) -> None:
        """Handle recovery for simple context manager (no retry count tracking)."""
        global _failed_gpus, _current_device

        error_type = self._last_error_type
        error_msg = self._last_error_msg

        # Check if migration is enabled
        if not _migration_enabled:
            raise RuntimeError(
                f"GPU error recovery requires migration support (driver 580+). "
                f"Original error: {self._last_error}"
            )

        # Mark current GPU as failed (for standalone mode tracking)
        if _current_device:
            _failed_gpus.add(_current_device)
            logger.debug(f"Marked {_current_device} as failed")

        # Clear CUDA error state
        _clear_cuda_error_state()

        # Estimate memory needed for OOM
        memory_needed = 0
        if error_type == "OOM":
            memory_needed = _estimate_memory_needed(error_msg)
            if memory_needed > 0:
                logger.info(f"OOM: estimated {memory_needed / 1e9:.2f} GB needed")

        # Request recovery target (orchestrator or local)
        target = _request_recovery_target(error_type, memory_needed)

        if target is None:
            raise RuntimeError(
                f"No suitable GPU available for recovery. "
                f"Original error: {self._last_error}"
            )

        # Migrate to new GPU
        logger.info(f"Migrating to {target} for error recovery")
        print(f"[flexium] Migrating to {target}...")
        sys.stdout.flush()

        success = _do_migration(target)

        if not success:
            raise RuntimeError(f"Migration to {target} failed")

    def _handle_recovery(self) -> None:
        """Handle recovery after a failed attempt."""
        global _failed_gpus, _current_device

        error_type = self._last_error_type
        error_msg = self._last_error_msg

        logger.warning(
            f"GPU error detected: {error_type} (attempt {self._attempts}/{self.max_retries})"
        )
        print(f"[flexium] GPU error: {error_type} (attempt {self._attempts}/{self.max_retries})")
        sys.stdout.flush()

        # Check if migration is enabled
        if not _migration_enabled:
            logger.error("Recovery requires migration, but migration is disabled")
            print("[flexium] ERROR: GPU error recovery requires migration support.")
            print("[flexium] Migration is disabled (driver 580+ required).")
            sys.stdout.flush()
            raise RuntimeError(
                f"GPU error recovery failed: migration disabled. "
                f"Original error: {self._last_error}"
            ) from self._last_error

        # Mark current GPU as failed (for standalone mode tracking)
        if _current_device:
            _failed_gpus.add(_current_device)
            logger.debug(f"Marked {_current_device} as failed")

        # Clear CUDA error state
        _clear_cuda_error_state()

        # Estimate memory needed for OOM
        memory_needed = 0
        if error_type == "OOM":
            memory_needed = _estimate_memory_needed(error_msg)
            if memory_needed > 0:
                logger.info(f"OOM: estimated {memory_needed / 1e9:.2f} GB needed")

        # Request recovery target (orchestrator or local)
        target = _request_recovery_target(error_type, memory_needed)

        if target is None:
            if self._attempts >= self.max_retries:
                logger.error(f"No recovery target available after {self._attempts} attempts")
                print("[flexium] ERROR: No GPU available for recovery.")
                sys.stdout.flush()
                raise RuntimeError(
                    f"GPU error recovery failed after {self._attempts} retries: "
                    f"no suitable GPU available. Original error: {self._last_error}"
                ) from self._last_error
            else:
                logger.info("No recovery target, will retry on same GPU...")
                print("[flexium] No recovery target available, retrying on current GPU...")
                sys.stdout.flush()
                time.sleep(1.0)
                return  # Continue to next attempt

        # Migrate to new GPU
        logger.info(f"Migrating to {target} for error recovery")
        print(f"[flexium] Recovering: migrating to {target}...")
        sys.stdout.flush()

        success = _do_migration(target)

        if not success:
            logger.error(f"Migration to {target} failed")
            print(f"[flexium] ERROR: Migration to {target} failed.")
            sys.stdout.flush()
            if self._attempts >= self.max_retries:
                raise RuntimeError(
                    f"GPU error recovery failed after {self._attempts} retries: "
                    f"migration failed. Original error: {self._last_error}"
                ) from self._last_error
            return  # Continue to next attempt

        print(f"[flexium] Recovery successful - now on {target}, retrying operation...")
        sys.stdout.flush()
