"""GPU information utilities for flexium.

Provides functions to retrieve GPU UUIDs, names, and other hardware
information. Handles CUDA_VISIBLE_DEVICES mapping correctly.

Supported GPU types:
- Physical GPUs: Standard NVIDIA GPUs (e.g., GPU-a1b2c3d4-...)
- MIG instances: Multi-Instance GPU slices (e.g., MIG-GPU-xxx/GI/CI)
- vGPU: Virtual GPUs in virtualized environments

UUID Formats:
- Physical GPU: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- MIG instance: MIG-GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/GI/CI
  or: MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- vGPU: Same as physical GPU format

CUDA_VISIBLE_DEVICES formats:
- Index: "0,1,2" (physical GPU indices)
- UUID: "GPU-xxx,GPU-yyy" (full or partial UUIDs)
- MIG: "MIG-GPU-xxx/0/0" (MIG instance UUID)
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional

from flexium.utils.logging import get_logger

logger = get_logger(__name__)


class GPUType(Enum):
    """Type of GPU device."""

    PHYSICAL = auto()      # Standard physical GPU
    MIG = auto()           # MIG (Multi-Instance GPU) slice
    VGPU = auto()          # Virtual GPU
    UNKNOWN = auto()       # Unknown type


@dataclass
class GPUInfo:
    """Information about a GPU device.

    Attributes:
        logical_index: The cuda:X index visible to PyTorch.
        physical_index: The physical GPU index on the system.
        uuid: NVIDIA GPU UUID (e.g., GPU-a1b2c3d4-...).
        name: GPU model name (e.g., Tesla V100-SXM2-32GB).
        memory_total: Total memory in bytes.
        gpu_type: Type of GPU (physical, MIG, vGPU).
        mig_gi: MIG GPU Instance ID (if MIG).
        mig_ci: MIG Compute Instance ID (if MIG).
        parent_gpu_uuid: Parent GPU UUID for MIG instances.
    """

    logical_index: int
    physical_index: int
    uuid: str
    name: str
    memory_total: int = 0
    gpu_type: GPUType = GPUType.PHYSICAL
    mig_gi: Optional[int] = None  # GPU Instance ID for MIG
    mig_ci: Optional[int] = None  # Compute Instance ID for MIG
    parent_gpu_uuid: Optional[str] = None  # Parent GPU for MIG

    @property
    def short_uuid(self) -> str:
        """Get shortened UUID (first 8 characters after prefix)."""
        uuid = self.uuid
        # Handle MIG UUID format: MIG-GPU-xxx or MIG-xxx
        if uuid.startswith("MIG-GPU-"):
            return uuid[8:16]
        elif uuid.startswith("MIG-"):
            return uuid[4:12]
        elif uuid.startswith("GPU-"):
            return uuid[4:12]
        return uuid[:8]

    @property
    def device_string(self) -> str:
        """Get the cuda:X device string."""
        return f"cuda:{self.logical_index}"

    @property
    def is_mig(self) -> bool:
        """Check if this is a MIG instance."""
        return self.gpu_type == GPUType.MIG

    @property
    def display_name(self) -> str:
        """Get a display-friendly name."""
        if self.is_mig and self.mig_gi is not None:
            return f"{self.name} (MIG {self.mig_gi}/{self.mig_ci})"
        return self.name

    def __str__(self) -> str:
        return f"cuda:{self.logical_index} ({self.display_name}, {self.short_uuid})"


def _detect_gpu_type(uuid: str) -> GPUType:
    """Detect GPU type from UUID format.

    Parameters:
        uuid: The GPU UUID string.

    Returns:
        GPUType enum value.
    """
    if uuid.startswith("MIG-"):
        return GPUType.MIG
    elif uuid.startswith("GPU-"):
        return GPUType.PHYSICAL
    return GPUType.UNKNOWN


def _parse_mig_uuid(uuid: str) -> tuple:
    """Parse MIG UUID to extract components.

    MIG UUID formats:
    - MIG-GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/GI/CI
    - MIG-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

    Parameters:
        uuid: The MIG UUID string.

    Returns:
        Tuple of (parent_gpu_uuid, gi, ci) or (None, None, None) if not MIG.
    """
    if not uuid.startswith("MIG-"):
        return None, None, None

    # Check for /GI/CI suffix
    if "/" in uuid:
        parts = uuid.rsplit("/", 2)
        if len(parts) == 3:
            base_uuid = parts[0]
            try:
                gi = int(parts[1])
                ci = int(parts[2])
                # Extract parent GPU UUID
                if base_uuid.startswith("MIG-GPU-"):
                    parent = "GPU-" + base_uuid[8:]
                else:
                    parent = None
                return parent, gi, ci
            except ValueError:
                pass

    return None, None, None


def _get_visible_device_indices() -> List[int]:
    """Get the physical GPU indices visible to CUDA.

    Parses CUDA_VISIBLE_DEVICES environment variable to determine
    which physical GPUs are visible and in what order.

    Returns:
        List of physical GPU indices in logical order.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if not cvd:
        # No restriction - all GPUs visible
        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            pynvml.nvmlShutdown()
            return list(range(count))
        except Exception:
            # Fallback: try nvidia-smi
            return _get_gpu_count_via_smi()

    # Parse CUDA_VISIBLE_DEVICES
    # Can be comma-separated indices, UUIDs, or MIG UUIDs
    indices = []
    for item in cvd.split(","):
        item = item.strip()
        if not item:
            continue

        if item.startswith("MIG-"):
            # MIG UUID format: MIG-GPU-xxx/GI/CI or MIG-xxx
            # Extract the parent GPU UUID and find its index
            parent_uuid, _, _ = _parse_mig_uuid(item)
            if parent_uuid:
                physical_idx = _uuid_to_physical_index(parent_uuid)
            else:
                # Try to find by the MIG UUID directly
                physical_idx = _mig_uuid_to_physical_index(item)
            if physical_idx is not None:
                indices.append(physical_idx)
            else:
                logger.warning(f"Could not resolve MIG UUID: {item}")
        elif item.startswith("GPU-"):
            # It's a UUID - find the physical index
            physical_idx = _uuid_to_physical_index(item)
            if physical_idx is not None:
                indices.append(physical_idx)
        else:
            # It's an index
            try:
                indices.append(int(item))
            except ValueError:
                logger.warning(f"Invalid CUDA_VISIBLE_DEVICES entry: {item}")

    return indices


def _get_gpu_count_via_smi() -> List[int]:
    """Get GPU indices via nvidia-smi fallback."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            indices = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    indices.append(int(line.strip()))
            return indices
    except Exception as e:
        logger.debug(f"nvidia-smi fallback failed: {e}")

    return []


def _uuid_to_physical_index(uuid: str) -> Optional[int]:
    """Convert a GPU UUID to its physical index."""
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
            if gpu_uuid == uuid:
                pynvml.nvmlShutdown()
                return i
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return None


def _mig_uuid_to_physical_index(mig_uuid: str) -> Optional[int]:
    """Convert a MIG UUID to the physical index of its parent GPU.

    This handles cases where CUDA_VISIBLE_DEVICES contains a MIG UUID
    without the parent GPU information embedded.

    Parameters:
        mig_uuid: The MIG UUID string.

    Returns:
        Physical GPU index of the parent GPU, or None if not found.
    """
    try:
        import pynvml
        pynvml.nvmlInit()

        # Iterate through all GPUs to find MIG instances
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            # Check if this GPU has MIG mode enabled
            try:
                mig_mode, _ = pynvml.nvmlDeviceGetMigMode(handle)
                if mig_mode != pynvml.NVML_DEVICE_MIG_ENABLE:
                    continue

                # Get MIG device count
                mig_count = pynvml.nvmlDeviceGetMaxMigDeviceCount(handle)
                for j in range(mig_count):
                    try:
                        mig_handle = pynvml.nvmlDeviceGetMigDeviceHandleByIndex(
                            handle, j
                        )
                        mig_device_uuid = pynvml.nvmlDeviceGetUUID(mig_handle)
                        if mig_device_uuid == mig_uuid:
                            pynvml.nvmlShutdown()
                            return i
                    except pynvml.NVMLError:
                        continue
            except (AttributeError, pynvml.NVMLError):
                # MIG not supported on this GPU
                continue

        pynvml.nvmlShutdown()
    except Exception:
        pass
    return None


def get_gpu_info_pynvml(physical_index: int) -> Optional[Dict]:
    """Get GPU info using pynvml.

    Also detects MIG instances if available.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)

        uuid = pynvml.nvmlDeviceGetUUID(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        memory_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total

        info = {
            "uuid": uuid,
            "name": name,
            "memory_total": memory_total,
            "gpu_type": _detect_gpu_type(uuid),
        }

        # Check for MIG mode
        try:
            mig_mode, pending = pynvml.nvmlDeviceGetMigMode(handle)
            info["mig_mode_enabled"] = (mig_mode == pynvml.NVML_DEVICE_MIG_ENABLE)
        except (AttributeError, pynvml.NVMLError):
            info["mig_mode_enabled"] = False

        # If MIG UUID, parse it
        if uuid.startswith("MIG-"):
            parent, gi, ci = _parse_mig_uuid(uuid)
            info["parent_gpu_uuid"] = parent
            info["mig_gi"] = gi
            info["mig_ci"] = ci
            info["gpu_type"] = GPUType.MIG

        pynvml.nvmlShutdown()
        return info
    except Exception as e:
        logger.debug(f"pynvml failed for GPU {physical_index}: {e}")
        return None


def get_gpu_info_smi(physical_index: int) -> Optional[Dict]:
    """Get GPU info using nvidia-smi fallback."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={physical_index}",
                "--query-gpu=uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 3:
                return {
                    "uuid": parts[0].strip(),
                    "name": parts[1].strip(),
                    "memory_total": int(parts[2].strip()) * 1024 * 1024,  # MiB to bytes
                }
    except Exception as e:
        logger.debug(f"nvidia-smi failed for GPU {physical_index}: {e}")

    return None


def get_all_gpu_info() -> List[GPUInfo]:
    """Get information about all visible GPUs.

    Returns GPUInfo for each GPU visible to CUDA, correctly mapping
    logical indices (cuda:0, cuda:1, ...) to physical GPU UUIDs.

    Returns:
        List of GPUInfo objects, one per visible GPU.
    """
    visible_indices = _get_visible_device_indices()
    gpus = []

    for logical_idx, physical_idx in enumerate(visible_indices):
        # Try pynvml first, fall back to nvidia-smi
        info = get_gpu_info_pynvml(physical_idx)
        if info is None:
            info = get_gpu_info_smi(physical_idx)

        if info is not None:
            gpus.append(GPUInfo(
                logical_index=logical_idx,
                physical_index=physical_idx,
                uuid=info["uuid"],
                name=info["name"],
                memory_total=info.get("memory_total", 0),
                gpu_type=info.get("gpu_type", GPUType.PHYSICAL),
                mig_gi=info.get("mig_gi"),
                mig_ci=info.get("mig_ci"),
                parent_gpu_uuid=info.get("parent_gpu_uuid"),
            ))
        else:
            # Minimal fallback
            gpus.append(GPUInfo(
                logical_index=logical_idx,
                physical_index=physical_idx,
                uuid=f"unknown-{physical_idx}",
                name="Unknown GPU",
                memory_total=0,
            ))

    return gpus


def get_gpu_info(device: str) -> Optional[GPUInfo]:
    """Get information about a specific GPU device.

    Parameters:
        device: Device string like "cuda:0", "cuda:1", a GPU UUID, or MIG UUID.

    Returns:
        GPUInfo for the device, or None if not found.
    """
    all_gpus = get_all_gpu_info()

    # Check if it's a cuda:X format
    if device.startswith("cuda:"):
        try:
            idx = int(device.split(":")[1])
            for gpu in all_gpus:
                if gpu.logical_index == idx:
                    return gpu
        except (ValueError, IndexError):
            pass

    # Check if it's a MIG UUID - exact match first
    if device.startswith("MIG-"):
        for gpu in all_gpus:
            if gpu.uuid == device:
                return gpu
        # Short UUID fallback (only if exact match not found)
        short_uuid = device[8:16] if device.startswith("MIG-GPU-") else device[4:12]
        for gpu in all_gpus:
            if gpu.short_uuid == short_uuid:
                return gpu

    # Check if it's a GPU UUID - exact match first
    if device.startswith("GPU-"):
        for gpu in all_gpus:
            if gpu.uuid == device:
                return gpu
        # Short UUID fallback (only if exact match not found)
        for gpu in all_gpus:
            if gpu.short_uuid == device[4:12]:
                return gpu

    return None


def _get_all_gpu_pids(physical_index: int) -> set:
    """Get all PIDs currently using a GPU.

    Parameters:
        physical_index: Physical GPU index.

    Returns:
        Set of PIDs using the GPU.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)

        pids = set()
        # Get both compute and graphics processes
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pids.add(proc.pid)
        try:
            for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                pids.add(proc.pid)
        except Exception:
            pass  # Graphics processes may not be available

        pynvml.nvmlShutdown()
        return pids
    except Exception:
        return set()


def _get_gpu_memory_for_pid(physical_index: int, pid: int) -> int:
    """Get GPU memory used by a specific PID.

    Parameters:
        physical_index: Physical GPU index.
        pid: Process ID to look up.

    Returns:
        GPU memory in bytes, or 0 if not found.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)

        # Check compute processes
        for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if proc.pid == pid:
                memory = proc.usedGpuMemory
                pynvml.nvmlShutdown()
                return memory

        # Check graphics processes
        try:
            for proc in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                if proc.pid == pid:
                    memory = proc.usedGpuMemory
                    pynvml.nvmlShutdown()
                    return memory
        except Exception:
            pass

        pynvml.nvmlShutdown()
    except Exception:
        pass

    return 0


# Cache for discovered GPU PID (the PID that pynvml sees for our process)
_discovered_gpu_pid: Optional[int] = None
# Cache for PIDs before CUDA initialization (captured early)
_pids_before_cuda: Optional[set] = None


def reset_gpu_pid_cache() -> None:
    """Reset the GPU PID discovery caches.

    Call this after migration to a new GPU to ensure fresh PID discovery.
    """
    global _discovered_gpu_pid, _pids_before_cuda
    _discovered_gpu_pid = None
    _pids_before_cuda = None
    logger.debug("GPU PID caches reset")


def capture_pids_before_cuda(device: str = "cuda:0") -> None:
    """Capture GPU PIDs before CUDA is initialized.

    Call this function early in process startup, before any CUDA operations,
    to enable accurate GPU PID discovery in containerized environments.

    Parameters:
        device: Device string like "cuda:0".
    """
    global _pids_before_cuda

    if _pids_before_cuda is not None:
        return  # Already captured

    try:
        import torch
        if torch.cuda.is_initialized():
            # Too late - CUDA already initialized
            logger.debug("capture_pids_before_cuda called after CUDA init")
            return

        gpu_info = get_gpu_info(device)
        if gpu_info is None:
            return

        _pids_before_cuda = _get_all_gpu_pids(gpu_info.physical_index)
        logger.debug(f"Captured {len(_pids_before_cuda)} PIDs before CUDA init")
    except Exception as e:
        logger.debug(f"Failed to capture PIDs before CUDA: {e}")


def discover_gpu_pid(device: str = "cuda:0") -> Optional[int]:
    """Discover the PID that pynvml uses for our CUDA context.

    In containerized environments, the PID that pynvml/nvidia-smi sees
    may differ from os.getpid(). This function discovers the correct PID
    by comparing GPU process lists before and after CUDA initialization.

    Parameters:
        device: Device string like "cuda:0".

    Returns:
        The PID that pynvml sees for our process, or None if not found.
    """
    global _discovered_gpu_pid, _pids_before_cuda

    # Return cached PID if already discovered
    if _discovered_gpu_pid is not None:
        return _discovered_gpu_pid

    gpu_info = get_gpu_info(device)
    if gpu_info is None:
        return None

    physical_index = gpu_info.physical_index

    try:
        import torch

        # Check if CUDA is already initialized
        cuda_already_init = torch.cuda.is_initialized()

        # Use pre-captured PIDs if available, otherwise capture now
        if _pids_before_cuda is not None:
            pids_before = _pids_before_cuda
        else:
            pids_before = _get_all_gpu_pids(physical_index)

        if not cuda_already_init:
            # CUDA not yet initialized - we can detect the new PID
            # Parse device index
            if device.startswith("cuda:"):
                dev_idx = int(device.split(":")[1])
            else:
                dev_idx = 0
            _ = torch.tensor([1], device=f"cuda:{dev_idx}")

            # Get PIDs after CUDA initialization
            pids_after = _get_all_gpu_pids(physical_index)

            # Find new PIDs (should be our process)
            new_pids = pids_after - pids_before

            if new_pids:
                _discovered_gpu_pid = new_pids.pop()
                logger.debug(f"Discovered GPU PID: {_discovered_gpu_pid} (os.getpid={os.getpid()})")
                return _discovered_gpu_pid
        else:
            # CUDA was already initialized
            # Get current PIDs on the GPU
            pids_current = _get_all_gpu_pids(physical_index)

            # If we have pre-captured PIDs, use them to find the new PID
            if _pids_before_cuda is not None:
                new_pids = pids_current - _pids_before_cuda
                if new_pids:
                    _discovered_gpu_pid = new_pids.pop()
                    logger.debug(f"Discovered GPU PID from pre-captured: {_discovered_gpu_pid}")
                    return _discovered_gpu_pid

            # Check if os.getpid() is in the list
            if os.getpid() in pids_current:
                _discovered_gpu_pid = os.getpid()
                logger.debug(f"Using os.getpid() as GPU PID: {_discovered_gpu_pid}")
                return _discovered_gpu_pid

            # In containerized environments, os.getpid() won't match
            # We need to try all current PIDs and see which one has memory usage
            # that roughly matches PyTorch's view
            pytorch_mem = torch.cuda.memory_reserved()
            if pytorch_mem > 0:
                # Find a PID with similar memory usage
                best_pid = None
                best_diff = float('inf')
                for pid in pids_current:
                    mem = _get_gpu_memory_for_pid(physical_index, pid)
                    # pynvml memory should be >= PyTorch memory_reserved
                    if mem >= pytorch_mem:
                        diff = mem - pytorch_mem
                        if diff < best_diff:
                            best_diff = diff
                            best_pid = pid
                if best_pid is not None:
                    _discovered_gpu_pid = best_pid
                    logger.debug(f"Discovered GPU PID by memory matching: {_discovered_gpu_pid}")
                    return _discovered_gpu_pid

    except Exception as e:
        logger.debug(f"Failed to discover GPU PID: {e}")

    return None


def get_process_gpu_memory(device: str = "cuda:0") -> int:
    """Get the GPU memory used by the current process.

    Uses pynvml to query the actual GPU memory usage as reported by the
    NVIDIA driver, which matches what nvidia-smi shows. This handles
    containerized environments where the visible PID may differ from os.getpid().

    Parameters:
        device: Device string like "cuda:0".

    Returns:
        GPU memory used by current process in bytes, or 0 if unavailable.
    """
    gpu_info = get_gpu_info(device)
    if gpu_info is None:
        return 0

    physical_index = gpu_info.physical_index

    # First try with os.getpid()
    memory = _get_gpu_memory_for_pid(physical_index, os.getpid())
    if memory > 0:
        return memory

    # Try with discovered GPU PID (for containerized environments)
    gpu_pid = discover_gpu_pid(device)
    if gpu_pid is not None:
        memory = _get_gpu_memory_for_pid(physical_index, gpu_pid)
        if memory > 0:
            return memory

    return 0


def get_gpu_memory_by_physical_index(physical_index: int, gpu_pid: Optional[int] = None) -> int:
    """Get GPU memory used by current process on a specific physical GPU.

    This is useful after driver migration migration where logical device indices
    may be remapped at the process level but we know the physical GPU index.

    Parameters:
        physical_index: Physical GPU index (0, 1, 2, ...).
        gpu_pid: Optional GPU PID (host PID as seen by nvidia-smi). If provided,
                 this is used instead of os.getpid() for containerized environments.

    Returns:
        GPU memory in bytes, or 0 if unavailable.
    """
    import os

    # Try with os.getpid() first
    memory = _get_gpu_memory_for_pid(physical_index, os.getpid())
    if memory > 0:
        return memory

    # Try with provided gpu_pid (for containerized environments)
    if gpu_pid is not None and gpu_pid > 0:
        memory = _get_gpu_memory_for_pid(physical_index, gpu_pid)
        if memory > 0:
            return memory

    return 0


def get_estimated_gpu_memory(device: str = "cuda:0") -> int:
    """Get GPU memory usage matching nvidia-smi.

    Uses pynvml to query actual per-process GPU memory. Handles containerized
    environments where the visible PID may differ from os.getpid() by
    discovering the correct PID through before/after comparison.

    Parameters:
        device: Device string like "cuda:0".

    Returns:
        GPU memory in bytes, or 0 if unavailable.
    """
    memory = get_process_gpu_memory(device)
    if memory > 0:
        return memory

    # Last resort fallback - shouldn't normally reach here
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_reserved()
    except Exception:
        pass

    return 0


def get_all_device_reports(hostname: str) -> List[Dict[str, Any]]:
    """Get status of all devices (CPU + GPUs) on this host for reporting to orchestrator.

    Collects detailed information about ALL devices on this host,
    including memory usage, utilization, temperature, and power for GPUs.
    This ignores CUDA_VISIBLE_DEVICES to report the full system state.

    Parameters:
        hostname: Hostname to include in each report.

    Returns:
        List of device reports with UUID, memory, utilization, etc.
    """
    reports = []

    # Always add CPU as a device
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=None)
        reports.append({
            "gpu_uuid": "CPU",
            "gpu_name": "CPU",
            "hostname": hostname,
            "memory_total": mem.total,
            "memory_used": mem.used,
            "memory_free": mem.available,
            "gpu_utilization": int(cpu_percent),
            "temperature": 0,  # Could use psutil.sensors_temperatures() if available
            "power_usage": 0,
            "process_count": 0,  # Not tracking CPU process count
        })
    except ImportError:
        # psutil not available, add basic CPU entry
        reports.append({
            "gpu_uuid": "CPU",
            "gpu_name": "CPU",
            "hostname": hostname,
            "memory_total": 0,
            "memory_used": 0,
            "memory_free": 0,
            "gpu_utilization": 0,
            "temperature": 0,
            "power_usage": 0,
            "process_count": 0,
        })
    except Exception as e:
        logger.debug(f"Failed to get CPU info: {e}")
        # Still add a basic CPU entry
        reports.append({
            "gpu_uuid": "CPU",
            "gpu_name": "CPU",
            "hostname": hostname,
            "memory_total": 0,
            "memory_used": 0,
            "memory_free": 0,
            "gpu_utilization": 0,
            "temperature": 0,
            "power_usage": 0,
            "process_count": 0,
        })

    # Add GPU devices
    try:
        import pynvml
        pynvml.nvmlInit()

        # Get ALL GPUs on the system (ignores CUDA_VISIBLE_DEVICES)
        # pynvml always sees all GPUs regardless of CUDA_VISIBLE_DEVICES
        device_count = pynvml.nvmlDeviceGetCount()

        for physical_idx in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(physical_idx)

                # Basic info
                gpu_uuid = pynvml.nvmlDeviceGetUUID(handle)
                gpu_name = pynvml.nvmlDeviceGetName(handle)

                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total = mem_info.total
                memory_used = mem_info.used
                memory_free = mem_info.free

                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = util.gpu
                except pynvml.NVMLError:
                    gpu_utilization = 0

                # Temperature
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except pynvml.NVMLError:
                    temperature = 0

                # Power usage
                try:
                    power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # mW to W
                except pynvml.NVMLError:
                    power_usage = 0

                # Process count on this GPU
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    process_count = len(procs)
                except pynvml.NVMLError:
                    process_count = 0

                reports.append({
                    "gpu_uuid": gpu_uuid,
                    "gpu_name": gpu_name,
                    "hostname": hostname,
                    "memory_total": memory_total,
                    "memory_used": memory_used,
                    "memory_free": memory_free,
                    "gpu_utilization": gpu_utilization,
                    "temperature": temperature,
                    "power_usage": power_usage,
                    "process_count": process_count,
                })

            except pynvml.NVMLError as e:
                logger.debug(f"Failed to get info for GPU {physical_idx}: {e}")
                continue

        pynvml.nvmlShutdown()

    except Exception as e:
        logger.warning(f"Failed to collect device reports: {e}")

    return reports
