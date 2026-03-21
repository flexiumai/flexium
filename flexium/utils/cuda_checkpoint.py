"""CUDA checkpoint utility management.

Manages the cuda-checkpoint binary from NVIDIA's official repository.
https://github.com/NVIDIA/cuda-checkpoint

The binary is bundled with flexium for immediate availability.
On first use, checks for updates from NVIDIA GitHub and downloads if newer.

The cuda-checkpoint utility enables driver-level GPU migration by checkpointing
and restoring CUDA state within a running process. It requires NVIDIA driver 550+
(580+ for GPU migration feature).
"""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

from flexium.utils.logging import get_logger

logger = get_logger(__name__)

# NVIDIA's official cuda-checkpoint repository
CUDA_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/NVIDIA/cuda-checkpoint/main/bin/x86_64_Linux/cuda-checkpoint"
)

# User installation directory (for updates)
USER_INSTALL_DIR = Path.home() / ".flexium" / "bin"

# Minimum driver version required for basic checkpoint/restore (pause/resume)
MIN_DRIVER_VERSION = 550

# Minimum driver version required for GPU migration (restore to different GPU)
MIGRATION_DRIVER_VERSION = 580

# Cache for the resolved path
_resolved_path: Optional[Path] = None


class CudaCheckpointError(Exception):
    """Error related to cuda-checkpoint utility."""

    pass


def get_bundled_path() -> Path:
    """Get path to the bundled cuda-checkpoint binary.

    Returns:
        Path to bundled binary in the package.
    """
    # Use importlib.resources for Python 3.9+, fallback for 3.8
    try:
        import importlib.resources as resources
        try:
            # Python 3.9+
            return Path(resources.files("flexium") / "bin" / "cuda-checkpoint")
        except AttributeError:
            # Python 3.8
            with resources.path("flexium.bin", "cuda-checkpoint") as p:
                return p
    except Exception:
        # Fallback: relative to this file
        return Path(__file__).parent.parent / "bin" / "cuda-checkpoint"


def get_driver_version() -> Optional[int]:
    """Get the major NVIDIA driver version.

    Returns:
        Major driver version (e.g., 590) or None if not available.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
        pynvml.nvmlShutdown()
        # Version is like "590.48.01", extract major
        major = int(version.split(".")[0])
        return major
    except Exception:
        pass

    # Fallback to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            version = result.stdout.strip().split("\n")[0]
            major = int(version.split(".")[0])
            return major
    except Exception:
        pass

    return None


def get_file_hash(path: Path) -> Optional[str]:
    """Get MD5 hash of a file.

    Args:
        path: Path to file.

    Returns:
        MD5 hash string or None if file doesn't exist.
    """
    if not path.exists():
        return None
    try:
        return hashlib.md5(path.read_bytes()).hexdigest()
    except Exception:
        return None


def check_for_update() -> Optional[str]:
    """Check if a newer version is available on GitHub.

    Compares hash of bundled/installed version with remote.

    Returns:
        URL to download if update available, None otherwise.
    """
    try:
        # Get current binary path and hash
        current_path = find_cuda_checkpoint()
        if current_path is None:
            return CUDA_CHECKPOINT_URL

        current_hash = get_file_hash(current_path)
        if current_hash is None:
            return CUDA_CHECKPOINT_URL

        # Fetch remote binary and compare hash
        # Use HEAD request or download small chunk to check
        # For simplicity, download full file (only 6KB)
        request = urllib.request.Request(CUDA_CHECKPOINT_URL)
        with urllib.request.urlopen(request, timeout=5) as response:
            remote_data = response.read()
            remote_hash = hashlib.md5(remote_data).hexdigest()

        if remote_hash != current_hash:
            logger.info(f"cuda-checkpoint update available (current: {current_hash[:8]}, remote: {remote_hash[:8]})")
            return CUDA_CHECKPOINT_URL

        return None

    except Exception as e:
        logger.debug(f"Could not check for cuda-checkpoint updates: {e}")
        return None


def find_cuda_checkpoint() -> Optional[Path]:
    """Find cuda-checkpoint binary.

    Searches in order:
    1. ~/.flexium/bin/cuda-checkpoint (user-installed/updated)
    2. Bundled in package
    3. System PATH
    4. Common system locations

    Returns:
        Path to cuda-checkpoint binary or None if not found.
    """
    # Check user installation directory first (for updates)
    user_path = USER_INSTALL_DIR / "cuda-checkpoint"
    if user_path.exists() and os.access(user_path, os.X_OK):
        return user_path

    # Check bundled version
    bundled_path = get_bundled_path()
    if bundled_path.exists() and os.access(bundled_path, os.X_OK):
        return bundled_path

    # Check PATH
    try:
        result = subprocess.run(
            ["which", "cuda-checkpoint"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            path = Path(result.stdout.strip())
            if path.exists():
                return path
    except Exception:
        pass

    # Check common locations
    common_paths = [
        Path("/usr/local/bin/cuda-checkpoint"),
        Path("/usr/bin/cuda-checkpoint"),
        Path.home() / "bin" / "cuda-checkpoint",
    ]
    for path in common_paths:
        if path.exists() and os.access(path, os.X_OK):
            return path

    return None


def download_cuda_checkpoint(install_dir: Optional[Path] = None) -> Path:
    """Download cuda-checkpoint from NVIDIA's GitHub repository.

    Args:
        install_dir: Directory to install to. Defaults to ~/.flexium/bin/

    Returns:
        Path to the downloaded binary.

    Raises:
        CudaCheckpointError: If download fails.
    """
    install_dir = install_dir or USER_INSTALL_DIR
    install_dir.mkdir(parents=True, exist_ok=True)

    target_path = install_dir / "cuda-checkpoint"

    logger.info("Downloading cuda-checkpoint update from NVIDIA GitHub...")
    logger.debug(f"URL: {CUDA_CHECKPOINT_URL}")
    logger.debug(f"Target: {target_path}")

    try:
        # Download the binary
        urllib.request.urlretrieve(CUDA_CHECKPOINT_URL, target_path)

        # Make executable
        target_path.chmod(target_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        logger.info(f"Successfully updated cuda-checkpoint at {target_path}")
        return target_path

    except urllib.error.URLError as e:
        raise CudaCheckpointError(
            f"Failed to download cuda-checkpoint: {e}\n"
            f"URL: {CUDA_CHECKPOINT_URL}\n"
            f"Please check your internet connection or download manually from:\n"
            f"https://github.com/NVIDIA/cuda-checkpoint"
        ) from e
    except Exception as e:
        raise CudaCheckpointError(f"Failed to install cuda-checkpoint: {e}") from e


def ensure_cuda_checkpoint(check_updates: bool = True) -> Path:
    """Ensure cuda-checkpoint is available, checking for updates if requested.

    Args:
        check_updates: Whether to check for updates from GitHub.

    Returns:
        Path to cuda-checkpoint binary.

    Raises:
        CudaCheckpointError: If cuda-checkpoint cannot be found,
            or if driver version is insufficient.
    """
    global _resolved_path

    # Return cached path if already resolved
    if _resolved_path is not None:
        return _resolved_path

    # Check driver version first
    driver_version = get_driver_version()
    if driver_version is None:
        raise CudaCheckpointError(
            "Could not detect NVIDIA driver version.\n"
            "Please ensure NVIDIA drivers are installed."
        )

    if driver_version < MIN_DRIVER_VERSION:
        raise CudaCheckpointError(
            f"NVIDIA driver {driver_version} is too old.\n"
            f"Minimum required: {MIN_DRIVER_VERSION}+ (pause/resume)\n"
            f"For GPU migration: {MIGRATION_DRIVER_VERSION}+\n"
            f"Please update your NVIDIA driver."
        )

    # Find existing installation (bundled or user-installed)
    existing = find_cuda_checkpoint()

    if existing is None:
        # This shouldn't happen if package is installed correctly
        # Try downloading as last resort
        logger.warning("Bundled cuda-checkpoint not found, downloading...")
        try:
            existing = download_cuda_checkpoint()
        except CudaCheckpointError:
            raise CudaCheckpointError(
                "cuda-checkpoint binary not found and download failed.\n"
                "Please reinstall flexium or download manually from:\n"
                "https://github.com/NVIDIA/cuda-checkpoint"
            )

    # Check for updates (non-blocking, best-effort)
    if check_updates:
        try:
            update_url = check_for_update()
            if update_url:
                try:
                    existing = download_cuda_checkpoint()
                except Exception as e:
                    # Update failed, but we have bundled version
                    logger.debug(f"Could not update cuda-checkpoint: {e}")
        except Exception as e:
            # Update check failed, use existing
            logger.debug(f"Could not check for updates: {e}")

    _resolved_path = existing
    logger.debug(f"Using cuda-checkpoint at {_resolved_path}")
    return _resolved_path


def get_cuda_checkpoint_path() -> Path:
    """Get path to cuda-checkpoint, ensuring it exists.

    This is the main entry point for other modules.

    Returns:
        Path to cuda-checkpoint binary.

    Raises:
        CudaCheckpointError: If cuda-checkpoint is not available.
    """
    return ensure_cuda_checkpoint()


def supports_migration() -> bool:
    """Check if the current driver supports GPU migration.

    GPU migration (restoring to a different GPU) requires driver 580+.
    Drivers 550-579 only support pause/resume on the same GPU.

    Returns:
        True if driver supports migration, False otherwise.
    """
    driver_version = get_driver_version()
    if driver_version is None:
        return False
    return driver_version >= MIGRATION_DRIVER_VERSION


def get_capabilities() -> dict:
    """Get cuda-checkpoint capabilities based on driver version.

    Returns:
        Dictionary with capability flags:
        - driver_version: int or None
        - pause_resume: bool (True if 550+)
        - migration: bool (True if 580+)
    """
    driver_version = get_driver_version()
    return {
        "driver_version": driver_version,
        "pause_resume": driver_version is not None and driver_version >= MIN_DRIVER_VERSION,
        "migration": driver_version is not None and driver_version >= MIGRATION_DRIVER_VERSION,
    }


def verify_cuda_checkpoint(path: Optional[Path] = None) -> bool:
    """Verify cuda-checkpoint binary works correctly.

    Args:
        path: Path to binary. If None, uses find_cuda_checkpoint().

    Returns:
        True if cuda-checkpoint is working.
    """
    if path is None:
        path = find_cuda_checkpoint()
        if path is None:
            return False

    try:
        result = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "CUDA checkpoint" in result.stdout
    except Exception:
        return False


def get_cuda_checkpoint_version(path: Optional[Path] = None) -> Optional[str]:
    """Get cuda-checkpoint version string.

    Args:
        path: Path to binary. If None, uses find_cuda_checkpoint().

    Returns:
        Version string (e.g., "590.48.01") or None.
    """
    if path is None:
        path = find_cuda_checkpoint()
        if path is None:
            return None

    try:
        result = subprocess.run(
            [str(path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # Parse "Version 590.48.01." from output
            for line in result.stdout.split("\n"):
                if "Version" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "Version" and i + 1 < len(parts):
                            version = parts[i + 1].rstrip(".")
                            return version
    except Exception:
        pass

    return None
