"""Configuration loading for flexium.

Handles configuration from multiple sources with priority:
1. Inline parameters (highest)
2. Environment variables
3. Config files (~/.flexiumrc, ./.flexiumrc)
4. Defaults (lowest)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from flexium.utils.logging import get_logger

logger = get_logger(__name__)

# Environment variable names
ENV_ORCHESTRATOR = "FLEXIUM_SERVER"
ENV_DEVICE = "GPU_DEVICE"

# Config file names (in priority order)
CONFIG_FILE_NAMES = [
    ".flexiumrc",  # Project-local (highest priority)
]
USER_CONFIG_PATH = Path.home() / ".flexiumrc"


@dataclass
class FlexiumConfig:
    """Configuration for flexium.

    Attributes:
        orchestrator: Orchestrator address (host:port). None means no orchestrator.
        device: Initial device to use for training.
        checkpoint_dir: Directory for storing checkpoints.
        heartbeat_interval: Interval between heartbeats in seconds.
        min_gpus: Minimum GPUs required.
        max_gpus: Maximum GPUs that can be utilized.
        max_vram: Peak VRAM requirement per GPU in bytes (0 = unlimited).
        can_share: Can run alongside other processes on same GPU.
        priority: Job priority 0-100, higher = more important.
        preemptible: Can be paused/migrated by higher priority jobs.
        migratable: Can be migrated at all.
    """

    orchestrator: Optional[str] = None
    device: str = "cuda:0"
    checkpoint_dir: str = "/tmp/flexium/checkpoints"
    heartbeat_interval: float = 3.0

    # Resource requirements
    min_gpus: int = 1
    max_gpus: int = 1
    max_vram: int = 0  # 0 = unlimited
    can_share: bool = True
    priority: int = 50
    preemptible: bool = True
    migratable: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FlexiumConfig:
        """Create config from dictionary, ignoring unknown keys.

        Parameters:
            data: Dictionary with configuration values.

        Returns:
            FlexiumConfig instance.
        """
        valid_keys = {
            "orchestrator", "device", "checkpoint_dir", "heartbeat_interval",
            "min_gpus", "max_gpus", "max_vram", "can_share",
            "priority", "preemptible", "migratable",
        }
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


def _find_config_file() -> Optional[Path]:
    """Find the first existing config file.

    Searches in order:
    1. Project-local (./.flexiumrc)
    2. User home (~/.flexiumrc)

    Returns:
        Path to config file if found, None otherwise.
    """
    # Check project-local configs
    cwd = Path.cwd()
    for name in CONFIG_FILE_NAMES:
        path = cwd / name
        if path.exists() and path.is_file():
            logger.debug(f"Found project config: {path}")
            return path

    # Check user config
    if USER_CONFIG_PATH.exists() and USER_CONFIG_PATH.is_file():
        logger.debug(f"Found user config: {USER_CONFIG_PATH}")
        return USER_CONFIG_PATH

    return None


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    """Load YAML config file.

    Parameters:
        path: Path to YAML file.

    Returns:
        Dictionary with config values.
    """
    try:
        import yaml
    except ImportError:
        logger.warning(
            "PyYAML not installed. Config file support disabled. "
            "Install with: pip install pyyaml"
        )
        return {}

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to load config file {path}: {e}")
        return {}


def load_config(
    orchestrator: Optional[str] = None,
    device: Optional[str] = None,
) -> FlexiumConfig:
    """Load configuration with priority handling.

    Priority (highest to lowest):
    1. Inline parameters passed to this function
    2. Environment variables (FLEXIUM_SERVER, GPU_DEVICE)
    3. Config file (~/.flexiumrc or ./.flexiumrc)
    4. Defaults

    Parameters:
        orchestrator: Override orchestrator address.
        device: Override device.

    Returns:
        FlexiumConfig with resolved values.
    """
    # Start with defaults
    config = FlexiumConfig()

    # Layer 1: Load from config file (lowest priority of overrides)
    config_path = _find_config_file()
    if config_path:
        file_data = _load_yaml_file(config_path)
        if file_data:
            config = FlexiumConfig.from_dict({**vars(config), **file_data})
            logger.debug(f"Loaded config from {config_path}")

    # Layer 2: Override with environment variables
    env_orchestrator = os.environ.get(ENV_ORCHESTRATOR)
    if env_orchestrator:
        config.orchestrator = env_orchestrator
        logger.debug(f"Using orchestrator from {ENV_ORCHESTRATOR}: {env_orchestrator}")

    env_device = os.environ.get(ENV_DEVICE)
    if env_device:
        config.device = env_device
        logger.debug(f"Using device from {ENV_DEVICE}: {env_device}")

    # Layer 3: Override with inline parameters (highest priority)
    if orchestrator is not None:
        config.orchestrator = orchestrator

    if device is not None:
        config.device = device

    return config


def print_no_orchestrator_warning() -> None:
    """Print a clear warning when no orchestrator is configured."""
    warning = """
============================================================
[flexium] WARNING: No orchestrator configured!
[flexium] Running in local mode (no migration support)
[flexium]
[flexium] To enable orchestrator, either:
[flexium]   - Set FLEXIUM_SERVER=host:port/workspace environment variable
[flexium]   - Create ~/.flexiumrc with: orchestrator: host:port
[flexium]   - Pass orchestrator='host:port' to run()
============================================================
"""
    print(warning)
