#!/usr/bin/env python3
"""Flexium setup CLI - install dependencies and verify environment.

Usage:
    flexium-setup              # Install cuda-checkpoint and verify environment
    flexium-setup --check      # Only check, don't install
    flexium-setup --force      # Force re-download even if exists
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    """Main entry point for flexium-setup."""
    parser = argparse.ArgumentParser(
        description="Install and verify Flexium dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    flexium-setup              Install cuda-checkpoint and verify environment
    flexium-setup --check      Check environment without installing
    flexium-setup --force      Force re-download of cuda-checkpoint
        """,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check environment, don't install anything",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cuda-checkpoint exists",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output except errors",
    )

    args = parser.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    log("=" * 60)
    log("  Flexium Setup")
    log("=" * 60)
    log("")

    # Check NVIDIA driver
    log("[1/3] Checking NVIDIA driver...")
    from flexium.utils.cuda_checkpoint import (
        get_driver_version,
        MIN_DRIVER_VERSION,
        MIGRATION_DRIVER_VERSION,
        supports_migration,
    )

    driver_version = get_driver_version()
    if driver_version is None:
        print("  ✗ NVIDIA driver not found")
        print(f"    Please install NVIDIA driver {MIN_DRIVER_VERSION}+")
        return 1

    if driver_version < MIN_DRIVER_VERSION:
        print(f"  ✗ Driver {driver_version} is too old (need {MIN_DRIVER_VERSION}+)")
        print("    Please update your NVIDIA driver")
        return 1

    log(f"  ✓ Driver version: {driver_version}")
    if supports_migration():
        log(f"  ✓ GPU migration supported (driver {MIGRATION_DRIVER_VERSION}+)")
    else:
        log(f"  ⚠ GPU migration requires driver {MIGRATION_DRIVER_VERSION}+")
        log(f"    Pause/resume on same GPU is available")

    # Check/install cuda-checkpoint
    log("")
    log("[2/3] Checking cuda-checkpoint...")
    from flexium.utils.cuda_checkpoint import (
        find_cuda_checkpoint,
        download_cuda_checkpoint,
        verify_cuda_checkpoint,
        get_cuda_checkpoint_version,
        CudaCheckpointError,
    )

    existing = find_cuda_checkpoint()

    if existing and not args.force:
        version = get_cuda_checkpoint_version(existing)
        if verify_cuda_checkpoint(existing):
            log(f"  ✓ Found at: {existing}")
            log(f"  ✓ Version: {version or 'unknown'}")
        else:
            print(f"  ✗ Found at {existing} but verification failed")
            if not args.check:
                log("    Re-downloading...")
                existing = None
    else:
        if args.force and existing:
            log(f"  → Force re-download requested")
        else:
            log("  → Not found")

    if not existing or args.force:
        if args.check:
            print("  ✗ cuda-checkpoint not installed")
            print("    Run 'flexium-setup' to install")
            return 1

        log("  → Downloading from NVIDIA GitHub...")
        try:
            path = download_cuda_checkpoint()
            version = get_cuda_checkpoint_version(path)
            log(f"  ✓ Installed to: {path}")
            log(f"  ✓ Version: {version or 'unknown'}")
        except CudaCheckpointError as e:
            print(f"  ✗ Failed to install: {e}")
            return 1

    # Verify environment
    log("")
    log("[3/3] Verifying environment...")

    # Check pynvml
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        log(f"  ✓ pynvml: {device_count} GPU(s) detected")
    except Exception as e:
        print(f"  ✗ pynvml error: {e}")

    # Check PyTorch (optional)
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            log(f"  ✓ PyTorch: CUDA {cuda_version}, {device_count} GPU(s)")
        else:
            log("  ⚠ PyTorch: CUDA not available (CPU only)")
    except ImportError:
        log("  ⚠ PyTorch: not installed (optional)")

    log("")
    log("=" * 60)
    log("  ✓ Flexium setup complete!")
    log("=" * 60)
    log("")
    log("Next steps:")
    log("  1. Set FLEXIUM_SERVER environment variable:")
    log('     export FLEXIUM_SERVER="app.flexium.ai/<workspace>"')
    log("")
    log("  2. Run your training script with flexium:")
    log("     import flexium.auto")
    log("     with flexium.auto.run():")
    log("         # your training code")
    log("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
