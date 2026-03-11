# Changelog

All notable changes to Flexium.AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-03-10

### Fixed
- Dashboard logo not displaying after pip install (logos now bundled in package)

### Changed
- Updated project URLs to point to flexiumai/flexium repository

## [0.1.0] - 2026-03-10

### Added
- Initial public release
- `flexium.auto.run()` context manager for migration-enabled training
- Live GPU migration with zero memory residue (requires NVIDIA driver 580+)
- gRPC-based orchestrator for multi-process coordination
- Web dashboard for monitoring and control
- CLI tool (`flexium-ctl`) for server and process management
- Pause/Resume functionality to free GPU completely
- Automatic GPU error recovery (OOM, ECC errors, device assert)
- Graceful degradation when orchestrator is unavailable
- PyTorch Lightning integration via `FlexiumCallback`
- GPU UUID targeting for specific hardware selection

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NVIDIA Driver 580+
- Linux x86_64
