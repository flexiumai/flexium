# Changelog

All notable changes to Flexium.AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-23

### Added
- New `flexium.init()` API - simpler than `flexium.auto.run()` context manager
- `disabled` parameter to bypass flexium for benchmarking
- Framework compatibility documentation for Hugging Face, timm, fastai

### Fixed
- Connection failures through Cloudflare proxy (now uses polling transport)
- WebSocket upgrade errors with "Invalid empty packet received"

### Changed
- Simplified API: `flexium.init()` replaces `flexium.auto.run()` as recommended API
- Transport changed from WebSocket to polling for better proxy compatibility

### Removed
- FlexiumCallback for PyTorch Lightning (use `flexium.init()` instead)
- gRPC transport (WebSocket/polling only now)

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
- WebSocket-based server communication
- Web dashboard for monitoring and control
- Pause/Resume functionality to free GPU completely
- Graceful degradation when server is unavailable
- Framework compatibility: PyTorch Lightning, Hugging Face, timm, and more
- GPU UUID targeting for specific hardware selection

### Requirements
- Python 3.8+
- PyTorch 2.0+
- NVIDIA Driver 580+
- Linux x86_64
