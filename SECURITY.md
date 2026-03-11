# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in Flexium.AI, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email details to the maintainers (see repository contact info)
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a detailed response within 7 days.

## Security Considerations

### Network Security

- The orchestrator listens on gRPC port 50051 by default
- The dashboard listens on HTTP port 8080 by default
- Consider running behind a firewall or VPN in production
- No built-in authentication (rely on network security)

### GPU Access

- Flexium requires access to NVIDIA GPUs
- Uses NVML for GPU monitoring
- Requires appropriate user permissions for GPU access

### Checkpoint Data

- Training checkpoints may contain sensitive model data
- Checkpoints are stored locally by default
- Consider encrypting checkpoint storage for sensitive models
