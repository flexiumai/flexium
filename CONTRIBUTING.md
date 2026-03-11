# Contributing to Flexium.AI

Thank you for your interest in contributing to Flexium.AI! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/flexium.git
   cd flexium
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Running Linting

```bash
ruff check flexium/
mypy flexium/
```

### Code Style

- We use [ruff](https://github.com/astral-sh/ruff) for linting
- We use [mypy](https://mypy.readthedocs.io/) for type checking
- Follow [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Add type hints to all function signatures

## Submitting Changes

1. Create a new branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run tests and linting:
   ```bash
   pytest tests/ -v
   ruff check flexium/
   mypy flexium/
   ```

4. Commit your changes with a clear message:
   ```bash
   git commit -m "Add feature: description of your changes"
   ```

5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

6. Open a Pull Request

## Pull Request Guidelines

- Include a clear description of the changes
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass
- Keep PRs focused - one feature/fix per PR

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Include system information (Python version, OS, GPU, driver version)

## Code of Conduct

Please be respectful and constructive in all interactions. We're all here to build something great together.

## Questions?

Feel free to open an issue for any questions about contributing.
