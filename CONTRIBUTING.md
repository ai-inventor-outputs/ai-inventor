# Contributing to AI Inventor

Thank you for your interest in contributing!

## Developer Certificate of Origin (DCO)

By contributing to this project, you agree to the [Developer Certificate of Origin](https://developercertificate.org/). Add a `Signed-off-by` line to your commit messages:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` to do this automatically.

## How to contribute

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run the pipeline locally to verify nothing breaks
5. Submit a pull request

## Dev environment

```bash
uv sync --group dev    # adds pytest on top of the runtime deps
.venv/bin/python -m pytest aii_launcher/tests/
```

## Code style

- Python 3.12+, type hints everywhere
- `src/` layout for packages
- `pathlib.Path` for file operations
- Loguru for logging

## Reporting issues

Open an issue on GitHub with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
