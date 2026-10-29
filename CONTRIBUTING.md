# Contributing to HR FAQ Chatbot

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork-url>`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt -r backend/requirements.txt`
5. Install the git hooks: `pip install pre-commit && pre-commit install`

## Development Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small

- Lint and format with ruff (rules in `ruff.toml`): `ruff check backend/ dspy_module/ tests/` and `ruff format backend/ dspy_module/ tests/`

### Testing

- Run the tests: `pytest -q` (no GPU, no network calls, no LLM API key needed)
- Add tests for new features
- If you change retrieval, OOD detection or generation, rerun the evaluation and report numbers with their sample size and confidence interval

### Commit Messages

- Follow [Conventional Commits](https://www.conventionalcommits.org/): `fix(rag): ...`, `feat(api): ...`, `docs: ...`
- Reference issues when applicable: `Fix #123`

### Pull Request Process

1. Update documentation if needed
2. Ensure code passes all checks
3. Update CHANGELOG.md if applicable
4. Submit PR with clear description

## Areas for Contribution

- Model improvements
- Additional evaluation metrics
- Documentation improvements
- Bug fixes
- Performance optimizations

## Questions?

Open an issue for questions or discussions. Report security problems privately, see [SECURITY.md](SECURITY.md).

