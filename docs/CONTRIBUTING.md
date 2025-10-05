# Contributing to SPIRIT

Thank you for your interest in contributing to SPIRIT! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of PyTorch and audio processing
- Familiarity with adversarial machine learning concepts

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/spirit-breaking.git
   cd spirit-breaking
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/original/spirit-breaking.git
   ```

## Development Setup

### 1. Create a Virtual Environment

```bash
# Using venv (recommended)
python -m venv spirit-env
source spirit-env/bin/activate  # On Windows: spirit-env\Scripts\activate

# Or using conda
conda create -n spirit-env python=3.8
conda activate spirit-env
```

### 2. Install Dependencies

```bash
# Install in development mode
pip install -e .

# Install additional development dependencies
pip install -r requirements-dev.txt
```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Docstrings**: NumPy style for all public functions and classes
- **Type hints**: Required for all function parameters and return values
- **Imports**: Organized using `isort`

### Code Formatting

We use automated tools to maintain consistent code style:

```bash
# Format code
black .
isort .

# Check formatting
black --check .
isort --check-only .
```

### Docstring Standards

All public functions and classes must have docstrings following the NumPy style:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of what the function does.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2. Default is 0.

    Returns
    -------
    bool
        Description of return value.

    Examples
    --------
    >>> example_function("test", 5)
    True
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=activation_patching --cov=torchattacks

# Run specific test file
pytest tests/test_noise_analyzer.py

# Run tests in parallel
pytest -n auto
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Include both unit tests and integration tests
- Mock external dependencies when appropriate

Example test structure:

```python
import pytest
import torch
from activation_patching.noise_analyzer import detect_noise_sensitive_neurons

class TestNoiseAnalyzer:
    def test_detect_noise_sensitive_neurons_basic(self):
        """Test basic functionality of noise-sensitive neuron detection."""
        # Setup
        model = create_mock_model()
        clean_input = torch.randn(1, 10, 100)
        noisy_input = torch.randn(1, 10, 100)
        
        # Execute
        neurons, clean_acts, noisy_acts = detect_noise_sensitive_neurons(
            model, clean_input, noisy_input, top_k_percent=0.1
        )
        
        # Assert
        assert isinstance(neurons, list)
        assert len(neurons) > 0
        assert all(isinstance(n, tuple) and len(n) == 2 for n in neurons)
```

## Documentation

### Documentation Standards

- Keep documentation up-to-date with code changes
- Use clear, concise language
- Include code examples where appropriate
- Update the README.md for significant changes

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

## Pull Request Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Write clear, well-documented code
- Add tests for new functionality
- Update documentation as needed
- Follow the code style guidelines

### 3. Commit Your Changes

Use conventional commit messages:

```bash
git commit -m "feat: add new defense strategy"
git commit -m "fix: resolve memory leak in neuron detection"
git commit -m "docs: update API documentation"
```

Commit types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### 4. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:

- **Clear title**: Describe the change concisely
- **Detailed description**: Explain what, why, and how
- **Related issues**: Link to any related issues
- **Screenshots**: If applicable (e.g., for UI changes)

### 5. Pull Request Template

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Code refactoring
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes

## Related Issues
Closes #123
```

## Issue Reporting

### Before Creating an Issue

1. Check existing issues to avoid duplicates
2. Search the documentation for solutions
3. Try to reproduce the issue with minimal code

### Issue Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.8.10]
- PyTorch version: [e.g., 2.0.0]
- CUDA version: [e.g., 11.8]

## Additional Information
Any other relevant information.
```

## Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate of others
- Use inclusive language
- Be open to constructive feedback
- Help others learn and grow

### Communication

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Pull Requests**: For code contributions

### Recognition

Contributors will be recognized in:
- The project README
- Release notes
- GitHub contributors page

## Getting Help

If you need help:

1. Check the [documentation](docs/)
2. Search existing [issues](https://github.com/original/spirit-breaking/issues)
3. Start a [discussion](https://github.com/original/spirit-breaking/discussions)
4. Contact maintainers directly

## Development Roadmap

### Current Priorities

- [ ] Improve test coverage
- [ ] Add support for more audio models
- [ ] Optimize defense performance
- [ ] Enhance documentation

### Future Ideas

- [ ] Web interface for easy experimentation
- [ ] Pre-trained defense models
- [ ] Integration with popular ML frameworks
- [ ] Real-time defense capabilities

## License

By contributing to SPIRIT, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to SPIRIT! Your contributions help make the project better for everyone. 