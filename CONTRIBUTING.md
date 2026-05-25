# Contributing to ML Algorithms from Scratch

Contributions are welcome! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone <your-fork>`
3. Create a branch: `git checkout -b feature/your-feature`
4. Make changes and commit: `git commit -am 'Add your feature'`
5. Push to branch: `git push origin feature/your-feature`
6. Submit a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/Devil-92/ml-algos-from-scratch.git
cd ml-algos-from-scratch

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v
```

## Code Style

- Follow PEP 8 conventions
- Use type hints for functions
- Write docstrings for all classes and functions
- Keep functions focused and modular
- Use meaningful variable names

## Example Docstring Format

```python
def function_name(param1: type1, param2: type2) -> return_type:
    """
    Brief description of function.
    
    Longer description if needed. Explain algorithm, mathematical
    foundations, or important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
    """
```

## Adding a New Algorithm

### 1. Create Implementation File

Create a new file in appropriate directory:
```
src/[supervised|unsupervised|neural_networks]/algorithm_name.py
```

### 2. Implement Algorithm

- Follow existing code style
- Include comprehensive docstring
- Add mathematical explanation
- Use type hints

### 3. Create Tests

Add tests in `tests/test_category.py`:
```python
class TestNewAlgorithm:
    """Tests for NewAlgorithm."""
    
    def test_initialization(self):
        """Test algorithm initialization."""
        
    def test_fit(self):
        """Test fitting."""
        
    def test_predict(self):
        """Test prediction."""
```

### 4. Update Documentation

- Add to relevant file in `docs/`
- Update `ALGORITHMS.md` with reference
- Update main `README.md` if needed

### 5. Add Example

Add example to demo script in `examples/`

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_supervised.py -v

# Run specific test class
pytest tests/test_supervised.py::TestLinearRegression -v

# Run specific test
pytest tests/test_supervised.py::TestLinearRegression::test_fit -v
```

## Commit Messages

Use clear, descriptive commit messages:

```
Add K-Means++ algorithm with better initialization

- Implement K-Means++ initialization
- Add tests for better convergence
- Update documentation with complexity analysis
```

## Pull Request Process

1. Update README.md with any changes
2. Update docs if adding new features
3. Ensure all tests pass: `pytest tests/`
4. Add tests for new features (at least 80% coverage)
5. Keep PR focused on single feature/fix
6. Write clear PR description

## Code Review

When submitting a PR:
- Explain what problem you're solving
- Describe the implementation approach
- Ask for specific feedback if needed
- Be open to suggestions

## Issues

### Reporting Issues

Include:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Error message/traceback

### Feature Requests

Include:
- Use case and motivation
- Proposed solution (if any)
- Alternative approaches considered

## Documentation

Help improve documentation:
- Fix typos
- Clarify explanations
- Add examples
- Improve mathematical notation
- Translate to other languages

## Testing Guidelines

- Write tests for all new features
- Aim for high coverage (>80%)
- Test edge cases
- Test error handling
- Test with different data sizes

## Performance

When optimizing:
- Benchmark before and after
- Document performance improvements
- Use NumPy vectorization
- Avoid nested loops where possible

## Mathematical Accuracy

When implementing algorithms:
- Verify against research papers
- Include citations
- Test against known results
- Provide numerical examples

## Style Guide

### File Organization
```python
# 1. Docstring
"""Module description."""

# 2. Imports
import numpy as np
import sys

# 3. Classes/Functions
class ClassName:
    """Class description."""
    pass

def function_name():
    """Function description."""
    pass
```

### Variable Naming
- Use snake_case for variables/functions
- Use UPPER_CASE for constants
- Use meaningful names (not x, y, z)
- Use mathematical notation only in math explanations

### Comments
Only comment:
- Complex algorithms
- Non-obvious math
- Important constraints
- Not on obvious code

## Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- Papers and textbooks for algorithms

## Questions?

- Open an issue
- Check existing issues
- Email maintainer

Thank you for contributing! 🎉
