# AutoRoot
AutoRoot : Differentiable Root Solvers for Cubic and Quartic Polynomials
AutoRoot is a fast, fully differentiable PyTorch library for solving cubic and quartic equations


## 📚 Table of Contents
- [Motivation](#motivation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Tests](#tests)

# Motivation
While initially developed for solving P3P problems (as part of improving the performance of models like YOLO-NAS), AutoRoot serves as a standalone and versatile solution for anyone needing to solve polynomial equations of these degrees with PyTorch.


# Project Structure :
```
📂 autoroot/                    # Main source director
├── 📁 torch/                   # Torch-related implementations
│   ├── 📁 complex/             # Complex functions utils
│   │   ├── 📄 complex_utils.py
│   ├── 📁 cubic/               # Cubic solver
│   │   ├── 📄 cubic.py
│   ├── 📁 quartic/             # Quartic solver
│   │   ├── 📄 quartic.py
│   ├── 📄 types.py

📂 tests/                        # Unit tests
├── 📄 conftest.py
├── 📄 test_cubic.py
├── 📄 test_quartic.py
├── 📄 test_complex.py

📄 .pre-commit-config.yaml       # Pre-commit hooks config
📄 pyproject.toml                # Build system and tool configs
📄 README.md                     # Project overview
📄 tox.ini                       # Tox testing configuration
```
# Dependencies
AutoRoot relies primarily on:
  - Python ≥ 3.8
  - PyTorch ≥ 1.10 (for tensor operations and autograd)
  - NumPy (for testing and reference comparisons)
  - Pytest (for running the test suite)

All dependencies are declared in the pyproject.toml.

# Usage
The main functions are located in the autoroot.torch.cubic and autoroot.torch.quartic modules.
  - Input: Each coefficient (a, b, ..., e) is a tensor of shape (batch_size, 1)
  - Output: A tensor of shape (batch_size, N, 2) where N = 3 for cubic or 4 for quartic.
        Each root is represented as a pair [real, imaginary].

Please note that these methods are designed for polynomials with real coefficients.

Complex numbers are represented as torch.Tensor of shape (..., 2), where the first value is the real part and the second is the imaginary part.
Utilities for these operations are available in autoroot.torch.complex.complex_utils.py.

# Tests
A comprehensive test suite is provided in the tests/ folder to ensure the reliability and robustness of the root calculations.
Tests are written using pytest and utilize numpy for result verification (especially for comparison with reference values or handling tolerances).
