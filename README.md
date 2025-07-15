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
AutoRoot serves as a standalone and versatile solution for anyone needing to solve polynomial equations of degrees 3 to 4 with *PyTorch*. This library leverages classical algebraic methods, specifically **Cardano's method** for cubic equations and **Ferrari's method** for quartic equations, to provide accurate and fully differentiable root solutions. The differentiability allows for seamless integration into PyTorch's autograd system, making it ideal for applications requiring gradient-based optimization and neural network integration.



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
  - Pytest (for running the test suite, only for contributors)

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

# Usage
The main functions are located in the `autoroot.torch.cubic` and `autoroot.torch.quartic` modules.

- **Input:** Each coefficient (a, b, ..., e) is a tensor of shape `(batch_size, 1)`.
- **Output:** A tensor of shape `(batch_size, N, 2)` where `N = 3` for cubic or `4` for quartic. Each root is represented as a pair `[real, imaginary]`.

Please note that these methods are designed for polynomials with real coefficients. Complex numbers are represented as `torch.Tensor` of shape `(..., 2)`, where the first value is the real part and the second is the imaginary part. Utilities for these operations are available in `autoroot.torch.complex.complex_utils.py`.

### Example: Solving a Cubic Equation

```python
import torch
from autoroot.torch.cubic import solve_cubic

# Example: x^3 - 6x^2 + 11x - 6 = 0 (roots are 1, 2, 3)
# Coefficients are ordered as [a, b, c, d] for ax^3 + bx^2 + cx + d = 0
a = torch.tensor([[1.0]])
b = torch.tensor([[-6.0]])
c = torch.tensor([[11.0]])
d = torch.tensor([[-6.0]])

roots = solve_cubic(a, b, c, d)
print("Cubic roots:", roots)
# Expected output (order might vary, imaginary parts should be close to zero):
# Cubic roots: tensor([[[1.0000e+00, 0.0000e+00],
#                        [3.0000e+00, 0.0000e+00],
#                        [2.0000e+00, 0.0000e+00]]])

# Example: With a batch of inputs
A = torch.tensor([[1.0], [1.0]])
B = torch.tensor([[-6.0], [0.0]])
C = torch.tensor([[11.0], [0.0]])
D = torch.tensor([[-6.0], [-8.0]]) # x^3 - 8 = 0 (roots: 2, -1+sqrt(3)i, -1-sqrt(3)i)

roots_batch = solve_cubic(A, B, C, D)
print("Batch cubic roots:", roots_batch)

# 🚀 Installation
You can install AutoRoot using pip:

Bash
pip install autoroot
Alternatively, to install from source:

Bash

git clone https://github.com/your-repo/autoroot.git # Replace with your actual repo URL
cd autoroot
pip install .
