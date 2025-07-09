# AutoRoot
AutoRoot: Differentiable Root Solvers for Cubic and Quartic Polynomials AutoRoot is a fast, fully differentiable PyTorch library for solving cubic and quartic equations


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
  
📄 .pre-commit-config.yaml       # Pre-commit hooks config   
📄 pyproject.toml                # Build system and tool configs  
📄 README.md                     # Project overview  
📄 tox.ini                       # Tox testing configuration  
```
# Dependencies
AutoRoot primarily relies on PyTorch for its core tensor operations and differentiability features. numpy is also used within the test suite for robust result verification.  

The primary dependencies for the library are:  
    - PyTorch: For tensor computation and automatic differentiation.
    - Python: Version 3.8 or higher.  

For running the test suite, you will also need:  
    - Pytest: A testing framework.
    - NumPy: Used for numerical comparisons and handling tolerances in tests.  

All required dependencies for the project and its testing are listed in the pyproject.toml file.

# Using 
The main functions are located in the autoroot.torch.cubic and autoroot.torch.quartic modules.   

Input : coefficiants of the polynomial ->  torch.Tensor of size (batch_size, 1)  
Output : roots of the polynomial -> torch.Tensor where each complex root is a [real, imaginary]  vector.  

Please note that these methods are designed for polynomials with real coefficients.    

Complex numbers are represented as torch.Tensor of shape (..., 2), where the first value is the real part and the second is the imaginary part. Utilities for these operations are available in autoroot.torch.complex.complex_utils.py.  

# Tests
A comprehensive test suite is provided in the tests/ folder to ensure the reliability and robustness of the root calculations.  
Tests are written using pytest and utilize numpy for result verification (especially for comparison with reference values or handling tolerances).
