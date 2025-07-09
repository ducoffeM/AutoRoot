import numpy as np
import torch
from autoroot.torch.cubic.cubic import polynomial_root_calculation_3rd_degree
from autoroot.torch.quartic.quartic import polynomial_root_calculation_4th_degree_ferrari


def check_polynomial_root_calculation_3rd_degree(a, b, c, d):
    """    Test the polynomial root calculation for a cubic polynomial.
    Args:
        a (float): Coefficient of x^3
        b (float): Coefficient of x^2
        c (float): Coefficient of x^1
        d (float): Constant term
    """
    # Calculation of the polynomial
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Computation of the roots
    a_torch = torch.tensor(a, dtype=torch.float64).reshape(-1, 1)  # Reshape to (batch_size, 1)
    b_torch = torch.tensor(b, dtype=torch.float64).reshape(-1, 1)
    c_torch = torch.tensor(c, dtype=torch.float64).reshape(-1, 1)
    d_torch = torch.tensor(d, dtype=torch.float64).reshape(-1, 1)

    roots_torch = polynomial_root_calculation_3rd_degree(a_torch, b_torch, c_torch, d_torch)

    roots_numpy = roots_torch.cpu().detach().numpy()
    roots_complex = roots_numpy[..., 0] + 1j * roots_numpy[..., 1]

    for r in roots_complex:
        # Calculation of the polynomial applied to the root
        y = f(r, a, b, c, d)
        np.testing.assert_allclose(
            np.linalg.norm(y), 0, atol=1e-10
        )  # Check if the polynomial evaluated at the root is close to zero (<10^(-10))

def check_polynomial_root_calculation_4th_degree_ferrari(a0,a1,a2,a3,a4,):
    """Test the polynomial root calculation for a quartic polynomial using Ferrari's method.
    Args:
        a0 (float): Constant term
        a1 (float): Coefficient of x^1
        a2 (float): Coefficient of x^2
        a3 (float): Coefficient of x^3
        a4 (float): Coefficient of x^4 need to b different from 0
    """

    a0_torch = torch.tensor(a0, dtype=torch.float64).reshape(-1, 1)  # Reshape to (batch_size, 1)
    a1_torch = torch.tensor(a1, dtype=torch.float64).reshape(-1, 1)
    a2_torch = torch.tensor(a2, dtype=torch.float64).reshape(-1, 1)
    a3_torch = torch.tensor(a3, dtype=torch.float64).reshape(-1, 1)
    a4_torch = torch.tensor(a4, dtype=torch.float64).reshape(-1, 1)

    # Calculation of the polynomial 
    def f(x,a0,a1,a2,a3,a4) :
        return a4*x**4+a3*x**3 + a2*x**2 + a1*x + a0

    # Computation of the roots
    roots_torch = polynomial_root_calculation_4th_degree_ferrari(a0_torch, a1_torch, a2_torch, a3_torch, a4_torch)
    roots_numpy = roots_torch.cpu().detach().numpy()
    roots_complex = roots_numpy[..., 0] + 1j * roots_numpy[..., 1]
  

    for r in roots_complex :
        # Calculation of the polynomial applied to the root 
        y = f(r,a0,a1,a2,a3,a4)
        np.testing.assert_allclose(np.linalg.norm(y), 0, atol=1e-6) 
         # Check if the polynomial evaluated at the root is close to zero (<10^(-10))
