import numpy as np
import torch
from autoroot.torch.cubic.cubic import polynomial_root_calculation_3rd_degree


def check_polynomial_root_calculation_3rd_degree(a, b, c, d):
    # Polynomial whose roots we are looking for :

    # Calculation of the polynomial
    def f(x, a, b, c, d):
        return a * x**3 + b * x**2 + c * x + d

    # Computation of the roots
    a_torch = torch.tensor(a, dtype=torch.float32).reshape(-1, 1)  # Reshape to (batch_size, 1)
    b_torch = torch.tensor(b, dtype=torch.float32).reshape(-1, 1)
    c_torch = torch.tensor(c, dtype=torch.float32).reshape(-1, 1)
    d_torch = torch.tensor(d, dtype=torch.float32).reshape(-1, 1)

    roots_torch = polynomial_root_calculation_3rd_degree(a_torch, b_torch, c_torch, d_torch)
    roots_numpy = roots_torch.cpu().detach().numpy()  # Convert to numpy array for easier handling

    valeur_ok = True  # true if all roots values are close to zero (<10^(-10))

    for r in roots_numpy:
        # Calculation of the polynomial applied to the root
        y = f(r, a, b, c, d)
        np.testing.assert_allclose(
            np.linalg.norm(y), 0, atol=1e-10
        )  # Check if the polynomial evaluated at the root is close to zero
