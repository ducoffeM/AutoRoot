from autoroot.torch.complex.complex_utils import *
from torch import Tensor


def polynomial_root_calculation_3rd_degree(
    a: Tensor, b: Tensor, c: Tensor, d: Tensor
) -> Tensor:
    """
    Calculate the roots of a cubic polynomial using Cardano's method.
    https://en.wikipedia.org/wiki/Cubic_equation
    Args:
        a : Coefficient of x^3, shape (batch_size, 1)
        b : Coefficient of x^2, shape (batch_size, 1)
        c : Coefficient of x, shape (batch_size, 1)
        d : Constant term, shape (batch_size, 1)
    Returns:
        Tensor: Roots of the polynomial, shape (batch_size, 3, 2)
                Each root is represented as a complex number (real, imaginary)
    """
    # This function calculates the roots of a cubic polynomial of the form:
    # a*x^3 + b*x^2 + c*x + d = 0
    # a (batch_size, 1)
    # b (batch_size, 1)
    # c (batch_size, 1)
    # d (batch_size, 1)

    # output: roots of the polynomial in the form of a tensor of shape (batch_size,3, 2)
    # where each root is represented as a complex number (real, imaginary)
    # each row is the i_th root of the polynomial

    batch_size: int = a.shape[0]  # Get the batch size from the shape of a

    # Discriminant terms
    p: Tensor = (3 * a * c - b**2) / (
        3 * a**2
    )  # (batch_size, 1) because element-wise opeations
    q: Tensor = (2 * b**3 - 9 * a * b * c + 27 * a**2 * d) / (
        27 * a**3
    )  # (batch_size, 1)
    delta: Tensor = -4 * p**3 - 27 * q**2  # (batch_size, 1)

    roots: Tensor = torch.empty(
        (batch_size, 3, 2)
    )  # Initialize roots tensor to store the roots

    j_: Tensor = torch.tensor(
        [-0.5, torch.sqrt(torch.tensor(3)) / 2]
    )  # cube root of unity

    for k in range(3):
        delta_sur_27: Tensor = -delta / 27  # (batch_size, 1)

        sqrt_term: Tensor = sqrt_batch(delta_sur_27)

        # todo: check batch
        j_exp_k: Tensor = complex_number_power_k_batch(
            j_, k
        )  # Compute j^k for each batch
        j_exp_sub_k: Tensor = complex_number_power_k_batch(
            j_, -k
        )  # Compute j^-k for each batch

        j_exp_k_batch: Tensor = j_exp_k.repeat(batch_size, 1)
        j_exp_sub_k_batch: Tensor = j_exp_sub_k.repeat(batch_size, 1)

        u_k: Tensor = product_of_2_complex_numbers_batch(
            j_exp_k_batch,
            sqrt_3_batch(
                torch.stack(
                    [0.5 * (-q.squeeze() + sqrt_term[:, 0]), sqrt_term[:, 1]], dim=-1
                )
            ),
        )
        # (batch_size, 2)
        v_k: Tensor = product_of_2_complex_numbers_batch(
            j_exp_sub_k_batch,
            sqrt_3_batch(
                torch.stack(
                    [0.5 * (-q.squeeze() - sqrt_term[:, 0]), -0.5 * sqrt_term[:, 1]],
                    dim=-1,
                )
            ),
        )
        # (batch_size, 2)

        root: Tensor = addition_batch(
            addition_batch(u_k, v_k),
            torch.stack([-b[:, 0] / (3 * a[:, 0]), 0.0 * b[:, 0]], dim=-1),
        )
        # (batch_size, 2)

        roots[:, k, :] = root  # Store the root in the roots tensor

    return roots
