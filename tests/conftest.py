import numpy as np
import torch
from autoroot.torch.complex.complex_utils import *
from autoroot.torch.cubic.cubic import polynomial_root_calculation_3rd_degree
from autoroot.torch.quartic.quartic import (
    polynomial_root_calculation_4th_degree_ferrari,
)

precision = 1e-25  # precision for the complex numbers operations


def check_polynomial_root_calculation_3rd_degree(a, b, c, d):
    """Test the polynomial root calculation for a cubic polynomial.
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


def check_polynomial_root_calculation_4th_degree_ferrari(
    a0,
    a1,
    a2,
    a3,
    a4,
):
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
    def f(x, a0, a1, a2, a3, a4):
        return a4 * x**4 + a3 * x**3 + a2 * x**2 + a1 * x + a0

    # Computation of the roots
    roots_torch = polynomial_root_calculation_4th_degree_ferrari(
        a0_torch, a1_torch, a2_torch, a3_torch, a4_torch
    )
    roots_numpy = roots_torch.cpu().detach().numpy()
    roots_complex = roots_numpy[..., 0] + 1j * roots_numpy[..., 1]

    for r in roots_complex:
        # Calculation of the polynomial applied to the root
        y = f(r, a0, a1, a2, a3, a4)
        np.testing.assert_allclose(np.linalg.norm(y), 0, atol=1e-6)
        # Check if the polynomial evaluated at the root is close to zero (<10^(-10))


def check_addition_batch(a, b):
    """
    Test the addition of two complex numbers using the addition_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)
    b_tensor = torch.tensor([np.real(b), np.imag(b)], dtype=torch.float64).unsqueeze(0)

    addition_tensor = addition_batch(a_tensor, b_tensor)
    addition_numpy = a + b

    np.testing.assert_allclose(addition_tensor[0, 0], np.real(addition_numpy), atol=precision)
    np.testing.assert_allclose(addition_tensor[0, 1], np.imag(addition_numpy), atol=precision)


def check_product_of_2_complex_numbers_batch(a, b):
    """Check the product of two complex numbers represented as tensors.
    Args :
        a (complex): First complex number
        b (complex): Second complex number
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)
    b_tensor = torch.tensor([np.real(b), np.imag(b)], dtype=torch.float64).unsqueeze(0)

    product_tensor = product_of_2_complex_numbers_batch(a_tensor, b_tensor)
    product_numpy = a * b

    np.testing.assert_allclose(product_tensor[0, 0], np.real(product_numpy), atol=precision)
    np.testing.assert_allclose(product_tensor[0, 1], np.imag(product_numpy), atol=precision)


def check_sqrt_batch(a):
    """Compute the square root of a tensor element-wise.
    Args : a (complex)"""
    a_tensor = torch.tensor([a], dtype=torch.float64).unsqueeze(0)

    sqrt_tensor = sqrt_batch(a_tensor)
    sqrt_numpy = np.sqrt(a, dtype=np.complex128)

    np.testing.assert_allclose(sqrt_tensor[0, 0], np.real(sqrt_numpy), atol=precision)
    np.testing.assert_allclose(sqrt_tensor[0, 1], np.imag(sqrt_numpy), atol=precision)


def check_product_complex_real_batch(a, b):
    """Check the product of a complex number and a real number represented as tensors.
    Args :
        a (complex): Complex number
        b (float): Real number
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)
    b_tensor = torch.tensor([b], dtype=torch.float64).unsqueeze(0)

    product_tensor = product_complex_real_batch(a_tensor, b_tensor)
    print(product_tensor)
    product_numpy = a * b
    print(product_numpy)

    np.testing.assert_allclose(product_tensor[0, 0], np.real(product_numpy), atol=precision)
    np.testing.assert_allclose(product_tensor[0, 1], np.imag(product_numpy), atol=precision)


def check_inverse_complex_batch(a):
    """
    Test the inverse of a complex number using the product_of_2_complex_numbers_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)

    inverse_tensor = inverse_complex_number(a_tensor)
    inverse_numpy = 1 / a

    np.testing.assert_allclose(inverse_tensor[0, 0], np.real(inverse_numpy), atol=precision)
    np.testing.assert_allclose(inverse_tensor[0, 1], np.imag(inverse_numpy), atol=precision)


def check_complex_number_power_k_batch(a, k):
    """
    Test the power of a complex number using the complex_number_power_k_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)

    power_tensor = complex_number_power_k_batch(a_tensor, k)
    power_numpy = a**k

    np.testing.assert_allclose(power_tensor[0, 0], np.real(power_numpy), atol=precision)
    np.testing.assert_allclose(power_tensor[0, 1], np.imag(power_numpy), atol=precision)


def check_argument_batch(a):
    """
    Test the argument of a complex number using the argument_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)

    argument_tensor = argument_batch(a_tensor)
    argument_numpy = np.angle(a)

    np.testing.assert_allclose(argument_tensor[0], argument_numpy, atol=precision)


def check_module_batch(a):
    """
    Test the module of a complex number using the module_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)

    module_tensor = module_batch(a_tensor)
    module_numpy = np.abs(a)

    np.testing.assert_allclose(module_tensor[0], module_numpy, atol=precision)


def check_sqrt_3_batch(a):
    """
    Test the square root of a real number (a complex withe the real part equal to zero) using the sqrt_3_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([a, 0.0], dtype=torch.float64).unsqueeze(0)

    sqrt_3_tensor = sqrt_3_batch(a_tensor)
    sqrt_3_numpy = np.cbrt(a)

    np.testing.assert_allclose(sqrt_3_tensor[0:0], sqrt_3_numpy, atol=precision)


def check_sqrt_complex_batch(a):
    """
    Test the square root of a complex number using the sqrt_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)

    sqrt_tensor = sqrt_complex_batch(a_tensor)
    sqrt_numpy = np.sqrt(a, dtype=np.complex128)

    np.testing.assert_allclose(sqrt_tensor[0, 0], np.real(sqrt_numpy), atol=precision)
    np.testing.assert_allclose(sqrt_tensor[0, 1], np.imag(sqrt_numpy), atol=precision)


def check_division_2_complex_numbers(a, b):
    """
    Test the division of two complex numbers using the division_2_complex_numbers function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)
    b_tensor = torch.tensor([np.real(b), np.imag(b)], dtype=torch.float64).unsqueeze(0)

    division_tensor = division_2_complex_numbers(a_tensor, b_tensor)
    division_numpy = a / b

    np.testing.assert_allclose(division_tensor[0, 0], np.real(division_numpy), atol=precision)
    np.testing.assert_allclose(division_tensor[0, 1], np.imag(division_numpy), atol=precision)


def check_addition_complex_real_batch(a, b):
    """
    Test the addition of a complex number and a real number using the addition_complex_real_batch function.
    This function uses pytest to run the test.
    """
    a_tensor = torch.tensor([np.real(a), np.imag(a)], dtype=torch.float64).unsqueeze(0)
    b_tensor = torch.tensor([b], dtype=torch.float64).unsqueeze(0)

    addition_tensor = addition_complex_real_batch(a_tensor, b_tensor)
    addition_numpy = a + b

    np.testing.assert_allclose(addition_tensor[0, 0], np.real(addition_numpy), atol=precision)
    np.testing.assert_allclose(addition_tensor[0, 1], np.imag(addition_numpy), atol=precision)
