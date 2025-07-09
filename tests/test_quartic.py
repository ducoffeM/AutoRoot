import pytest
from .conftest import check_polynomial_root_calculation_4th_degree_ferrari
@pytest.mark.parametrize(
    "a0, a1, a2, a3, a4",
    [
        (1, -6, 11, -6, 1),     # Roots: 1, 2, 3
        (1, -4, 6, -4, 1),      # Roots: 1, 1, 2
        (2, -8, 8, -2, 2),      # Roots: 0.5, 1.0, 2.0
        (1, -5, 6, -4, 1),      # Roots: 1.0 + i*sqrt(3), 1.0 - i*sqrt(3)
        (1, 0, -4, 4, 1),       # Roots: -2.0, 2.0
        (1, 0, 0, 0, 1),        # Roots: 0.0, 0.0, 0.0, 0.0
        (1, -2, 1, -1, 1),      # Roots: 1.0 + i*sqrt(3), 1.0 - i*sqrt(3)
    ],
)
def test_quartic(a0, a1, a2, a3, a4):
    """
    Test the polynomial root calculation for a quartic polynomial using Ferrari's method.
    This function uses pytest to run the test.
    """
    check_polynomial_root_calculation_4th_degree_ferrari(a0, a1, a2, a3, a4)