import pytest

from .conftest import check_polynomial_root_calculation_3rd_degree


@pytest.mark.parametrize(
    "a, b, c, d",
    [
        (1, -6, 11, -6),  # Roots: 1, 2, 3
        (1, -3, 3, -1),  # Roots: 1, 1, 1
        (2, -4, 2, -1),  # Roots: 0.5, 1.0, 1.0
        (1, 0, -4, 4),  # Roots: -2.0, 2.0
    ],
)
def test_cubic(a, b, c, d):
    """
    Test the polynomial root calculation for a cubic polynomial.
    This function uses pytest to run the test.
    """
    pytest.skip()
    check_polynomial_root_calculation_3rd_degree(a, b, c, d)


def test_toto():
    assert 3 == 4
