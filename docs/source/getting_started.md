# Getting started

First, define the coefficients of your quartic polynomial $ax^4 + bx^3 + cx^2 + dx + e = 0$ using `torch.tensor`. You can then call the `solve_quartic` method from `autoroot_torch` to obtain the roots of the polynomial.

```python
# Imports
import torch
from autoroot.torch.quartic import solve_quartic

# Define the coefficients of the quartic equation: x^4 - 10x^3 + 35x^2 - 50x + 24 = 0
# (This polynomial has real roots at 1, 2, 3, and 4)
a = torch.tensor([[1.0]])
b = torch.tensor([[-10.0]])
c = torch.tensor([[35.0]])
d = torch.tensor([[-50.0]])
e = torch.tensor([[24.0]])

# Solve the quartic equation
# The roots are returned as a tensor of shape (batch_size, 4, 2),
# where each root is represented as [real_part, imaginary_part].
roots = solve_quartic(a, b, c, d, e)

print("Roots found by autoroot_torch:")
print(roots)
```

Other advanced features, such as differentiability for optimization tasks, are possible and illustrated
in our [tutorials](tutorials).
