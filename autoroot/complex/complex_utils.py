import torch
import math
from types import Tensor

def addition_batch(a : Tensor,b : Tensor) -> Tensor :
    """
    Adds two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the sum of the two
        batches of complex numbers.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    return torch.stack([a[:,0] + b[:,0], a[:,1] + b[:,1]],dim=-1)       # (batch_size, 2)

def product_of_2_complex_numbers_batch(a : Tensor, b : Tensor) -> Tensor :
   """ 
    Multiplies two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the product of the two
        batches of complex numbers.
    """
   # a = torch.tensor(batch_size*[a_real, a_imag])
   # b = torch.tensor(batch_size*[b_real, b_imag])

   real_part : Tensor  = a[:,0] * b[:,0] - a[:,1] * b[:,1]
   imag_part : Tensor  = a[:,0] * b[:,1] + a[:,1] * b[:,0]
   return torch.stack([real_part, imag_part],dim=-1) # (batch_size, 2)

def sqrt_batch(a : Tensor) -> Tensor :
    """
    Computes the square root of a batch of real numbers.
    Each number is represented as a tensor of shape (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the square root of the
        batch of real numbers, where the first element is the real part and the
        second element is the imaginary part.
    """
    # a.shape = (batch_size,1)    
    # a is a tensor of real numbers, sqrt is element-wise
    real_part : Tensor = torch.where(a >=0, torch.sqrt(a), torch.tensor(0.0)*a)  #(batch_size,1)
    imag_part : Tensor = torch.where(a <0 , torch.sqrt(-a), torch.tensor(0.0)*a) #(batch_size,1)

    return torch.cat((real_part, imag_part),dim=1)  #(batch_size, 2)


def product_complex_real_batch(a : Tensor, b : Tensor) -> Tensor :
    """
    Multiplies a batch of complex numbers by a batch of real numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part. Each real number is represented as a tensor of shape
    (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        b : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the product of the batch
        of complex numbers and the batch of real numbers, where the first element
        is the real part and the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[,a_real, a_imag]) # (batch_size, 2)
    # b is a real number (batch_size,1) # (batch_size, 1)
     
    return torch.stack([a[:,0] * b.squeeze(), a[:,1] * b.squeeze()],dim=-1) # (batch_size, 2)

def inverse_complex_number(a : Tensor) -> Tensor :
    """
    Computes the inverse of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the inverse of the batch
        of complex numbers, where the first element is the real part and the
        second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # a need to be =/= 0 
    # Returns the inverse of the complex number a

    denom : Tensor = a[:,0]**2 + a[:,1]**2
    if torch.any(denom == 0):
        raise("Cannot compute inverse of zero complex number")
    return torch.stack([a[:,0] / denom, -a[:,1] / denom],dim=-1)  # (batch_size, 2)

def complex_number_power_k_batch(a : Tensor, k : int) -> Tensor:
    """
    Computes the k-th power of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        k : An integer representing the power to which the complex numbers are
        raised.
    Returns:
        A tensor of shape (batch_size, 2) representing the k-th power of the
        batch of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # k is an integer

    if k == 0 : 
       return torch.tensor([1.0, 0.0]).repeat(a.shape[0], 1)  # (batch_size, 2)
    elif k == 1:
       return a
    elif k < 0:
       b_exp_sub_k : Tensor = complex_number_power_k_batch(a, -k)
       return inverse_complex_number(b_exp_sub_k)  
    else : 
       result : Tensor = a
       for _ in range(1,k):
           result = product_of_2_complex_numbers_batch(result, a)
    return result

def argument_batch(a : Tensor) -> Tensor : #potentiellemen pb si (0,0)
    """
    Computes the argument (angle) of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 1) representing the argument of the
        batch of complex numbers, where each element is the angle in radians.
    """
   # a = torch.tensor(batch_size*[a_real, a_imag])

    cas_a0_nul : Tensor = torch.where(a[:,1]>0, torch.tensor(math.pi / 2),torch.tensor(-math.pi / 2))
    cas_a0_negatif : Tensor = torch.where(a[:,1] >= 0 ,torch.atan(a[:,1] / a[:,0]) + torch.tensor(math.pi), torch.atan(a[:,1] / a[:,0]) - torch.tensor(math.pi))  
   
    cas_a0_negatif_ou_nul : Tensor = torch.where(a[:,0]==0, cas_a0_nul, cas_a0_negatif) 
   
    result : Tensor = torch.where(a[:,0] > 0, torch.atan(a[:,1] / a[:,0]), cas_a0_negatif_ou_nul) 

    return result.unsqueeze(-1)  # (batch_size, 1)  

def module_batch(a : Tensor) -> Tensor : 
   """
    Computes the modulus of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 1) representing the modulus of the
        batch of complex numbers, where each element is the modulus.
    """
   # a = torch.tensor(batch_size*[a_real, a_imag])
   return torch.sqrt(a[:,0]**2 + a[:,1]**2).unsqueeze(-1)  # (batch_size, 1)

def sqrt_3_batch(a : Tensor) -> Tensor : # for a real number a_imag = 0
    """
    Computes the cube root of a batch of real numbers.
    Each number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part (which is zero for real numbers).
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of real
        numbers, where the second element (imaginary part) is zero.
    Returns:
        A tensor of shape (batch_size, 2) representing the cube root of the
        batch of real numbers, where the first element is the real part and the
        second element is the imaginary part (which is zero).
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    real_part : Tensor = torch.where(a[:,0] >= 0,a[:,0]**(1/3),-(-a[:,0])**(1/3))  # (batch_size, 1)
    imag_part : Tensor = real_part * 0.0  # (batch_size, 1)
    return torch.stack((real_part, imag_part), dim=-1)  # (batch_size, 2)

def sqrt_complex_batch(a : Tensor) -> Tensor:
    """
    Computes the square root of a batch of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the square root of the
        batch of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size,[a_real, a_imag])
    r : Tensor = module_batch(a)     # (batch_size, 1)


    real_part : Tensor = torch.where(r[:,0] != 0.0,torch.sqrt(r[:,0]) * torch.cos(argument_batch(a)[:,0] / 2),r[:,0] * 0.0)
    imag_part : Tensor = torch.where(r[:,0] != 0.0,torch.sqrt(r[:,0]) * torch.sin(argument_batch(a)[:,0] / 2), r[:,0] * 0.0)
    
    return torch.stack((real_part, imag_part), dim=-1)  # (batch_size, 2)

def division_2_complex_numbers(a : Tensor, b :Tensor) -> Tensor:
    """
    Divides two batches of complex numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part.
    Args:
        a : A tensor of shape (batch_size, 2) representing the first
        batch of complex numbers.
        b : A tensor of shape (batch_size, 2) representing the second
        batch of complex numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the division of the two
        batches of complex numbers, where the first element is the real part and
        the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    # b = torch.tensor(batch_size*[b_real, b_imag])
    inv_b : Tensor = inverse_complex_number(b)  # (batch_size, 2)
    return product_of_2_complex_numbers_batch(a, inv_b)  # (batch_size, 2)

def addition_complex_real_batch(a:Tensor, b:Tensor) -> Tensor:
    """
    Adds a batch of complex numbers to a batch of real numbers.
    Each complex number is represented as a tensor of shape (batch_size, 2),
    where the first element is the real part and the second element is the
    imaginary part. Each real number is represented as a tensor of shape
    (batch_size, 1).
    Args:
        a : A tensor of shape (batch_size, 2) representing the batch of complex
        numbers.
        b : A tensor of shape (batch_size, 1) representing the batch of real
        numbers.
    Returns:
        A tensor of shape (batch_size, 2) representing the sum of the batch
        of complex numbers and the batch of real numbers, where the first element
        is the real part and the second element is the imaginary part.
    """
    # a = torch.tensor(batch_size*[a_real, a_imag])
    #  b is a real number (batch_size,1)
    
    return torch.stack([a[:,0] + b[:,0], a[:,1]],dim=-1)  # (batch_size, 2)

    


'''
Test the functions with a batch of complex numbers


batch_size = 5
batch_b = torch.randn(batch_size, 2) 

batch_a_real = torch.randn(batch_size, 1)
batch_a_real_complex_form = torch.cat((batch_a_real, torch.zeros(batch_size, 1)), dim=-1)  # (batch_size, 2)

print("Batch a:", batch_a)
print("Batch a_real:", batch_a_real)


print("Result of addition:", addition_batch(batch_a, batch_b))
print("Result of product:", product_of_2_complex_numbers_batch(batch_a, batch_b))
print("Result of sqrt:", sqrt_batch(batch_a_real))
print("Result of product with real number:", product_complex_real_batch(batch_a, batch_a_real))
print("Result of inverse:", inverse_complex_number(batch_a))
print("Result of power:", complex_number_power_k_batch(batch_a, 0))
print("\n with 1 : ", complex_number_power_k_batch(batch_a, 1))
print("\n with -1 : ", complex_number_power_k_batch(batch_a, -1))
print("\n with 2 : ", complex_number_power_k_batch(batch_a, 2))
print("\n with -2 : ", complex_number_power_k_batch(batch_a, -2))
print("Result of argument:", argument_batch(batch_a))
print("Result of module:", module_batch(batch_a))
print("batch_a_real_complex_form:", batch_a_real_complex_form)
print("Result of sqrt_3_batch:", sqrt_3_batch(batch_a_real_complex_form))
print("Result of sqrt_complex_batch:", sqrt_complex_batch(batch_a))
print("Result of division:", division_2_complex_numbers(batch_a, batch_b))
batch_size = 4  # Define the batch size
batch_a_real = torch.randn(batch_size, 1)

print("Batch a_real:", batch_a_real)
print(batch_a_real.shape)
result = sqrt_batch(batch_a_real)
print("Result of sqrt:",result)
print("Result of sqrt shape:", result.shape)  # Should be (batch_size, 2)


batch_a = torch.randn(batch_size, 2)

print("Result of addition with real number:", addition_complex_real_batch(batch_a, batch_a_real))


'''

