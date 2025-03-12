from tinygrad.tensor import Tensor
from tinygrad import dtypes
import numpy as np

def ntt(polynomial, prime, primitive_root):
    """
    Compute the Number Theoretic Transform of a polynomial.
    
    Args:
        polynomial: Coefficient vector of the polynomial (Tensor or list/array)
        prime: The prime modulus
        primitive_root: A primitive root modulo prime
    
    Returns:
        The NTT of the polynomial as a Tensor
    """
    return _transform(polynomial, prime, primitive_root, inverse=False)

def intt(transformed, prime, primitive_root):
    """
    Compute the Inverse Number Theoretic Transform.
    
    Args:
        transformed: The transformed polynomial (Tensor or list/array)
        prime: The prime modulus
        primitive_root: A primitive root modulo prime
    
    Returns:
        The original polynomial coefficients as a Tensor
    """
    return _transform(transformed, prime, primitive_root, inverse=True)

def _transform(x, prime, primitive_root, inverse=False):
    """Internal function to perform NTT or INTT transformation"""
    if not isinstance(x, Tensor):
        x = Tensor(x, dtype=dtypes.uint32)
    
    dtype = x.dtype
    n = len(x)
    
    # Create powers matrix (i*j) % n
    i = Tensor.arange(n, dtype=dtype).reshape(n, 1)
    j = Tensor.arange(n, dtype=dtype).reshape(1, n)
    powers = (i * j) % n
    
    # Get omega (or inverse omega for INTT)
    omega = pow(primitive_root, (prime - 1) // n, prime)
    if inverse:
        omega = pow(omega, prime - 2, prime)  # Modular inverse
    
    # Compute all powers of omega
    omega_powers = np.ones(n, dtype=np.uint32)
    current = 1
    for k in range(n):
        omega_powers[k] = current
        current = (current * omega) % prime
    
    # Create transformation matrix
    matrix = Tensor(omega_powers, dtype=dtypes.uint32)[powers]
    
    # For inverse, apply scaling factor (1/n mod prime)
    if inverse:
        n_inv = pow(n, prime - 2, prime)
        matrix = (matrix * n_inv) % prime
    
    # Perform the transformation
    return (matrix @ (x % prime)) % prime

if __name__ == "__main__":
    # Example
    n, prime, primitive_root = 8, 17, 3
    polynomial = Tensor([4, 3, 2, 1, 0, 0, 0, 0], dtype=dtypes.uint32)
    
    transformed = ntt(polynomial, prime, primitive_root)
    print("NTT result:", transformed.numpy())
    
    original = intt(transformed, prime, primitive_root)
    print("INTT result:", original.numpy())
    
    assert (polynomial.numpy() == original.numpy()).all(), "INTT(NTT(polynomial)) != polynomial"
    print("Verification successful")