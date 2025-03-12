from tinygrad.tensor import Tensor
from tinygrad import dtypes

def ntt_matrix(n, prime, primitive_root):
    # Create omega as the primitive n-th root of unity
    omega = pow(primitive_root, (prime - 1) // n, prime)
    
    # Create row and column indices for broadcasting
    i = Tensor.arange(n, dtype=dtypes.int32).reshape(n, 1)
    j = Tensor.arange(n, dtype=dtypes.int32).reshape(1, n)
    
    # Compute powers matrix (i*j) % n using tinygrad operations
    powers = (i * j) % n
    
    # Precompute all powers of omega as a tensor directly
    omega_values = []
    current = 1
    for k in range(n):
        omega_values.append(current)
        current = (current * omega) % prime
    
    omega_tensor = Tensor(omega_values, dtype=dtypes.int32)
    
    # Using advanced indexing with the powers tensor to get omega^(i*j) % prime
    ntt_matrix = omega_tensor[powers]
    
    return ntt_matrix

def intt_matrix(n, prime, primitive_root):
    # Create omega as the primitive n-th root of unity
    omega = pow(primitive_root, (prime - 1) // n, prime)
    omega_inv = pow(omega, prime - 2, prime)
    
    # Create row and column indices for broadcasting
    i = Tensor.arange(n, dtype=dtypes.int32).reshape(n, 1)
    j = Tensor.arange(n, dtype=dtypes.int32).reshape(1, n)
    
    # Compute powers matrix (i*j) % n using tinygrad operations
    powers = (i * j) % n
    
    # Precompute all powers of omega as a tensor directly
    omega_values = []
    current = 1
    for k in range(n):
        omega_values.append(current)
        current = (current * omega_inv) % prime
    
    omega_tensor = Tensor(omega_values, dtype=dtypes.int32)
    
    # Using advanced indexing with the powers tensor to get omega^(i*j) % prime
    ntt_matrix = omega_tensor[powers]
    n_inv = pow(n, prime - 2, prime)
    ntt_matrix = (ntt_matrix * n_inv) % prime

    return ntt_matrix

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
    # Convert polynomial to Tensor if it's not already
    if not isinstance(polynomial, Tensor):
        polynomial = Tensor(polynomial, dtype=dtypes.uint32)
    
    n = len(polynomial)
    matrix = ntt_matrix(n, prime, primitive_root)
    
    # Ensure polynomial coefficients are within the field
    polynomial = polynomial % prime
    
    # Compute the matrix multiplication (dot product) and take modulo prime
    result = matrix.matmul(polynomial) % prime
    
    return result

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
    # Convert transformed to Tensor if it's not already
    if not isinstance(transformed, Tensor):
        transformed = Tensor(transformed, dtype=dtypes.uint32)
    
    n = len(transformed)
    matrix = intt_matrix(n, prime, primitive_root)
    
    # Ensure transformed values are within the field
    transformed = transformed % prime
    
    # Compute the matrix multiplication (dot product) and take modulo prime
    result = matrix.matmul(transformed) % prime
    
    return result

# Example usage
if __name__ == "__main__":
    # Parameters for a small example
    n = 8  # Must be a power of 2
    prime = 17  # A prime number where (prime - 1) is divisible by n
    primitive_root = 3  # A primitive root modulo prime
    
    # Example polynomial: x^3 + 2x^2 + 3x + 4
    polynomial = Tensor([4, 3, 2, 1, 0, 0, 0, 0], dtype=dtypes.uint32)
    
    # Perform NTT
    transformed = ntt(polynomial, prime, primitive_root)
    print("NTT result:", transformed.numpy())
    
    # Perform INTT to get back the original polynomial
    original = intt(transformed, prime, primitive_root)
    print("INTT result (should match original polynomial):", original.numpy())
    
    # Verify the result
    assert (polynomial.numpy() == original.numpy()).all(), "INTT(NTT(polynomial)) != polynomial"
    print("Verification successful: INTT(NTT(polynomial)) = polynomial")