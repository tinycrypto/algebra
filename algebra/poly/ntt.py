from tinygrad.tensor import Tensor
from tinygrad import dtypes

def ntt_matrix(n, prime, primitive_root):
    """
    Generate the NTT matrix for a given size n, prime modulus, and primitive root.
    
    Args:
        n: Size of the transform
        prime: The prime modulus
        primitive_root: A primitive n-th root of unity modulo prime
        
    Returns:
        n x n Tensor for the NTT transform
    """
    # Create omega as the primitive n-th root of unity
    omega = pow(primitive_root, (prime - 1) // n, prime)
    
    # Generate the NTT matrix using tinygrad Tensor
    matrix = Tensor.zeros(n, n, dtype=dtypes.uint32).contiguous()
    
    # Fill the matrix manually using elementwise operations
    for i in range(n):
        for j in range(n):
            # Compute omega^(i*j) mod prime
            power = (i * j) % n
            value = pow(omega, power, prime)
            # Assign value to matrix position (i,j)
            # We need to use a temporary tensor and update row by row
            row = matrix[i].numpy()
            row[j] = value
            matrix[i] = Tensor(row, dtype=dtypes.uint32)
    
    return matrix

def intt_matrix(n, prime, primitive_root):
    """
    Generate the inverse NTT matrix for a given size n, prime modulus, and primitive root.
    
    Args:
        n: Size of the transform
        prime: The prime modulus
        primitive_root: A primitive n-th root of unity modulo prime
        
    Returns:
        n x n Tensor for the inverse NTT transform
    """
    # Create omega as the primitive n-th root of unity
    omega = pow(primitive_root, (prime - 1) // n, prime)
    # Compute inverse of omega
    omega_inv = pow(omega, prime - 2, prime)
    
    # Generate the inverse NTT matrix
    matrix = Tensor.zeros(n, n, dtype=dtypes.uint32).contiguous()
    
    # Fill the matrix manually using elementwise operations
    for i in range(n):
        for j in range(n):
            # Compute omega_inv^(i*j) mod prime
            power = (i * j) % n
            value = pow(omega_inv, power, prime)
            # Assign value to matrix position (i,j)
            row = matrix[i].numpy()
            row[j] = value
            matrix[i] = Tensor(row, dtype=dtypes.uint32)
    
    # Multiply by n_inv (modular multiplicative inverse of n)
    n_inv = pow(n, prime - 2, prime)
    matrix = (matrix * n_inv) % prime
    
    return matrix

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