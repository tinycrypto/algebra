import random
import numpy as np
from sympy import isprime
from tinygrad.tensor import Tensor

# Default parameters
DEFAULT_N = 8  # Size of transform (power of 2)
DEFAULT_M = 17  # Prime modulus where (M-1) is divisible by N
DEFAULT_W = None  # Will be calculated on first use

# Utility functions
def is_integer(M):
    return isinstance(M, int)

def is_prime(M):
    assert is_integer(M), 'Not an integer.'
    return isprime(M)

# Modular arithmetic functions
def mod_exponent(base, power, M):
    """
    Standard implementation of modular exponentiation for single values
    """
    result = 1
    power = int(power)
    base = base % M
    while power > 0:
        if power & 1:
            result = (result * base) % M
        base = (base * base) % M
        power = power >> 1
    return result

def vector_mod_exponent(bases, power, M):
    """
    Vectorized implementation using tinygrad
    For an array of bases raised to the same power
    """
    # Convert inputs to Tensor
    if not isinstance(bases, Tensor):
        bases_tensor = Tensor(bases)
    else:
        bases_tensor = bases
        
    result = Tensor.ones(*bases_tensor.shape)
    power_val = int(power)
    bases_tensor = bases_tensor % M
    
    while power_val > 0:
        if power_val & 1:
            result = (result * bases_tensor) % M
        bases_tensor = (bases_tensor * bases_tensor) % M
        power_val = power_val >> 1
        
    return result

def mod_inv(x, M):
    """
    Extended Euclidean Algorithm for modular inverse
    """
    t, new_t, r, new_r = 0, 1, M, x

    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, (t - quotient * new_t)
        r, new_r = new_r, (r % new_r)
        
    if r > 1:
        return "x is not invertible."
    if t < 0:
        t = t + M
    return t

def exist_small_order(r, M, N):
    """
    Check if r is a primitive Nth root of unity
    """
    for k in range(2, N):
        if mod_exponent(r, k, M) == 1:
            return True
    return False

def get_nth_root_of_unity(M=DEFAULT_M, N=DEFAULT_N):
    """
    Generate primitive nth root of unity
    """
    assert is_prime(M), 'Not a prime.'  # modulus should be a prime
    assert (M - 1) % N == 0, 'N cannot divide phi(M)'
    phi_M = M - 1
    
    while True:
        alpha = random.randrange(1, M)
        beta = mod_exponent(alpha, phi_M // N, M)
        # check if beta can be k th root of unity, k<N
        if not exist_small_order(beta, M, N):
            return int(beta)

def is_nth_root_of_unity(beta, N=DEFAULT_N, M=DEFAULT_M):
    """
    Verify B^N = 1 (mod M)
    """
    return mod_exponent(beta, N, M) == 1

def bit_reverse(num, length):
    """
    Bit reverse using binary string manipulation
    """
    return int(bin(num)[2:].zfill(length)[::-1], 2)

def order_reverse(poly, N_bit):
    """
    Vectorized bit reversal permutation
    """
    # Convert input to Tensor if it's not already
    if not isinstance(poly, Tensor):
        poly = Tensor(poly)
        
    n = poly.shape[0]
    
    # Create indices and their bit-reversed counterparts
    indices = np.arange(n)
    bit_reversed = np.array([bit_reverse(i, N_bit) for i in indices])
    
    # Create a new tensor with bit-reversed order
    poly_np = poly.numpy()
    result = Tensor(poly_np[bit_reversed])
    
    return result

# Main NTT functions
def ntt(points, N=DEFAULT_N, M=DEFAULT_M, w=DEFAULT_W):
    """
    Number theoretic transform algorithm using tinygrad
    """
    global DEFAULT_W
    
    # If w is not provided, calculate it
    if w is None:
        if DEFAULT_W is None:
            DEFAULT_W = get_nth_root_of_unity(M, N)
        w = DEFAULT_W
    
    # Convert input to Tensor if it's not already
    if not isinstance(points, Tensor):
        points = Tensor(points)
        
    N_bit = N.bit_length() - 1
    
    # Perform bit-reversal permutation
    indices = np.arange(N)
    bit_reversed = np.array([bit_reverse(i, N_bit) for i in indices])
    
    # Create a new tensor with bit-reversed order
    points_np = points.numpy()
    bit_reversed_points = Tensor(points_np[bit_reversed])
    
    poly_tensor = bit_reversed_points
    
    for i in range(N_bit):
        # Create vectors for even and odd indices
        even_indices = np.arange(0, N, 2)
        odd_indices = np.arange(1, N, 2)
        
        # Get current polynomial coefficients as numpy array for indexing
        poly_np = poly_tensor.numpy()
        
        # Create powers for twiddle factors
        shift_bits = N_bit - 1 - i
        P_values = (np.arange(N//2) >> shift_bits) << shift_bits
        
        # Compute twiddle factors
        w_P = np.array([mod_exponent(w, P, M) for P in P_values])
        
        # Extract even and odd elements
        even = poly_np[even_indices]
        odd = poly_np[odd_indices]
        
        # Apply twiddle factors to odd elements
        odd_twisted = (odd * w_P) % M
        
        # Compute DFT results
        points1 = (even + odd_twisted) % M
        points2 = (even - odd_twisted) % M
        
        # Combine results
        poly_np = np.concatenate([points1, points2])
        poly_tensor = Tensor(poly_np)
        
    return poly_tensor

def intt(points, N=DEFAULT_N, M=DEFAULT_M, w=DEFAULT_W):
    """
    Inverse number theoretic transform algorithm
    """
    global DEFAULT_W
    
    # If w is not provided, calculate it
    if w is None:
        if DEFAULT_W is None:
            DEFAULT_W = get_nth_root_of_unity(M, N)
        w = DEFAULT_W
    
    # Convert input to Tensor if it's not already
    if not isinstance(points, Tensor):
        points = Tensor(points)
        
    inv_w = mod_inv(w, M)
    inv_N = mod_inv(N, M)
    
    # Perform normal NTT with inverse of w
    poly = ntt(points, N, M, inv_w)
    
    # Apply scaling factor
    poly_np = poly.numpy()
    poly_np = (poly_np * inv_N) % M
    
    return Tensor(poly_np)

# Polynomial operations
def polynomial_multiply(poly1, poly2, N=DEFAULT_N, M=DEFAULT_M, w=DEFAULT_W):
    """
    Multiply two polynomials using NTT
    """
    # Check if inputs are Tensors
    if not isinstance(poly1, Tensor):
        poly1 = Tensor(poly1)
    if not isinstance(poly2, Tensor):
        poly2 = Tensor(poly2)
    
    # Transform to frequency domain
    freq1 = ntt(poly1, N, M, w)
    freq2 = ntt(poly2, N, M, w)
    
    # Multiply in frequency domain (element-wise)
    freq_prod = (freq1 * freq2) % M
    
    # Transform back to time domain
    poly_prod = intt(freq_prod, N, M, w)
    
    return poly_prod

# Example usage function
def example():
    # Verify our parameters
    if not is_prime(DEFAULT_M) or (DEFAULT_M-1) % DEFAULT_N != 0:
        print(f"Invalid parameters: M={DEFAULT_M} must be prime and (M-1) must be divisible by N={DEFAULT_N}")
        return
    
    # Generate a primitive Nth root of unity
    w = get_nth_root_of_unity(DEFAULT_M, DEFAULT_N)
    print(f"Primitive {DEFAULT_N}th root of unity modulo {DEFAULT_M}: {w}")
    
    # Create a sample polynomial (coefficients)
    poly = Tensor([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Original polynomial: {poly.numpy()}")
    
    # Perform forward NTT
    freq_domain = ntt(poly)
    print(f"After NTT: {freq_domain.numpy()}")
    
    # Perform inverse NTT
    time_domain = intt(freq_domain)
    print(f"After INTT (should match original): {time_domain.numpy()}")
    
    # Verify the transformation is correct
    if np.allclose(poly.numpy(), time_domain.numpy()):
        print("✓ NTT transformation is correct!")
    else:
        print("✗ NTT transformation failed!")
    
    # Example of polynomial multiplication using NTT
    poly1 = Tensor([1, 2, 3, 4, 0, 0, 0, 0])  # 4x^3 + 3x^2 + 2x + 1
    poly2 = Tensor([5, 6, 7, 0, 0, 0, 0, 0])  # 7x^2 + 6x + 5
    
    print(f"\nPolynomial 1: {poly1.numpy()}")
    print(f"Polynomial 2: {poly2.numpy()}")
    
    # Multiply polynomials using our function
    poly_prod = polynomial_multiply(poly1, poly2)
    print(f"Polynomial product: {poly_prod.numpy()}")
    
    # Let's calculate the expected result by direct convolution
    expected = np.zeros(8, dtype=int)
    for i in range(4):  # poly1 has 4 non-zero terms
        for j in range(3):  # poly2 has 3 non-zero terms
            expected[i+j] = (expected[i+j] + poly1.numpy()[i] * poly2.numpy()[j]) % DEFAULT_M
    
    print(f"Expected (direct convolution): {expected}")
    
    # Check if our computed result matches the expected polynomial product
    if np.allclose(poly_prod.numpy(), expected):
        print("✓ Polynomial multiplication using NTT is correct!")
    else:
        print("✗ Polynomial multiplication using NTT failed!")

if __name__ == "__main__":
    example()