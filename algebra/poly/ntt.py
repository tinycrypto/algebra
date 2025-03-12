import random
import numpy as np
from sympy import isprime
from tinygrad.tensor import Tensor

class NTT:
    def isInteger(self, M):
        return isinstance(M, int)

    def isPrime(self, M):
        assert self.isInteger(M), 'Not an integer.'
        return isprime(M)

    # modular exponential algorithm
    # complexity is O(log N)
    def modExponent(self, base, power, M):
        """
        Standard implementation for single values
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
    
    # Vectorized version of modular exponentiation
    def vector_modExponent(self, bases, power, M):
        """
        Vectorized implementation using tinygrad
        For an array of bases raised to the same power
        
        Parameters:
        bases (array-like): Array of base values
        power (int): Power to raise each base to
        M (int): Modulus
        
        Returns:
        Tensor: Result of modular exponentiation for each base
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

    # calculate x^(-1) mod M
    def modInv(self, x, M):
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

    # check if r^k = 1 (mod M), k<N
    def existSmallN(self, r, M, N):
        """
        Check if r is a primitive Nth root of unity
        
        Parameters:
        r (int): Value to check
        M (int): Modulus
        N (int): Upper bound for checking powers
        
        Returns:
        bool: True if r^k = 1 (mod M) for any k < N
        """
        # For small N, we can just check directly without vectorization
        for k in range(2, N):
            if self.modExponent(r, k, M) == 1:
                return True
        return False

    # generate primitive nth root of unity
    def NthRootOfUnity(self, M, N):
        assert self.isPrime(M), 'Not a prime.'  # modulus should be a prime
        assert (M - 1) % N == 0, 'N cannot divide phi(M)'
        phi_M = M - 1
        
        while True:
            alpha = random.randrange(1, M)
            beta = self.modExponent(alpha, phi_M // N, M)
            # check if beta can be k th root of unity, k<N
            if not self.existSmallN(beta, M, N):
                return int(beta)

    # verify B^N = 1 (mod M)
    def isNthRootOfUnity(self, M, N, beta):
        return self.modExponent(beta, N, M) == 1

    def bitReverse(self, num, length):
        """
        Efficient bit reverse using numpy
        """
        # Convert to binary string, pad, reverse, and convert back to int
        return int(bin(num)[2:].zfill(length)[::-1], 2)
    
    def orderReverse(self, poly, N_bit):
        """
        Vectorized bit reversal permutation
        
        Parameters:
        poly (Tensor or list): Input polynomial coefficients
        N_bit (int): Bit length for the reversal
        
        Returns:
        Tensor: Bit-reversed polynomial
        """
        # Convert input to Tensor if it's not already
        if not isinstance(poly, Tensor):
            poly = Tensor(poly)
            
        n = poly.shape[0]
        
        # Create indices and their bit-reversed counterparts
        indices = np.arange(n)
        bit_reversed = np.array([self.bitReverse(i, N_bit) for i in indices])
        
        # Create a new tensor with bit-reversed order
        poly_np = poly.numpy()
        result = Tensor(poly_np[bit_reversed])
        
        return result

    # NTT implementation using tinygrad vectorization
    def ntt(self, poly, M, N, w):
        """
        Number theoretic transform algorithm using tinygrad
        
        Parameters:
        poly (Tensor or list): Input polynomial coefficients
        M (int): Modulus for the NTT
        N (int): Size of the transform (power of 2)
        w (int): Nth root of unity modulo M
        
        Returns:
        Tensor: Result of the NTT transform
        """
        # Convert input to Tensor if it's not already
        if not isinstance(poly, Tensor):
            poly = Tensor(poly)
            
        N_bit = N.bit_length() - 1
        
        # Perform bit-reversal permutation
        indices = np.arange(N)
        bit_reversed = np.array([self.bitReverse(i, N_bit) for i in indices])
        
        # Create a new tensor with bit-reversed order
        poly_np = poly.numpy()
        bit_reversed_poly = Tensor(poly_np[bit_reversed])
        
        poly_tensor = bit_reversed_poly
        
        for i in range(N_bit):
            # Create vectors for even and odd indices
            even_indices = np.arange(0, N, 2)
            odd_indices = np.arange(1, N, 2)
            
            # Get current polynomial coefficients as numpy array for indexing
            poly_np = poly_tensor.numpy()
            
            # Create powers for twiddle factors
            shift_bits = N_bit - 1 - i
            P_values = (np.arange(N//2) >> shift_bits) << shift_bits
            
            # Compute twiddle factors - use numpy directly for simplicity
            w_P = np.array([self.modExponent(w, P, M) for P in P_values])
            
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

    # Inverse NTT implementation
    def intt(self, points, M, N, w):
        """
        Inverse number theoretic transform algorithm
        
        Parameters:
        points (Tensor or list): Input points from the frequency domain
        M (int): Modulus for the NTT
        N (int): Size of the transform (power of 2)
        w (int): Nth root of unity modulo M
        
        Returns:
        Tensor: Result of the inverse NTT transform
        """
        # Convert input to Tensor if it's not already
        if not isinstance(points, Tensor):
            points = Tensor(points)
            
        inv_w = self.modInv(w, M)
        inv_N = self.modInv(N, M)
        
        # Perform normal NTT with inverse of w
        poly = self.ntt(points, M, N, inv_w)
        
        # Apply scaling factor - using numpy for simplicity
        poly_np = poly.numpy()
        poly_np = (poly_np * inv_N) % M
        
        return Tensor(poly_np)

def main():
    # Initialize the NTT class
    ntt = NTT()
    
    # Choose parameters for NTT
    # We need a prime modulus M such that (M-1) is divisible by N
    N = 8  # Power of 2
    M = 17  # A prime where (M-1) is divisible by N
    
    # Verify our parameters
    if not ntt.isPrime(M) or (M-1) % N != 0:
        print(f"Invalid parameters: M={M} must be prime and (M-1) must be divisible by N={N}")
        return
    
    # Generate a primitive Nth root of unity
    w = ntt.NthRootOfUnity(M, N)
    print(f"Primitive {N}th root of unity modulo {M}: {w}")
    
    # Create a sample polynomial (coefficients)
    poly = Tensor([3, 1, 4, 1, 5, 9, 2, 6])
    print(f"Original polynomial: {poly.numpy()}")
    
    # Perform forward NTT
    freq_domain = ntt.ntt(poly, M, N, w)
    print(f"After NTT: {freq_domain.numpy()}")
    
    # Perform inverse NTT
    time_domain = ntt.intt(freq_domain, M, N, w)
    print(f"After INTT (should match original): {time_domain.numpy()}")
    
    # Verify the transformation is correct
    if np.allclose(poly.numpy(), time_domain.numpy()):
        print("✓ NTT transformation is correct!")
    else:
        print("✗ NTT transformation failed!")
    
    # Example of polynomial multiplication using NTT
    # (a*b = INTT(NTT(a) * NTT(b)))
    poly1 = Tensor([1, 2, 3, 4, 0, 0, 0, 0])  # 4x^3 + 3x^2 + 2x + 1
    poly2 = Tensor([5, 6, 7, 0, 0, 0, 0, 0])  # 7x^2 + 6x + 5
    
    print(f"\nPolynomial 1: {poly1.numpy()}")
    print(f"Polynomial 2: {poly2.numpy()}")
    
    # Transform to frequency domain
    freq1 = ntt.ntt(poly1, M, N, w)
    freq2 = ntt.ntt(poly2, M, N, w)
    
    # Multiply in frequency domain (element-wise)
    freq_prod = (freq1 * freq2) % M
    
    # Transform back to time domain
    poly_prod = ntt.intt(freq_prod, M, N, w)
    print(f"Polynomial product: {poly_prod.numpy()}")
    
    # Let's calculate the expected result by direct convolution
    expected = np.zeros(8, dtype=int)
    for i in range(4):  # poly1 has 4 non-zero terms
        for j in range(3):  # poly2 has 3 non-zero terms
            expected[i+j] = (expected[i+j] + poly1.numpy()[i] * poly2.numpy()[j]) % M
    
    expected_tensor = Tensor(expected)
    print(f"Expected (direct convolution): {expected}")
    
    # Check if our computed result matches the expected polynomial product
    if np.allclose(poly_prod.numpy(), expected_tensor.numpy()):
        print("✓ Polynomial multiplication using NTT is correct!")
    else:
        print("✗ Polynomial multiplication using NTT failed!")
        print(f"Expected: {expected}")
        print(f"Got:      {poly_prod.numpy()}")

if __name__ == "__main__":
    main()