from tinygrad import Tensor
import numpy as np
import time
from scipy import linalg  # Import scipy.linalg for built-in LU decomposition

def lu_decomposition(A):
    n = A.shape[0]
    L = Tensor.eye(n).contiguous()
    U = Tensor.zeros((n, n)).contiguous()
    
    for k in range(n):
        U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
        L[k+1:, k] = (A[k+1:, k] - L[k+1:, :k] @ U[:k, k]) / U[k, k] if k < n-1 else L[k+1:, k]
        
    return L, U

def lu_decomposition_scipy(A):
    P, L, U = linalg.lu(A)
    
    return L, U

def benchmark(sizes=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]):
    print("Benchmarking LU decomposition: tinygrad vs SciPy")
    print("-" * 60)
    print(f"{'Size':<10}{'tinygrad (ms)':<20}{'SciPy (ms)':<20}{'Ratio':<10}")
    print("-" * 60)
    
    for n in sizes:
        # Create random matrices
        A_np = np.random.randint(0, 1 << 16, size=(n, n))
        A_tiny = Tensor(A_np)
        
        # Benchmark tinygrad
        start = time.time()
        L_tiny, U_tiny = lu_decomposition(A_tiny)
        tinygrad_time = (time.time() - start) * 1000  # Convert to ms
        
        # Benchmark SciPy built-in implementation
        start = time.time()
        L_scipy, U_scipy = lu_decomposition_scipy(A_np)
        scipy_time = (time.time() - start) * 1000  # Convert to ms
        
        # Calculate ratio (higher means SciPy is faster)
        ratio = tinygrad_time / scipy_time if scipy_time > 0 else float('inf')
        
        print(f"{n:<10}{tinygrad_time:.2f}ms{'':<12}{scipy_time:.2f}ms{'':<12}{ratio:.2f}x")
        
        # Verify correctness for small matrices
        if n <= 50:
            prod_tiny = (L_tiny @ U_tiny).numpy()
            prod_scipy = L_scipy @ U_scipy
            tiny_error = np.max(np.abs(A_np - prod_tiny))
            scipy_error = np.max(np.abs(A_np - prod_scipy))
            print(f"  Max error - tinygrad: {tiny_error:.2e}, SciPy: {scipy_error:.2e}")
    
    print("-" * 60)

if __name__ == "__main__":
    # Basic test
    A = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    L, U = lu_decomposition(A)
    prod = L @ U
    assert (A == prod).all().numpy()
    
    # Run benchmarks
    benchmark()

