from tinygrad import Tensor
import numpy as np
import time

def lu_decomposition(A):
    """Perform LU decomposition using the Doolittle factorisation with vectorized operations."""
    
    n = A.shape[0]
    L = Tensor.zeros((n, n)).contiguous()
    U = L.clone()
    
    for k in range(n):
        L[k, k] = 1
        U[k, k] = (A[k, k] - (L[k, :k] @ U[:k, k])) / L[k, k]
        if k+1 < n:
            U[k, k+1:] = (A[k, k+1:] - (L[k, :k] @ U[:k, k+1:])) / L[k, k]
        if k+1 < n:
            L[k+1:, k] = (A[k+1:, k] - (L[k+1:, :k] @ U[:k, k])) / U[k, k]
            
    return L, U

def lu_decomposition_numpy(A):
    """Perform LU decomposition using the Doolittle factorisation in NumPy."""
    A_np = A.numpy() if hasattr(A, 'numpy') else A
    
    L = np.zeros_like(A_np)
    U = np.zeros_like(A_np)
    N = A_np.shape[0]

    for k in range(N):
        L[k, k] = 1
        U[k, k] = (A_np[k, k] - np.dot(L[k, :k], U[:k, k])) / L[k, k]
        for j in range(k+1, N):
            U[k, j] = (A_np[k, j] - np.dot(L[k, :k], U[:k, j])) / L[k, k]
        for i in range(k+1, N):
            L[i, k] = (A_np[i, k] - np.dot(L[i, :k], U[:k, k])) / U[k, k]

    return L, U

def benchmark(sizes=[3, 10, 50, 100, 200]):
    """Benchmark tinygrad vs NumPy LU decomposition for various matrix sizes."""
    print("Benchmarking LU decomposition: tinygrad vs NumPy")
    print("-" * 60)
    print(f"{'Size':<10}{'tinygrad (ms)':<20}{'NumPy (ms)':<20}{'Ratio':<10}")
    print("-" * 60)
    
    for n in sizes:
        # Create random matrices
        A_np = np.random.randint(0, 100, size=(n, n))
        A_tiny = Tensor(A_np)

        print(A_tiny.numpy())
        
        # Benchmark tinygrad
        start = time.time()
        L_tiny, U_tiny = lu_decomposition(A_tiny)
        tinygrad_time = (time.time() - start) * 1000  # Convert to ms
        
        # Benchmark NumPy
        start = time.time()
        L_np, U_np = lu_decomposition_numpy(A_np)
        numpy_time = (time.time() - start) * 1000  # Convert to ms
        
        # Calculate ratio (higher means NumPy is faster)
        ratio = tinygrad_time / numpy_time if numpy_time > 0 else float('inf')
        
        print(f"{n:<10}{tinygrad_time:.2f}ms{'':<12}{numpy_time:.2f}ms{'':<12}{ratio:.2f}x")
        
        # Verify correctness for small matrices
        if n <= 50:
            prod_tiny = (L_tiny @ U_tiny).numpy()
            prod_np = L_np @ U_np
            tiny_error = np.max(np.abs(A_np - prod_tiny))
            np_error = np.max(np.abs(A_np - prod_np))
            print(f"  Max error - tinygrad: {tiny_error:.2e}, NumPy: {np_error:.2e}")
    
    print("-" * 60)

if __name__ == "__main__":
    # Basic test
    A = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    L, U = lu_decomposition(A)
    prod = L @ U
    assert (A == prod).all().numpy()
    
    # Run benchmarks
    benchmark()

