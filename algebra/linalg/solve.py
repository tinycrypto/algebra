from tinygrad import Tensor


def lu_decomposition(A):
  n = A.shape[0]
  L = Tensor.eye(n, dtype=A.dtype).contiguous()
  U = Tensor.zeros((n, n), dtype=A.dtype).contiguous()

  for k in range(n):
    U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
    if k < n - 1:
      L[k + 1 :, k] = (A[k + 1 :, k] - L[k + 1 :, :k] @ U[:k, k]) / U[k, k]

  return L, U
