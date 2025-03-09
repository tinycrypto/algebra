from tinygrad import Tensor


def lud(A):
  n = A.shape[0]
  L = Tensor.eye(n, dtype=A.dtype).contiguous()
  U = Tensor.zeros((n, n), dtype=A.dtype).contiguous()
  for k in range(n):
    U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
    if k < n - 1:
      L[k + 1 :, k] = (A[k + 1 :, k] - L[k + 1 :, :k] @ U[:k, k]) / U[k, k]
  idx = Tensor.arange(n)
  if (U[idx, idx].abs() < 1e-10).any().numpy():
    raise ValueError("Matrix is singular, cannot compute LU decomposition")
  return L, U


def itriu(U):
  n = U.shape[0]
  U_inv = Tensor.zeros((n, n), dtype=U.dtype).contiguous()
  I = Tensor.eye(n, dtype=U.dtype)
  idx = Tensor.arange(n)
  diag = U[idx, idx]
  for i in range(n - 1, -1, -1):
    if i == n - 1:
      U_inv[i] = I[i] / diag[i]
    else:
      U_inv[i] = (I[i] - U[i, i + 1 :] @ U_inv[i + 1 :]) / diag[i]
  return U_inv


def itril(L):
  n = L.shape[0]
  L_inv = Tensor.zeros((n, n), dtype=L.dtype).contiguous()
  I = Tensor.eye(n, dtype=L.dtype)
  idx = Tensor.arange(n)
  diag = L[idx, idx]
  for i in range(n):
    if i == 0:
      L_inv[i] = I[i] / diag[i]
    else:
      L_inv[i] = (I[i] - L[i, :i] @ L_inv[:i]) / diag[i]
  return L_inv


def matrix_inverse(A: Tensor) -> Tensor:
  if A.shape[0] != A.shape[1]:
    raise ValueError("Matrix must be square")

  L, U = lud(A)
  U_inv = itriu(U)
  L_inv = itril(L)
  A_inv = U_inv @ L_inv

  return A_inv


def solve(A: Tensor, b: Tensor) -> Tensor:
  return matrix_inverse(A) @ b
