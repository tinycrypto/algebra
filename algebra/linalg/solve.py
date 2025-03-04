from tinygrad import Tensor


def lud(A):
  n = A.shape[0]
  L = Tensor.eye(n, dtype=A.dtype).contiguous()
  U = Tensor.zeros((n, n), dtype=A.dtype).contiguous()

  for k in range(n):
    U[k, k:] = A[k, k:] - L[k, :k] @ U[:k, k:]
    if k < n - 1:
      L[k + 1 :, k] = (A[k + 1 :, k] - L[k + 1 :, :k] @ U[:k, k]) / U[k, k]

  return L, U


def itriu(U):
  n = U.shape[0]
  U_inv = Tensor.zeros((n, n), dtype=U.dtype).contiguous()
  I = Tensor.eye(n, dtype=U.dtype)
  diag = U[Tensor.arange(n), Tensor.arange(n)]
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
  diag = L[Tensor.arange(n), Tensor.arange(n)]
  for i in range(n):
    if i == 0:
      L_inv[i] = I[i] / diag[i]
    else:
      L_inv[i] = (I[i] - L[i, :i] @ L_inv[:i]) / diag[i]
  return L_inv

