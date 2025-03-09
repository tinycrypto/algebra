import pytest
import numpy as np
from tinygrad import Tensor
from algebra.linalg.solve import lud, itriu, itril, matrix_inverse, solve


def test_lud():
  A = Tensor([[4.0, 7.0], [2.0, 6.0]])
  L, U = lud(A)

  LU = L @ U
  assert np.allclose(LU.numpy(), A.numpy(), rtol=1e-5)

  L_np, U_np = L.numpy(), U.numpy()
  assert np.allclose(np.tril(L_np), L_np)
  assert np.allclose(np.diag(L_np), 1.0)
  assert np.allclose(np.triu(U_np), U_np)


def test_lud_singular():
  A = Tensor([[1.0, 2.0], [2.0, 4.0]])
  with pytest.raises(ValueError):
    L, U = lud(A)
    print(f"L: {L.numpy()}")
    print(f"U: {U.numpy()}")


def test_itriu():
  U = Tensor([[2.0, 3.0], [0.0, 5.0]])
  U_inv = itriu(U)
  I = U @ U_inv
  assert np.allclose(I.numpy(), np.eye(2), rtol=1e-5)


def test_itril():
  L = Tensor([[1.0, 0.0], [2.0, 1.0]])
  L_inv = itril(L)
  I = L @ L_inv
  assert np.allclose(I.numpy(), np.eye(2), rtol=1e-5)


def test_matrix_inverse():
  A = Tensor([[4.0, 7.0], [2.0, 6.0]])
  A_inv = matrix_inverse(A)
  I = A @ A_inv
  assert np.allclose(I.numpy(), np.eye(2), rtol=1e-5)

  expected_inv = np.array([[0.6, -0.7], [-0.2, 0.4]])
  assert np.allclose(A_inv.numpy(), expected_inv, rtol=1e-5)


def test_matrix_inverse_3x3():
  A = Tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]])
  A_inv = matrix_inverse(A)
  I = A @ A_inv
  assert np.allclose(I.numpy(), np.eye(3), rtol=1e-5)


def test_matrix_inverse_singular():
  A = Tensor([[1.0, 2.0], [2.0, 4.0]])
  with pytest.raises(ValueError):
    matrix_inverse(A)


def test_matrix_inverse_non_square():
  A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  with pytest.raises(ValueError):
    matrix_inverse(A)


def test_solve():
  # Test with a simple 2x2 system
  A = Tensor([[4.0, 7.0], [2.0, 6.0]])
  b = Tensor([11.0, 8.0])
  x = solve(A, b)
  print(f"x: {x.numpy()}")

  # Check that A @ x ≈ b
  result = A @ x
  assert np.allclose(result.numpy(), b.numpy(), rtol=1e-5)

  # Check against known solution: x = [1.0, 1.0]
  expected = np.array([1.0, 1.0])
  assert np.allclose(x.numpy(), expected, rtol=1e-5)

  # Test with a 3x3 system
  A = Tensor([[3.0, 1.0, 2.0], [2.0, 6.0, -1.0], [1.0, 0.0, 4.0]])
  b = Tensor([10.0, 1.0, 5.0])
  x = solve(A, b)

  # Check that A @ x ≈ b
  result = A @ x
  assert np.allclose(result.numpy(), b.numpy(), rtol=1e-5)
