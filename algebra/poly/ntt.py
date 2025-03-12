from tinygrad.tensor import Tensor
from tinygrad import dtypes
import numpy as np


def _next_valid_n(l, p):
  """Find next n >= l that divides p-1 efficiently"""
  p_minus_1 = p - 1
  for n in range(l, min(p, 2 * l)):
    if p_minus_1 % n == 0:
      return n

  bit_length = l.bit_length()
  power_of_two = 1 << bit_length
  if power_of_two < l:
    power_of_two = power_of_two << 1

  while power_of_two < p:
    if p_minus_1 % power_of_two == 0:
      return power_of_two
    power_of_two = power_of_two << 1

  for divisor in range(l, int(p_minus_1**0.5) + 1):
    if p_minus_1 % divisor == 0:
      quotient = p_minus_1 // divisor
      if quotient >= l:
        return quotient
      return divisor

  return p - 1


def ntt(polynomial: Tensor, prime: int, primitive_root: int) -> Tensor:
  """
  Compute the Number Theoretic Transform of a polynomial.

  Args:
      polynomial: Coefficient vector of the polynomial (Tensor or list/array)
      prime: The prime modulus
      primitive_root: A primitive root modulo prime

  Returns:
      The NTT of the polynomial as a Tensor
  """
  return _transform(polynomial, prime, primitive_root, inverse=False)


def intt(transformed: Tensor, prime: int, primitive_root: int) -> Tensor:
  """
  Compute the Inverse Number Theoretic Transform.

  Args:
      transformed: The transformed polynomial (Tensor or list/array)
      prime: The prime modulus
      primitive_root: A primitive root modulo prime

  Returns:
      The original polynomial coefficients as a Tensor
  """
  return _transform(transformed, prime, primitive_root, inverse=True)


def _transform(x, prime: int, primitive_root: int, inverse: bool = False) -> Tensor:
  """Internal function to perform NTT or INTT transformation"""
  if not isinstance(x, Tensor):
    x = Tensor(x, dtype=dtypes.uint64)

  dtype = x.dtype
  n = _next_valid_n(len(x), prime)

  if len(x) < n:
    padded_np = Tensor.zeros(n, dtype=dtype).contiguous()
    padded_np[: len(x)] = x
    x = padded_np

  # Create powers matrix (i*j) % n
  i = Tensor.arange(n, dtype=dtypes.uint64).reshape(n, 1)
  j = Tensor.arange(n, dtype=dtypes.uint64).reshape(1, n)
  powers = (i * j) % n

  # Get omega (or inverse omega for INTT)
  omega = pow(primitive_root, (prime - 1) // n, prime)
  if inverse:
    omega = pow(omega, prime - 2, prime)  # Modular inverse

  # Compute all powers of omega
  omega_powers = np.ones(n, dtype=np.uint64)
  current = 1
  for k in range(n):
    omega_powers[k] = current
    current = (current * omega) % prime

  # Create transformation matrix
  matrix = Tensor(omega_powers, dtype=dtypes.uint64)[powers]

  # For inverse, apply scaling factor (1/n mod prime)
  if inverse:
    n_inv = pow(n, prime - 2, prime)
    matrix = (matrix * n_inv) % prime

  # Perform the transformation
  return (matrix @ (x % prime)) % prime


if __name__ == "__main__":
  from algebra.ff.m31 import M31

  n, prime, primitive_root = 8, M31.P, M31.w
  polynomial = Tensor([1, 2, 3], dtype=dtypes.uint64)

  transformed = ntt(polynomial, prime, primitive_root)
  print("NTT result:", transformed.numpy())

  original = intt(transformed, prime, primitive_root)
  print("INTT result:", original.numpy())

  assert (polynomial.numpy() == original.numpy()[: len(polynomial)]).all(), "INTT(NTT(polynomial)) != polynomial"
  print("Verification successful")
