from tinygrad.tensor import Tensor
from tinygrad import dtypes
from algebra.ff.prime_field import PrimeField as PF
from algebra.poly.ntt import ntt, intt
import numpy as np


class Polynomial:
  PrimeField = None

  """
    A univariate polynomial over a prime field.
    The polynomial is represented in coefficient form:
        p(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d
    where each coefficient ai is an instance of a PrimeField element.
    """

  def __init__(self, coeffs: list[int] | Tensor, prime_field: PF = None):
    """Initialize polynomial with coefficients and optional prime field."""
    self.PrimeField = prime_field
    if isinstance(coeffs, list):
      coeffs = np.trim_zeros(coeffs, "b")
      self.coeffs = Tensor(coeffs, dtype=dtypes.int32)
    else:
      coeffs_np = coeffs.numpy()
      coeffs_np = np.trim_zeros(coeffs_np, "b")
      self.coeffs = Tensor(coeffs_np, dtype=coeffs.dtype)

  def degree(self) -> int:
    """
    Return the degree of the polynomial.
    By convention, the zero polynomial has degree 0.
    """
    return max(self.coeffs.size(dim=0) - 1, 0)

  def evaluate(self, x: int | Tensor):
    """
    Evaluate the polynomial at the field element x using Horner's method.

    x: a field element (instance of the same PrimeField subclass)
    """
    if isinstance(x, Tensor):
      return self.__evaluate_all(x)

    if self.coeffs.shape[0] == 0:
      return x * 0
    result = self.coeffs[0] * 0
    for coeff in self.coeffs[::-1]:
      result = result * x + coeff
    return result

  def ntt(self):
    """
    Compute the Number Theoretic Transform of the polynomial.
    """
    p_ntt = ntt(self.coeffs.cast(dtypes.uint64), self.PrimeField.P, self.PrimeField.w).cast(self.coeffs.dtype)
    return Polynomial(p_ntt, self.PrimeField)

  def intt(self):
    """
    Compute the Inverse Number Theoretic Transform of the polynomial.
    """
    p_intt = intt(self.coeffs.cast(dtypes.uint64), self.PrimeField.P, self.PrimeField.w).cast(self.coeffs.dtype)
    return Polynomial(p_intt, self.PrimeField)

  def __evaluate_all(self, xs: Tensor):
    """
    Evaluate the polynomial at all elements in xs using Horner's method.
    """
    if self.coeffs.shape[0] == 0:
      return xs * 0

    results = Tensor.zeros(xs.shape[0], dtype=xs.dtype)

    for coeff in self.coeffs[::-1]:
      results = results * xs + coeff
      if self.PrimeField is not None:
        results = results.mod(Tensor([self.PrimeField.P]))

    return results

  def __add__(self, other: "Polynomial"):
    """
    Add two polynomials.
    """
    max_len = max(self.coeffs.shape[0], other.coeffs.shape[0])
    self_padded = self.coeffs.pad((0, max_len - self.coeffs.shape[0]), mode="constant", value=0)
    other_padded = other.coeffs.pad((0, max_len - other.coeffs.shape[0]), mode="constant", value=0)

    if self.PrimeField is not None:
      new_coeffs = self.PrimeField.add(self_padded, other_padded)
    else:
      new_coeffs = self_padded + other_padded

    return Polynomial(new_coeffs, self.PrimeField)

  def __sub__(self, other):
    """
    Subtract another polynomial from this polynomial.
    """
    max_len = max(self.coeffs.shape[0], other.coeffs.shape[0])
    self_padded = self.coeffs.pad((0, max_len - self.coeffs.shape[0]), mode="constant", value=0)
    other_padded = other.coeffs.pad((0, max_len - other.coeffs.shape[0]), mode="constant", value=0)

    if self.PrimeField is not None:
      new_coeffs = self.PrimeField.sub(self_padded, other_padded)
    else:
      new_coeffs = self_padded - other_padded

    return Polynomial(new_coeffs, self.PrimeField)

  def __neg__(self):
    """
    Negate the polynomial.
    """
    if self.PrimeField is not None:
      new_coeffs = self.PrimeField.neg(self.coeffs)
    else:
      new_coeffs = -self.coeffs

    return Polynomial(new_coeffs, self.PrimeField)

  def __mul__(self, other: Tensor | int):
    """
    Multiply by another polynomial or by a scalar.
    """
    if isinstance(other, Polynomial):
      na, nb = self.coeffs.shape[0], other.coeffs.shape[0]
      result_len = na + nb - 1

      idx = Tensor.arange(result_len, dtype=dtypes.int32).reshape(result_len, 1, 1)
      ai = Tensor.arange(na, dtype=dtypes.int32).reshape(1, na, 1)
      bi = Tensor.arange(nb, dtype=dtypes.int32).reshape(1, 1, nb)

      idx_expanded = idx.expand(result_len, na, nb)
      ai_expanded = ai.expand(result_len, na, nb)
      bi_expanded = bi.expand(result_len, na, nb)

      mask = ai_expanded + bi_expanded == idx_expanded

      a_exp = self.coeffs.reshape(1, na, 1).expand(result_len, na, nb)
      b_exp = other.coeffs.reshape(1, 1, nb).expand(result_len, na, nb)

      products = a_exp * b_exp * mask
      new_coeffs = products.sum(axis=(1, 2))

      if self.PrimeField is not None:
        new_coeffs = new_coeffs.mod(Tensor([self.PrimeField.P]))

      return Polynomial(new_coeffs, self.PrimeField)
    else:
      if self.PrimeField is not None:
        new_coeffs = (self.coeffs.mul(Tensor([other]))).mod(Tensor([self.PrimeField.P]))
      else:
        new_coeffs = self.coeffs.mul(Tensor([other]))
      return Polynomial(new_coeffs, self.PrimeField)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __repr__(self):
    coeffs_list = self.coeffs.numpy().tolist()
    return f"Polynomial({coeffs_list})"

  def __call__(self, x: int | Tensor):
    return self.evaluate(x)
