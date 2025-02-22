import numpy as np
from tinygrad.tensor import Tensor
from algebra.ff.prime_field import PrimeField as PF


class Polynomial:
  PrimeField = None

  """
    A univariate polynomial over a prime field.
    The polynomial is represented in coefficient form:
        p(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d
    where each coefficient ai is an instance of a PrimeField element.
    """

  def __init__(self, coeffs: list[int] | Tensor, prime_field: PF):
    """
    Initialize the polynomial.

    coeffs: a list of field elements (instances of a PrimeField subclass)
    """
    self.PrimeField = prime_field
    if isinstance(coeffs, list):
      self.coeffs = Tensor(np.array(coeffs, dtype=np.int32), requires_grad=False)
    elif isinstance(coeffs, Tensor):
      self.coeffs = coeffs

  def degree(self) -> int:
    """
    Return the degree of the polynomial.
    By convention, the zero polynomial has degree 0.
    """
    return max(self.coeffs.size(dim=0) - 1, 0)

  def evaluate(self, x: int):
    """
    Evaluate the polynomial at the field element x using Horner's method.

    x: a field element (instance of the same PrimeField subclass)
    """
    if self.coeffs.shape[0] == 0:
      return x * 0
    result = self.coeffs[0] * 0
    for coeff in self.coeffs[::-1]:
      result = result * x + coeff
    return result

  def __add__(self, other: "Polynomial"):
    """
    Add two polynomials.
    """
    max_len = max(self.coeffs.size(dim=0), other.coeffs.size(dim=0))
    self_padded = self.coeffs.pad((0, max_len - self.coeffs.size(dim=0)), mode="constant", value=0)
    other_padded = other.coeffs.pad((0, max_len - other.coeffs.size(dim=0)), mode="constant", value=0)
    new_coeffs = self_padded + other_padded
    return Polynomial(new_coeffs, self.PrimeField)

  def __sub__(self, other):
    """
    Subtract another polynomial from this polynomial.
    """
    max_len = max(self.coeffs.size(dim=0), other.coeffs.size(dim=0))
    self_padded = self.coeffs.pad((0, max_len - self.coeffs.size(dim=0)), mode="constant", value=0)
    other_padded = other.coeffs.pad((0, max_len - other.coeffs.size(dim=0)), mode="constant", value=0)
    new_coeffs = (self_padded - other_padded).mod(Tensor([self.PrimeField.P]))
    return Polynomial(new_coeffs, self.PrimeField)

  def __neg__(self):
    """
    Negate the polynomial.
    """
    return Polynomial([-c for c in self.coeffs], self.PrimeField)

  def __mul__(self, other):
    """
    Multiply by another polynomial or by a scalar.
    """
    # If 'other' is a Polynomial, perform polynomial multiplication (convolution)
    if isinstance(other, Polynomial):
      # TODO: wip
      return Polynomial(other.coeffs, self.PrimeField)
    else:
      new_coeffs = (self.coeffs.mul(Tensor([other]))).mod(Tensor([self.PrimeField.P]))
      return Polynomial(new_coeffs, self.PrimeField)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __repr__(self):
    if not self.coeffs:
      return "0"
    terms = []
    for i, coeff in enumerate(self.coeffs):
      if i == 0:
        terms.append(f"{coeff}")
      elif i == 1:
        terms.append(f"({coeff})*x")
      else:
        terms.append(f"({coeff})*x^{i}")
    return " + ".join(terms)
