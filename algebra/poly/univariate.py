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
      if len(coeffs) == 0:
        coeffs = np.array([0])
      self.coeffs = Tensor(coeffs, dtype=dtypes.int32)
    elif isinstance(coeffs, np.ndarray):
      coeffs = np.trim_zeros(coeffs, "b")
      if len(coeffs) == 0:
        coeffs = np.array([0])
      self.coeffs = Tensor(coeffs, dtype=dtypes.int32)
    else:
      coeffs_np = coeffs.numpy()
      coeffs_np = np.trim_zeros(coeffs_np, "b")
      if len(coeffs_np) == 0:
        coeffs_np = np.array([0])
      self.coeffs = Tensor(coeffs_np, dtype=coeffs.dtype)

  def degree(self) -> int:
    """
    Return the degree of the polynomial.
    By convention, the zero polynomial has degree 0.
    """
    return max(self.coeffs.shape[0] - 1, 0)

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

  def reduce(self, modulus=None):
    """
    Reduce the coefficients of the polynomial modulo the prime field.
    """
    if modulus is None:
      if self.PrimeField is not None:
        self.coeffs = self.coeffs % self.PrimeField.P
    else:
      self.coeffs = self.coeffs % modulus
    return self

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

    # Start with zeros of the same shape and dtype as xs
    results = Tensor.zeros_like(xs)

    # Apply Horner's method vectorized
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
      # Cast to int64 to avoid overflow, then cast back
      self_64 = self_padded.cast(dtypes.int64)
      other_64 = other_padded.cast(dtypes.int64)
      new_coeffs = self.PrimeField.add(self_64, other_64).cast(self.coeffs.dtype)
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
      # Cast to int64 to avoid overflow, then cast back
      self_64 = self_padded.cast(dtypes.int64)
      other_64 = other_padded.cast(dtypes.int64)
      new_coeffs = self.PrimeField.sub(self_64, other_64).cast(self.coeffs.dtype)
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
        # Cast to int64 to avoid overflow, then back
        coeffs_64 = self.coeffs.cast(dtypes.int64)
        other_64 = Tensor([other], dtype=dtypes.int64)
        new_coeffs = (coeffs_64 * other_64).mod(Tensor([self.PrimeField.P], dtype=dtypes.int64)).cast(self.coeffs.dtype)
      else:
        new_coeffs = self.coeffs.mul(Tensor([other]))
      return Polynomial(new_coeffs, self.PrimeField)

  def __rmul__(self, other):
    return self.__mul__(other)

  def __mod__(self, other: "Polynomial") -> "Polynomial":
    """
    Polynomial modulo operation.
    Returns the remainder when dividing self by other.
    """
    _, remainder = self.divmod(other)
    return remainder

  def __floordiv__(self, other: "Polynomial") -> "Polynomial":
    """
    Polynomial floor division.
    Returns the quotient when dividing self by other.
    """
    quotient, _ = self.divmod(other)
    return quotient

  def __repr__(self):
    coeffs_list = self.coeffs.numpy().tolist()
    return f"Polynomial({coeffs_list})"

  def __call__(self, x: int | Tensor):
    return self.evaluate(x)

  def gcd(self, other: "Polynomial") -> "Polynomial":
    """
    Compute the greatest common divisor of two polynomials using Euclidean algorithm.
    Returns a polynomial that divides both self and other.
    """
    # Handle zero polynomials using tensor operations
    if self.coeffs.shape[0] == 0 or self._is_zero():
      return Polynomial(other.coeffs, self.PrimeField)
    if other.coeffs.shape[0] == 0 or other._is_zero():
      return Polynomial(self.coeffs, self.PrimeField)

    # Euclidean algorithm
    a = Polynomial(self.coeffs, self.PrimeField)
    b = Polynomial(other.coeffs, self.PrimeField)

    while b.coeffs.shape[0] > 0 and not b._is_zero():
      r = a % b
      a = b
      b = r

    # Make monic (leading coefficient = 1) if in a field
    if self.PrimeField is not None and a.coeffs.shape[0] > 0:
      # Handle case where all coefficients might be zero after reduction
      if a._is_zero():
        return Polynomial([1], self.PrimeField)  # Return 1 as GCD
      # Get leading coefficient as tensor
      lead_coeff_tensor = a.coeffs[-1:]
      # Create inverse using field operations
      lead_inv = self.PrimeField(lead_coeff_tensor.item()).inv()
      a = a * lead_inv.value.item()

    return a

  def _is_zero(self) -> bool:
    """Check if polynomial is zero without converting to numpy."""
    # For small polynomials, converting to numpy is acceptable
    return (self.coeffs == 0).all().item()

  def derivative(self) -> "Polynomial":
    """
    Compute the derivative of the polynomial.
    For p(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
    p'(x) = a1 + 2*a2*x + 3*a3*x^2 + ... + n*an*x^(n-1)
    """
    if self.coeffs.shape[0] <= 1:
      # Derivative of constant is 0
      return Polynomial([0], self.PrimeField)

    # Create indices tensor [1, 2, 3, ..., n]
    indices = Tensor.arange(1, self.coeffs.shape[0], dtype=self.coeffs.dtype)

    # Multiply coefficients by their indices
    new_coeffs = self.coeffs[1:] * indices

    # Apply modular reduction if in a prime field
    if self.PrimeField is not None:
      new_coeffs = new_coeffs.mod(Tensor([self.PrimeField.P]))

    return Polynomial(new_coeffs, self.PrimeField)

  def compose(self, other: "Polynomial") -> "Polynomial":
    """
    Polynomial composition: compute p(q(x)) where self is p and other is q.
    Uses Horner's method for efficiency.
    """
    if self.coeffs.shape[0] == 0:
      return Polynomial([0], self.PrimeField)

    # Start with the highest degree coefficient as a tensor
    result = Polynomial(self.coeffs[-1:], self.PrimeField)

    # Apply Horner's method: p(x) = (...((an*x + an-1)*x + an-2)*x + ... + a0)
    for i in range(self.coeffs.shape[0] - 2, -1, -1):
      # result = result * other + coeffs[i]
      result = result * other + Polynomial(self.coeffs[i : i + 1], self.PrimeField)

    return result

  def divmod(self, divisor: "Polynomial") -> tuple["Polynomial", "Polynomial"]:
    """
    Polynomial division with remainder.
    Returns (quotient, remainder) such that self = divisor * quotient + remainder
    and degree(remainder) < degree(divisor).
    """
    if divisor.coeffs.shape[0] == 0 or divisor._is_zero():
      raise ValueError("Division by zero polynomial")

    # If dividend degree < divisor degree, quotient is 0 and remainder is dividend
    if self.degree() < divisor.degree():
      zero_poly = Polynomial([0], self.PrimeField)
      return zero_poly, Polynomial(self.coeffs, self.PrimeField)

    # For now, we need to use numpy for division algorithm
    # A fully tensorized version would require more complex tensor manipulations
    remainder_coeffs = self.coeffs.numpy().copy()
    quotient_coeffs = []

    divisor_coeffs = divisor.coeffs.numpy()
    divisor_lead = divisor_coeffs[-1]

    # Compute modular inverse of leading coefficient
    if self.PrimeField is not None:
      divisor_lead_inv = self.PrimeField(int(divisor_lead)).inv().value.item()
    else:
      divisor_lead_inv = 1 / divisor_lead

    while len(remainder_coeffs) >= len(divisor_coeffs):
      # Compute next quotient coefficient
      if self.PrimeField is not None:
        coeff = int((int(remainder_coeffs[-1]) * divisor_lead_inv) % self.PrimeField.P)
      else:
        coeff = remainder_coeffs[-1] * divisor_lead_inv

      quotient_coeffs.append(coeff)

      # Subtract divisor * coeff from remainder
      for i in range(len(divisor_coeffs)):
        if self.PrimeField is not None:
          remainder_coeffs[-(i + 1)] = int((int(remainder_coeffs[-(i + 1)]) - int(coeff) * int(divisor_coeffs[-(i + 1)])) % self.PrimeField.P)
        else:
          remainder_coeffs[-(i + 1)] -= coeff * divisor_coeffs[-(i + 1)]

      # Remove leading term
      remainder_coeffs = remainder_coeffs[:-1]

    # Reverse quotient coefficients
    quotient_coeffs = quotient_coeffs[::-1] if quotient_coeffs else [0]
    remainder_coeffs = remainder_coeffs if len(remainder_coeffs) > 0 else [0]

    return (Polynomial(quotient_coeffs, self.PrimeField), Polynomial(remainder_coeffs, self.PrimeField))
