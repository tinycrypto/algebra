"""BigInt-based prime field implementation that avoids tinygrad tensor operations with large constants"""

from algebra.bigint.bigint import BigInt
from tinygrad.tensor import Tensor
from tinygrad import dtypes


class BigIntPrimeField:
  """Prime field implementation using BigInt for all arithmetic"""

  P: int = None

  def __init__(self, x):
    if isinstance(x, int):
      # Use Python's modulo operator for correct handling of negative numbers
      reduced_x = x % self.P
      self._value = BigInt(reduced_x)
    elif isinstance(x, BigInt):
      # Convert to int, apply Python modulo, then back to BigInt
      reduced_x = x.to_int() % self.P
      self._value = BigInt(reduced_x)
    elif isinstance(x, BigIntPrimeField):
      self._value = x._value
    else:
      raise ValueError(f"Cannot create {self.__class__.__name__} from {type(x)}")

  @property
  def value(self):
    """Return as a tensor with a small reduced value"""
    # Always return the reduced integer value as a tensor
    reduced_val = self._value.to_int()
    return Tensor([reduced_val], dtype=dtypes.int64)

  def __add__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    result = (self._value.to_int() + other._value.to_int()) % self.P
    return type(self)(result)

  def __sub__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    result = (self._value.to_int() - other._value.to_int()) % self.P
    return type(self)(result)

  def __mul__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    result = (self._value.to_int() * other._value.to_int()) % self.P
    return type(self)(result)

  def __pow__(self, exponent):
    result = pow(self._value.to_int(), exponent, self.P)
    return type(self)(result)

  def __neg__(self):
    result = (-self._value.to_int()) % self.P
    return type(self)(result)

  def __truediv__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return self * other.inv()

  def inv(self):
    """Modular inverse"""
    # Use Python's built-in pow function with -1 exponent for modular inverse
    result = pow(self._value.to_int(), -1, self.P)
    return type(self)(result)

  def __eq__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return self._value == other._value

  def __repr__(self):
    return f"{self._value.to_int()}"

  def __int__(self):
    return self._value.to_int()

  # Aliases for compatibility
  __radd__ = __add__
  __rmul__ = __mul__

  def __rsub__(self, other):
    return -(self - other)

  def __rtruediv__(self, other):
    return self.inv() * other

  # Batch operation methods for tensor operations
  @classmethod
  def add(cls, a: Tensor, b: Tensor) -> Tensor:
    """Add two tensors with modular reduction using element-wise operations"""
    # Use pure element-wise tensor operations - no manual iteration!
    # Cast to int64 for safe arithmetic if needed
    if cls.P < (1 << 63):
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      return ((a_64 + b_64) % cls.P).cast(a.dtype)
    else:
      # For very large primes, still use element-wise operations
      # Python's modulo works element-wise on tensors
      return (a + b) % cls.P

  @classmethod
  def sub(cls, a: Tensor, b: Tensor) -> Tensor:
    """Subtract two tensors with modular reduction using element-wise operations"""
    # Use pure element-wise tensor operations - no manual iteration!
    if cls.P < (1 << 63):
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      return ((a_64 - b_64) % cls.P).cast(a.dtype)
    else:
      # For very large primes, use element-wise operations
      return (a - b) % cls.P

  @classmethod
  def mul_mod(cls, a: Tensor, b: Tensor) -> Tensor:
    """Multiply two tensors with modular reduction using element-wise operations"""
    # Use pure element-wise tensor operations - no manual iteration!
    if cls.P < (1 << 31):
      # For smaller primes, use 64-bit intermediate to prevent overflow
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      return ((a_64 * b_64) % cls.P).cast(a.dtype)
    else:
      # For larger primes, use element-wise operations directly
      return (a * b) % cls.P

  @classmethod
  def eq_t(cls, x: Tensor, y: Tensor) -> Tensor:
    """Compare two tensors element-wise using pure tensor operations"""
    # Use element-wise tensor comparison - no manual iteration!
    x_mod = x % cls.P
    y_mod = y % cls.P
    return (x_mod == y_mod).float()
