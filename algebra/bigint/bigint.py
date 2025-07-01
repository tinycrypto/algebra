from tinygrad import Tensor, dtypes


class BigInt:
  """Big integer using Tinygrad tensors with simple operations."""

  LIMB_BITS = 26  # Use 26-bit limbs to prevent overflow
  LIMB_MASK = (1 << LIMB_BITS) - 1
  LIMB_BASE = 1 << LIMB_BITS

  def __init__(self, value: int | Tensor, sign: int = 1):
    """Initialize from integer or tensor of limbs."""
    if isinstance(value, int):
      if value == 0:
        self.limbs = Tensor.zeros(1, dtype=dtypes.int32)
        self.sign = 1
      else:
        # Convert to limbs
        v = abs(value)
        limbs = []
        while v > 0:
          limbs.append(v & self.LIMB_MASK)
          v >>= self.LIMB_BITS
        self.limbs = Tensor(limbs, dtype=dtypes.int32)
        self.sign = -1 if value < 0 else 1
    else:
      # Keep as original dtype for intermediate calculations
      self.limbs = value
      self.sign = sign

  def _normalize(self) -> "BigInt":
    """Normalize using simple tensor operations."""
    # Convert to numpy for normalization, then back to tensor
    limbs_np = self.limbs.numpy().astype(int)

    # Handle carries properly for large numbers
    carry = 0
    normalized = []
    for limb in limbs_np:
      val = int(limb) + carry
      normalized.append(val % self.LIMB_BASE)
      carry = val // self.LIMB_BASE

    # Add remaining carry
    while carry > 0:
      normalized.append(carry % self.LIMB_BASE)
      carry //= self.LIMB_BASE

    # Remove leading zeros
    while len(normalized) > 1 and normalized[-1] == 0:
      normalized.pop()

    # Always return as int32 after normalization
    return BigInt(Tensor(normalized, dtype=dtypes.int32), self.sign)

  def __add__(self, other: "BigInt") -> "BigInt":
    """Addition using tensor operations."""
    if not isinstance(other, BigInt):
      other = BigInt(other)

    if self.sign != other.sign:
      return self.__sub__(BigInt(other.limbs, -other.sign))

    # Pad to same length using tensor operations
    max_len = max(self.limbs.shape[0], other.limbs.shape[0])
    a = self.limbs.cast(dtypes.int64).pad((0, max_len - self.limbs.shape[0]))
    b = other.limbs.cast(dtypes.int64).pad((0, max_len - other.limbs.shape[0]))

    # Simple vectorized addition
    result = BigInt(a + b, self.sign)
    return result._normalize()

  def __sub__(self, other: "BigInt") -> "BigInt":
    """Subtraction using tensor operations."""
    if not isinstance(other, BigInt):
      other = BigInt(other)

    if self.sign != other.sign:
      return self.__add__(BigInt(other.limbs, -other.sign))

    # Compare magnitudes
    cmp = self._compare_mag(other)
    if cmp < 0:
      result = other.__sub__(self)
      result.sign = -result.sign
      return result
    elif cmp == 0:
      return BigInt(0)

    # Simple subtraction using numpy for borrow handling
    a_np = self.limbs.numpy().astype(int)
    b_np = other.limbs.numpy().astype(int)

    # Pad to same length
    max_len = max(len(a_np), len(b_np))
    a_padded = list(a_np) + [0] * (max_len - len(a_np))
    b_padded = list(b_np) + [0] * (max_len - len(b_np))

    # Subtract with borrow
    result = []
    borrow = 0
    for i in range(max_len):
      val = a_padded[i] - b_padded[i] - borrow
      if val < 0:
        val += self.LIMB_BASE
        borrow = 1
      else:
        borrow = 0
      result.append(val)

    return BigInt(Tensor(result, dtype=dtypes.int32), self.sign)._normalize()

  def __mul__(self, other: "BigInt") -> "BigInt":
    """Multiplication using schoolbook algorithm for correctness."""
    if not isinstance(other, BigInt):
      other = BigInt(other)

    # Use schoolbook multiplication for correctness
    a_np = self.limbs.numpy().astype(int)
    b_np = other.limbs.numpy().astype(int)

    # Initialize result array
    result = [0] * (len(a_np) + len(b_np))

    # Schoolbook multiplication
    for i in range(len(a_np)):
      for j in range(len(b_np)):
        result[i + j] += a_np[i] * b_np[j]

    # Create result and normalize
    res = BigInt(Tensor(result, dtype=dtypes.int64), self.sign * other.sign)
    return res._normalize()

  def __divmod__(self, other: "BigInt") -> tuple["BigInt", "BigInt"]:
    """Division with remainder."""
    if not isinstance(other, BigInt):
      other = BigInt(other)

    if other._is_zero():
      raise ValueError("Division by zero")

    # Handle signs
    sign_q = self.sign * other.sign
    sign_r = self.sign

    # Work with absolute values
    dividend = abs(self)
    divisor = abs(other)

    # Simple case
    if dividend < divisor:
      return BigInt(0), self

    # Use Python division for simplicity
    dividend_int = dividend.to_int()
    divisor_int = divisor.to_int()

    q, r = divmod(dividend_int, divisor_int)

    quotient = BigInt(q)
    remainder = BigInt(r)

    quotient.sign = sign_q
    remainder.sign = sign_r if r != 0 else 1

    return quotient, remainder

  def __floordiv__(self, other: "BigInt") -> "BigInt":
    """Floor division."""
    q, _ = divmod(self, other)
    return q

  def __mod__(self, other: "BigInt") -> "BigInt":
    """Modulo."""
    _, r = divmod(self, other)
    return r

  def __pow__(self, exp: int, mod: "BigInt" = None) -> "BigInt":
    """Exponentiation using square-and-multiply."""
    if exp < 0:
      raise ValueError("Negative exponent not supported")

    if exp == 0:
      return BigInt(1)

    # Square and multiply
    result = BigInt(1)
    base = self

    while exp > 0:
      if exp & 1:
        result = result * base
        if mod:
          result = result % mod
      base = base * base
      if mod:
        base = base % mod
      exp >>= 1

    return result

  def __lshift__(self, bits: int) -> "BigInt":
    """Left shift."""
    if bits == 0:
      return BigInt(self.limbs, self.sign)

    # Convert to int, shift, convert back (simple but correct)
    val = self.to_int()
    return BigInt(val << bits)

  def __rshift__(self, bits: int) -> "BigInt":
    """Right shift."""
    if bits == 0:
      return BigInt(self.limbs, self.sign)

    # Convert to int, shift, convert back
    val = abs(self.to_int())
    result = BigInt(val >> bits)
    result.sign = self.sign
    return result

  def _compare_mag(self, other: "BigInt") -> int:
    """Compare magnitudes."""
    if self.limbs.shape[0] != other.limbs.shape[0]:
      return 1 if self.limbs.shape[0] > other.limbs.shape[0] else -1

    # Compare limb by limb from most significant
    self_np = self.limbs.numpy()
    other_np = other.limbs.numpy()

    for i in range(len(self_np) - 1, -1, -1):
      if self_np[i] != other_np[i]:
        return 1 if self_np[i] > other_np[i] else -1

    return 0

  def _is_zero(self) -> bool:
    """Check if zero."""
    return not (self.limbs != 0).any().item()

  def to_int(self) -> int:
    """Convert to Python int."""
    result = 0
    limbs_np = self.limbs.numpy()
    for i in range(len(limbs_np) - 1, -1, -1):
      result = (result << self.LIMB_BITS) + int(limbs_np[i])
    return result * self.sign

  # Comparison operators
  def __eq__(self, other) -> bool:
    if not isinstance(other, BigInt):
      other = BigInt(other)
    return self.sign == other.sign and self._compare_mag(other) == 0

  def __lt__(self, other) -> bool:
    if not isinstance(other, BigInt):
      other = BigInt(other)
    if self.sign != other.sign:
      return self.sign < other.sign
    return (self._compare_mag(other) < 0) if self.sign > 0 else (self._compare_mag(other) > 0)

  def __le__(self, other) -> bool:
    return self == other or self < other

  def __gt__(self, other) -> bool:
    return not self <= other

  def __ge__(self, other) -> bool:
    return not self < other

  # Unary operators
  def __neg__(self) -> "BigInt":
    """Negation."""
    return BigInt(self.limbs, -self.sign)

  def __abs__(self) -> "BigInt":
    """Absolute value."""
    return BigInt(self.limbs, 1)

  def __int__(self) -> int:
    """Convert to int."""
    return self.to_int()

  def __repr__(self) -> str:
    return f"BigInt({self.to_int()})"


# Utility functions using basic BigInt operations
def mod_inverse(a: BigInt, n: BigInt) -> BigInt:
  """Modular inverse using extended Euclidean algorithm."""
  if a._is_zero():
    raise ValueError("No inverse for 0")

  # Extended GCD
  old_r, r = n, a
  old_s, s = BigInt(0), BigInt(1)

  while not r._is_zero():
    q, _ = divmod(old_r, r)
    old_r, r = r, old_r - q * r
    old_s, s = s, old_s - q * s

  # Make positive
  if old_s.sign < 0:
    old_s = old_s + n

  return old_s


def gcd(a: BigInt, b: BigInt) -> BigInt:
  """Greatest common divisor using Euclidean algorithm."""
  while not b._is_zero():
    a, b = b, a % b
  return abs(a)


def lcm(a: BigInt, b: BigInt) -> BigInt:
  """Least common multiple."""
  return abs(a * b) // gcd(a, b)
