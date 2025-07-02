from tinygrad import Tensor
from algebra.ff.bigint_field import BigIntPrimeField


class EllipticCurve:
  """Elliptic curve in Weierstrass form: y^2 = x^3 + ax + b"""

  def __init__(self, a: int, b: int, field: type[BigIntPrimeField]):
    self.a = field(a)
    self.b = field(b)
    self.field = field

    # Check discriminant: -16(4a^3 + 27b^2) != 0
    discriminant = field(4) * self.a**3 + field(27) * self.b**2
    if int(discriminant) == 0:
      raise ValueError("Invalid curve: discriminant is zero")

  def batch_is_on_curve(self, xs: Tensor, ys: Tensor) -> Tensor:
    """Check if batch of points are on curve"""
    # Use field operations instead of direct modulo
    x3 = self.field.mul_mod(self.field.mul_mod(xs, xs), xs)
    ax = self.field.mul_mod(self.a.value, xs) 
    y2 = self.field.mul_mod(ys, ys)
    rhs = self.field.add(self.field.add(x3, ax), self.b.value)
    return self.field.eq_t(y2, rhs)


class ECPoint:
  """Point on an elliptic curve"""

  def __init__(self, x: int | BigIntPrimeField | None, y: int | BigIntPrimeField | None, curve: EllipticCurve):
    self.curve = curve
    self.field = curve.field

    if x is None and y is None:
      # Point at infinity
      self.x = None
      self.y = None
    else:
      self.x = self.field(x) if not isinstance(x, self.field) else x
      self.y = self.field(y) if not isinstance(y, self.field) else y

      # Verify point is on curve
      if not self._verify_on_curve():
        raise ValueError(f"Point ({int(self.x)}, {int(self.y)}) is not on curve")

  @classmethod
  def infinity(cls, curve: EllipticCurve) -> "ECPoint":
    """Create point at infinity"""
    return cls(None, None, curve)

  def is_infinity(self) -> bool:
    """Check if point is at infinity"""
    return self.x is None

  def _verify_on_curve(self) -> bool:
    """Verify point satisfies curve equation"""
    if self.is_infinity():
      return True
    # y^2 = x^3 + ax + b
    y2 = self.y * self.y
    x3 = self.x * self.x * self.x
    return y2 == x3 + self.curve.a * self.x + self.curve.b

  def is_on_curve(self) -> bool:
    """Check if point is on curve"""
    return self._verify_on_curve()

  def equals(self, other: "ECPoint") -> bool:
    """Check if two points are equal"""
    if self.is_infinity() and other.is_infinity():
      return True
    if self.is_infinity() or other.is_infinity():
      return False
    return self.x == other.x and self.y == other.y

  def __neg__(self) -> "ECPoint":
    """Negate point"""
    if self.is_infinity():
      return self
    return ECPoint(self.x, -self.y, self.curve)

  def __add__(self, other: "ECPoint") -> "ECPoint":
    """Add two points"""
    if self.curve != other.curve:
      raise ValueError("Points must be on same curve")

    # O + P = P
    if self.is_infinity():
      return other
    if other.is_infinity():
      return self

    # P + (-P) = O
    if self.x == other.x:
      if self.y == other.y:
        return self.double()
      else:
        return ECPoint.infinity(self.curve)

    # General case: P + Q where P != Q
    # slope = (y2 - y1) / (x2 - x1)
    dx = other.x - self.x
    dy = other.y - self.y
    slope = dy / dx

    # x3 = slope^2 - x1 - x2
    x3 = slope * slope - self.x - other.x

    # y3 = slope * (x1 - x3) - y1
    y3 = slope * (self.x - x3) - self.y

    return ECPoint(x3, y3, self.curve)

  def double(self) -> "ECPoint":
    """Double a point"""
    if self.is_infinity():
      return self

    # If y = 0, then 2P = O
    if int(self.y) == 0:
      return ECPoint.infinity(self.curve)

    # slope = (3x^2 + a) / (2y)
    three = self.field(3)
    two = self.field(2)

    numerator = three * self.x * self.x + self.curve.a
    denominator = two * self.y
    slope = numerator / denominator

    # x3 = slope^2 - 2x
    x3 = slope * slope - two * self.x

    # y3 = slope * (x - x3) - y
    y3 = slope * (self.x - x3) - self.y

    return ECPoint(x3, y3, self.curve)

  def scalar_mul(self, k: int) -> "ECPoint":
    """Scalar multiplication using double-and-add"""
    if k == 0:
      return ECPoint.infinity(self.curve)
    if k < 0:
      return (-self).scalar_mul(-k)

    # Double-and-add algorithm
    result = ECPoint.infinity(self.curve)
    addend = self

    while k:
      if k & 1:
        result = result + addend
      addend = addend.double()
      k >>= 1

    return result

  @staticmethod
  def multi_scalar_mul(points: list["ECPoint"], scalars: list[int]) -> "ECPoint":
    """Multi-scalar multiplication: sum(k_i * P_i) using windowed method"""
    if not points:
      raise ValueError("Empty points list")

    if len(points) != len(scalars):
      raise ValueError("Points and scalars must have same length")

    # Use Shamir's trick for small number of points
    if len(points) <= 3:
      result = ECPoint.infinity(points[0].curve)
      for point, scalar in zip(points, scalars):
        result = result + point.scalar_mul(scalar)
      return result

    # For larger sets, use bucket method (simplified Pippenger)
    # Find max scalar bit length
    max_scalar = max(abs(s) for s in scalars)
    if max_scalar == 0:
      return ECPoint.infinity(points[0].curve)

    bit_len = max_scalar.bit_length()
    window_size = min(8, max(1, bit_len // 3))  # Adaptive window size

    # Process in windows
    result = ECPoint.infinity(points[0].curve)

    for window_start in range(0, bit_len, window_size):
      # Double result window_size times
      for _ in range(min(window_size, bit_len - window_start)):
        result = result.double()

      # Create buckets for this window
      num_buckets = 1 << window_size
      buckets = [ECPoint.infinity(points[0].curve) for _ in range(num_buckets)]

      # Add points to buckets based on scalar bits in this window
      for point, scalar in zip(points, scalars):
        if scalar < 0:
          point = -point
          scalar = -scalar

        # Extract window bits
        window_bits = (scalar >> window_start) & ((1 << window_size) - 1)
        if window_bits > 0:
          buckets[window_bits] = buckets[window_bits] + point

      # Sum buckets with appropriate multipliers
      for i in range(num_buckets - 1, 0, -1):
        if not buckets[i].is_infinity():
          result = result + buckets[i]

    return result
