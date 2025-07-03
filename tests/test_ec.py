from algebra.ec.curve import ECPoint, EllipticCurve
from algebra.ff.prime_field import PrimeField
from tinygrad import Tensor, dtypes


class Fp(PrimeField):
  """Field for testing - using a small prime for easy verification"""

  P = 101


def test_ec_point_creation():
  # Test curve: y^2 = x^3 + 2x + 2 over Fp
  curve = EllipticCurve(2, 2, Fp)

  # Point at infinity
  inf_point = ECPoint.infinity(curve)
  assert inf_point.is_infinity()

  # Valid point (5, 6) on curve
  P = ECPoint(5, 6, curve)
  assert P.is_on_curve()

  # Invalid point should raise error
  try:
    ECPoint(0, 1, curve)
    assert False, "Should have raised error for invalid point"
  except ValueError:
    pass


def test_ec_point_addition():
  curve = EllipticCurve(2, 2, Fp)

  # P + O = P (identity)
  P = ECPoint(5, 6, curve)
  inf_point = ECPoint.infinity(curve)
  assert (P + inf_point).equals(P)
  assert (inf_point + P).equals(P)

  # P + (-P) = O
  neg_P = -P
  assert (P + neg_P).is_infinity()

  # Addition of two different points
  Q = ECPoint(7, 37, curve)
  R = P + Q
  assert R.is_on_curve()
  # We'll verify the exact coordinates after running


def test_ec_point_doubling():
  curve = EllipticCurve(2, 2, Fp)

  P = ECPoint(5, 6, curve)
  P2 = P.double()
  assert P2.is_on_curve()

  # 2P should equal P + P
  assert P2.equals(P + P)


def test_scalar_multiplication():
  curve = EllipticCurve(2, 2, Fp)
  P = ECPoint(5, 6, curve)

  # 0 * P = O
  assert P.scalar_mul(0).is_infinity()

  # 1 * P = P
  assert P.scalar_mul(1).equals(P)

  # 2 * P = P + P
  assert P.scalar_mul(2).equals(P + P)

  # 5 * P
  P5 = P.scalar_mul(5)
  assert P5.is_on_curve()

  # k * P + m * P = (k + m) * P
  P3 = P.scalar_mul(3)
  assert (P3 + P.scalar_mul(2)).equals(P5)


def test_ec_batch_operations():
  """Test batch point operations for efficiency"""
  curve = EllipticCurve(2, 2, Fp)

  # Create batch of points
  xs = Tensor([5, 7, 8], dtype=dtypes.int32)
  ys = Tensor([6, 37, 5], dtype=dtypes.int32)

  # Batch verify they're on curve
  on_curve = curve.batch_is_on_curve(xs, ys)
  assert on_curve.all().item()

  # Test multi-scalar multiplication
  points = [ECPoint(5, 6, curve), ECPoint(7, 37, curve), ECPoint(8, 5, curve)]
  scalars = [2, 3, 1]

  result = ECPoint.multi_scalar_mul(points, scalars)
  expected = points[0].scalar_mul(2) + points[1].scalar_mul(3) + points[2].scalar_mul(1)
  assert result.equals(expected)
