"""secp384r1 (P-384) curve implementation (NIST standard curve)"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.prime_field import PrimeField


class Fp(PrimeField):
  """secp384r1 base field"""

  P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFF0000000000000000FFFFFFFF


class Fr(PrimeField):
  """secp384r1 scalar field"""

  P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFC7634D81F4372DDF581A0DB248B0A77AECEC196ACCC52973


class Secp384r1(EllipticCurve):
  """secp384r1 curve: y^2 = x^3 - 3x + b"""

  def __init__(self):
    a = -3
    b = 0xB3312FA7E23EE7E4988E056BE3F82D19181D9C6EFE8141120314088F5013875AC656398D8A2ED19D2A85C8EDD3EC2AEF
    super().__init__(a, b, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point"""
    curve = cls()
    x = 0xAA87CA22BE8B05378EB1C71EF320AD746E1D3B628BA79B9859F741E082542A385502F25DBF55296C3A545E3872760AB7
    y = 0x3617DE4A96262C6F5D9E98BF9292DC29F8F41DBD289A147CE9DA3113B5F0B8C00A60B1CE1D7E819D7A431D7C90EA0E5F
    return ECPoint(x, y, curve)


def test_secp384r1():
  """Basic secp384r1 tests"""
  # Create curve
  curve = Secp384r1()

  # Generator
  G = Secp384r1.generator()
  assert G.is_on_curve()

  # Scalar field order
  n = Fr.P

  # nG should be infinity
  nG = G.scalar_mul(n)
  assert nG.is_infinity()

  # Test some operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))

  # Test with larger scalar
  G100 = G.scalar_mul(100)
  assert G100.is_on_curve()

  print("secp384r1 tests passed!")


if __name__ == "__main__":
  test_secp384r1()
