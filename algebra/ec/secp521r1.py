"""secp521r1 (P-521) curve implementation (NIST standard curve)"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.prime_field import PrimeField


class Fp(PrimeField):
  """secp521r1 base field"""

  P = 0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class Fr(PrimeField):
  """secp521r1 scalar field"""

  P = 0x01FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFA51868783BF2F966B7FCC0148F709A5D03BB5C9B8899C47AEBB6FB71E91386409


class Secp521r1(EllipticCurve):
  """secp521r1 curve: y^2 = x^3 - 3x + b"""

  def __init__(self):
    a = -3
    b = 0x0051953EB9618E1C9A1F929A21A0B68540EEA2DA725B99B315F3B8B489918EF109E156193951EC7E937B1652C0BD3BB1BF073573DF883D2C34F1EF451FD46B503F00
    super().__init__(a, b, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point"""
    curve = cls()
    x = 0x00C6858E06B70404E9CD9E3ECB662395B4429C648139053FB521F828AF606B4D3DBAA14B5E77EFE75928FE1DC127A2FFA8DE3348B3C1856A429BF97E7E31C2E5BD66
    y = 0x011839296A789A3BC0045C8A5FB42C7D1BD998F54449579B446817AFBD17273E662C97EE72995EF42640C550B9013FAD0761353C7086A272C24088BE94769FD16650
    return ECPoint(x, y, curve)


def test_secp521r1():
  """Basic secp521r1 tests"""
  # Create curve
  curve = Secp521r1()

  # Generator
  G = Secp521r1.generator()
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
  G1000 = G.scalar_mul(1000)
  assert G1000.is_on_curve()

  print("secp521r1 tests passed!")


if __name__ == "__main__":
  test_secp521r1()
