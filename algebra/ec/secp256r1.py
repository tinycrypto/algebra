"""secp256r1 (P-256) curve implementation (NIST standard curve)"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.prime_field import PrimeField


class Fp(PrimeField):
  """secp256r1 base field"""

  P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF


class Fr(PrimeField):
  """secp256r1 scalar field"""

  P = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551


class Secp256r1(EllipticCurve):
  """secp256r1 curve: y^2 = x^3 - 3x + b"""

  def __init__(self):
    a = -3
    b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
    super().__init__(a, b, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point"""
    curve = cls()
    x = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
    y = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
    return ECPoint(x, y, curve)


def test_secp256r1():
  """Basic secp256r1 tests"""
  # Create curve
  curve = Secp256r1()

  # Generator
  G = Secp256r1.generator()
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

  # Test known values for 2G
  expected_2G_x = 0x7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978
  expected_2G_y = 0x07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1

  actual_2G = G.double()
  assert actual_2G.x.value.item() == expected_2G_x
  assert actual_2G.y.value.item() == expected_2G_y

  print("secp256r1 tests passed!")


if __name__ == "__main__":
  test_secp256r1()
