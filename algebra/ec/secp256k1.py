"""secp256k1 curve implementation (Bitcoin/Ethereum curve)"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.prime_field import PrimeField


class Fp(PrimeField):
  """secp256k1 base field"""

  P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F


class Fr(PrimeField):
  """secp256k1 scalar field"""

  P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


class Secp256k1(EllipticCurve):
  """secp256k1 curve: y^2 = x^3 + 7"""

  def __init__(self):
    super().__init__(0, 7, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point"""
    curve = cls()
    x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    return ECPoint(x, y, curve)


def test_secp256k1():
  """Basic secp256k1 tests"""
  # Create curve
  curve = Secp256k1()

  # Generator
  G = Secp256k1.generator()
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

  # Test known values
  # 2G = (x, y) where x, y are known values
  expected_2G_x = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
  expected_2G_y = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A

  actual_2G = G.double()
  assert actual_2G.x.value.item() == expected_2G_x
  assert actual_2G.y.value.item() == expected_2G_y

  print("secp256k1 tests passed!")


if __name__ == "__main__":
  test_secp256k1()
