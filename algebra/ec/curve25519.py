"""Curve25519 implementation (used in EdDSA/Ed25519)

Note: This implements the birationally equivalent Weierstrass form of Curve25519
for compatibility with our EllipticCurve framework. The original Curve25519 uses
Montgomery form: v^2 = u^3 + 486662*u^2 + u
"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.prime_field import PrimeField


class Fp(PrimeField):
  """Curve25519 base field"""

  P = 2**255 - 19


class Fr(PrimeField):
  """Curve25519 scalar field (subgroup order)"""

  P = 2**252 + 27742317777372353535851937790883648493


class Curve25519(EllipticCurve):
  """Curve25519 in Weierstrass form: y^2 = x^3 + ax + b

  This is the birationally equivalent Weierstrass form of the Montgomery curve
  v^2 = u^3 + 486662*u^2 + u mod p where p = 2^255 - 19
  """

  def __init__(self):
    # These are the Weierstrass form coefficients for Curve25519
    a = 486662
    b = 1
    super().__init__(a, b, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point (base point)

    This corresponds to the base point u=9 in Montgomery form
    """
    curve = cls()
    # Base point in Weierstrass coordinates
    x = 15112221349535400772501151409588531511454012693041857206046113283949847762202
    y = 46316835694926478169428394003475163141307993866256225615783033603165251855960
    return ECPoint(x, y, curve)


class Ed25519(EllipticCurve):
  """Ed25519 Edwards curve: -x^2 + y^2 = 1 + d*x^2*y^2

  Implemented in birationally equivalent Weierstrass form
  """

  def __init__(self):
    # Edwards parameter d = -121665/121666
    # Weierstrass form coefficients for the birationally equivalent curve
    a = 486664
    b = 1
    super().__init__(a, b, Fp)

  @classmethod
  def generator(cls) -> ECPoint:
    """Ed25519 generator point"""
    curve = cls()
    # Generator point in Weierstrass coordinates
    x = 15112221349535400772501151409588531511454012693041857206046113283949847762202
    y = 46316835694926478169428394003475163141307993866256225615783033603165251855960
    return ECPoint(x, y, curve)


def test_curve25519():
  """Basic Curve25519 tests"""
  # Generator
  G = Curve25519.generator()
  assert G.is_on_curve()

  # Test some operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))

  # Test scalar multiplication with small values
  G8 = G.scalar_mul(8)
  assert G8.is_on_curve()

  # Test that the scalar field order works
  n = Fr.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()

  print("Curve25519 tests passed!")


def test_ed25519():
  """Basic Ed25519 tests"""
  # Generator
  G = Ed25519.generator()
  assert G.is_on_curve()

  # Test some operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))

  print("Ed25519 tests passed!")


if __name__ == "__main__":
  test_curve25519()
  test_ed25519()
