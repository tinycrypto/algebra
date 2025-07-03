"""BN254 curve implementation for ZK proofs"""

from algebra.ec.curve import EllipticCurve, ECPoint
from algebra.ff.bigint_field import BigIntPrimeField


class Fq(BigIntPrimeField):
  """BN254 base field"""

  P = 21888242871839275222246405745257275088696311157297823662689037894645226208583


class Fr(BigIntPrimeField):
  """BN254 scalar field"""

  P = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class G1(EllipticCurve):
  """BN254 G1 curve: y^2 = x^3 + 3"""

  def __init__(self):
    super().__init__(0, 3, Fq)

  @classmethod
  def generator(cls) -> ECPoint:
    """Standard generator point"""
    curve = cls()
    return ECPoint(1, 2, curve)


class G2:
  """BN254 G2 curve (extension field) - placeholder for pairing support"""

  # Full G2 implementation would require Fq2 extension field
  pass


def test_bn254():
  """Basic BN254 tests"""
  # Create curve
  _ = G1()

  # Generator
  P = G1.generator()
  assert P.is_on_curve()

  # Scalar field order
  n = Fr.P

  # nP should be infinity (point at infinity)
  nP = P.scalar_mul(n)
  assert nP.is_infinity()

  # Test some operations
  P2 = P.double()
  assert P2.is_on_curve()

  P3 = P + P2
  assert P3.is_on_curve()
  assert P3.equals(P.scalar_mul(3))

  print("BN254 tests passed!")


if __name__ == "__main__":
  test_bn254()
