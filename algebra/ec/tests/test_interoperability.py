"""Cross-curve interoperability and consistency tests"""

import pytest
from algebra.ec.bn254 import G1 as BN254
from algebra.ec.secp256k1 import Secp256k1
from algebra.ec.secp256r1 import Secp256r1
from algebra.ec.curve25519 import Curve25519


def test_scalar_multiplication_consistency():
  """Test that scalar multiplication is consistent across curves"""
  curves_and_generators = [
    (BN254.generator(), "BN254"),
    (Secp256k1.generator(), "secp256k1"),
    (Secp256r1.generator(), "secp256r1"),
    (Curve25519.generator(), "Curve25519"),
  ]

  for G, name in curves_and_generators:
    # Test that k*G = G + G + ... + G (k times) for small k
    k = 5
    kG_scalar = G.scalar_mul(k)
    kG_addition = G
    for _ in range(k - 1):
      kG_addition = kG_addition + G

    assert kG_scalar.equals(kG_addition), f"Scalar multiplication inconsistent for {name}"

    # Test that (a + b)*G = a*G + b*G
    a, b = 7, 11
    ab_G = G.scalar_mul(a + b)
    aG_plus_bG = G.scalar_mul(a) + G.scalar_mul(b)

    assert ab_G.equals(aG_plus_bG), f"Scalar multiplication distributivity failed for {name}"


def test_point_operations_consistency():
  """Test that point operations are consistent across curves"""
  curves_and_generators = [
    (BN254.generator(), "BN254"),
    (Secp256k1.generator(), "secp256k1"),
    (Secp256r1.generator(), "secp256r1"),
    (Curve25519.generator(), "Curve25519"),
  ]

  for G, name in curves_and_generators:
    # Test that -(-P) = P
    neg_G = -G
    neg_neg_G = -neg_G
    assert G.equals(neg_neg_G), f"Double negation failed for {name}"

    # Test that P + (-P) = O
    sum_with_neg = G + (-G)
    assert sum_with_neg.is_infinity(), f"P + (-P) != O for {name}"

    # Test that 2*P = P + P
    double_P = G.double()
    add_P = G + G
    assert double_P.equals(add_P), f"Double != Add for {name}"


def test_multi_scalar_multiplication():
  """Test multi-scalar multiplication on different curves"""
  curves_and_generators = [
    (BN254.generator(), "BN254"),
    (Secp256k1.generator(), "secp256k1"),
    (Secp256r1.generator(), "secp256r1"),
    (Curve25519.generator(), "Curve25519"),
  ]

  for G, name in curves_and_generators:
    # Test MSM with small values
    points = [G, G.double(), G.scalar_mul(3)]
    scalars = [2, 3, 4]

    # Expected: 2*G + 3*(2*G) + 4*(3*G) = 2*G + 6*G + 12*G = 20*G
    expected = G.scalar_mul(20)
    actual = G.multi_scalar_mul(points, scalars)

    assert expected.equals(actual), f"MSM failed for {name}"


def test_curve_properties():
  """Test mathematical properties that should hold for all curves"""
  curves = [
    (BN254(), "BN254"),
    (Secp256k1(), "secp256k1"),
    (Secp256r1(), "secp256r1"),
    (Curve25519(), "Curve25519"),
  ]

  for curve, name in curves:
    # Test that discriminant is non-zero (curves are non-singular)
    discriminant = curve.field(4) * curve.a**3 + curve.field(27) * curve.b**2
    assert discriminant.value.item() != 0, f"Curve {name} has zero discriminant"

    # Test that generator is on curve
    G = curve.__class__.generator()
    assert G.is_on_curve(), f"Generator not on curve for {name}"


def test_field_arithmetic_consistency():
  """Test that field arithmetic works consistently"""
  from algebra.ec.secp256k1 import Fp as Fp_secp256k1
  from algebra.ec.secp256r1 import Fp as Fp_secp256r1

  # Test basic field operations
  for field_class, name in [(Fp_secp256k1, "secp256k1"), (Fp_secp256r1, "secp256r1")]:
    a = field_class(123)
    b = field_class(456)

    # Test commutivity
    assert (a + b).equals(b + a), f"Addition not commutative for {name}"
    assert (a * b).equals(b * a), f"Multiplication not commutative for {name}"

    # Test associativity for addition
    c = field_class(789)
    assert ((a + b) + c).equals(a + (b + c)), f"Addition not associative for {name}"

    # Test distributivity
    assert (a * (b + c)).equals(a * b + a * c), f"Distributivity failed for {name}"


if __name__ == "__main__":
  test_scalar_multiplication_consistency()
  test_point_operations_consistency()
  test_multi_scalar_multiplication()
  test_curve_properties()
  test_field_arithmetic_consistency()
  print("Interoperability tests passed!")
