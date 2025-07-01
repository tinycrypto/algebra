"""Tests for BN254 curve"""

import pytest
from algebra.ec.bn254 import G1 as BN254, Fr


def test_bn254_basic():
  """Test BN254 basic operations"""
  # Create curve and generator
  G = BN254.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_bn254_order():
  """Test BN254 curve order"""
  G = BN254.generator()
  n = Fr.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_bn254_point_operations():
  """Test BN254 point arithmetic"""
  G = BN254.generator()

  # Test that -(-P) = P
  neg_G = -G
  neg_neg_G = -neg_G
  assert G.equals(neg_neg_G)

  # Test that P + (-P) = O
  sum_with_neg = G + (-G)
  assert sum_with_neg.is_infinity()

  # Test that 2*P = P + P
  double_P = G.double()
  add_P = G + G
  assert double_P.equals(add_P)


def test_bn254_scalar_multiplication():
  """Test BN254 scalar multiplication properties"""
  G = BN254.generator()

  # Test that k*G = G + G + ... + G (k times) for small k
  k = 5
  kG_scalar = G.scalar_mul(k)
  kG_addition = G
  for _ in range(k - 1):
    kG_addition = kG_addition + G

  assert kG_scalar.equals(kG_addition)

  # Test that (a + b)*G = a*G + b*G
  a, b = 7, 11
  ab_G = G.scalar_mul(a + b)
  aG_plus_bG = G.scalar_mul(a) + G.scalar_mul(b)

  assert ab_G.equals(aG_plus_bG)


if __name__ == "__main__":
  test_bn254_basic()
  test_bn254_order()
  test_bn254_point_operations()
  test_bn254_scalar_multiplication()
  print("BN254 tests passed!")
