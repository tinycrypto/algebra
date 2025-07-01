"""Tests for Curve25519 and Ed25519"""

import pytest
from algebra.ec.curve25519 import Curve25519, Ed25519, Fr


def test_curve25519_basic():
  """Test Curve25519 basic operations"""
  # Create curve and generator
  G = Curve25519.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_curve25519_order():
  """Test Curve25519 curve order"""
  G = Curve25519.generator()
  n = Fr.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_curve25519_small_multiples():
  """Test Curve25519 with small scalar multiples"""
  G = Curve25519.generator()

  # Test small multiples
  for k in range(1, 16):
    kG = G.scalar_mul(k)
    assert kG.is_on_curve()

  # Test that 8*G is on curve (cofactor related)
  G8 = G.scalar_mul(8)
  assert G8.is_on_curve()


def test_ed25519_basic():
  """Test Ed25519 basic operations"""
  # Create curve and generator
  G = Ed25519.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_ed25519_field_properties():
  """Test Ed25519 field properties"""
  curve = Ed25519()

  # Verify field prime (2^255 - 19)
  expected_p = 2**255 - 19
  assert curve.field.P == expected_p


if __name__ == "__main__":
  test_curve25519_basic()
  test_curve25519_order()
  test_curve25519_small_multiples()
  test_ed25519_basic()
  test_ed25519_field_properties()
  print("Curve25519 and Ed25519 tests passed!")
