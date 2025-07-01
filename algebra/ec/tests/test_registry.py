"""Tests for curve registry"""

import pytest
from algebra.ec.registry import get_curve, list_curves, CURVES


def test_registry_basic():
  """Test basic registry functionality"""
  # Test that we can get curves by name
  secp256k1_class = get_curve("secp256k1")
  assert secp256k1_class is not None

  # Test aliases
  bitcoin_class = get_curve("bitcoin")
  assert bitcoin_class is not None
  assert bitcoin_class == secp256k1_class

  # Test case insensitivity
  p256_class = get_curve("P256")
  assert p256_class is not None


def test_registry_list():
  """Test registry listing"""
  curves = list_curves()
  assert len(curves) > 0
  assert "secp256k1" in curves
  assert "bn254" in curves
  assert "p256" in curves


def test_registry_aliases():
  """Test that aliases work correctly"""
  # Bitcoin/Ethereum should map to secp256k1
  assert get_curve("bitcoin") == get_curve("secp256k1")
  assert get_curve("ethereum") == get_curve("secp256k1")

  # NIST curve aliases
  assert get_curve("p256") == get_curve("secp256r1")
  assert get_curve("p384") == get_curve("secp384r1")
  assert get_curve("p521") == get_curve("secp521r1")


def test_registry_invalid():
  """Test invalid curve names"""
  assert get_curve("nonexistent") is None
  assert get_curve("") is None


def test_registry_instantiation():
  """Test that registry returns valid curve classes"""
  for name in ["secp256k1", "secp256r1", "bn254"]:
    curve_class = get_curve(name)
    assert curve_class is not None

    # Test that we can create instances
    curve = curve_class()
    generator = curve_class.generator()
    assert generator.is_on_curve()


if __name__ == "__main__":
  test_registry_basic()
  test_registry_list()
  test_registry_aliases()
  test_registry_invalid()
  test_registry_instantiation()
  print("Registry tests passed!")
