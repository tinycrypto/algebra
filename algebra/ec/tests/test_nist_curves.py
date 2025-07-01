"""Tests for NIST curves (P-384, P-521)"""

from algebra.ec.secp384r1 import Secp384r1, Fr as Fr384
from algebra.ec.secp521r1 import Secp521r1, Fr as Fr521


def test_secp384r1_basic():
  """Test secp384r1 (P-384) basic operations"""
  # Create curve and generator
  G = Secp384r1.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_secp384r1_order():
  """Test secp384r1 curve order"""
  G = Secp384r1.generator()
  n = Fr384.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_secp384r1_properties():
  """Test secp384r1 specific properties"""
  curve = Secp384r1()

  # Verify curve parameters
  assert curve.a.value.item() == -3

  # Test larger scalar multiplication
  G = Secp384r1.generator()
  G100 = G.scalar_mul(100)
  assert G100.is_on_curve()


def test_secp521r1_basic():
  """Test secp521r1 (P-521) basic operations"""
  # Create curve and generator
  G = Secp521r1.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_secp521r1_order():
  """Test secp521r1 curve order"""
  G = Secp521r1.generator()
  n = Fr521.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_secp521r1_properties():
  """Test secp521r1 specific properties"""
  curve = Secp521r1()

  # Verify curve parameters
  assert curve.a.value.item() == -3

  # Test larger scalar multiplication
  G = Secp521r1.generator()
  G1000 = G.scalar_mul(1000)
  assert G1000.is_on_curve()


def test_nist_curve_security_levels():
  """Test that NIST curves have expected bit lengths"""
  # P-384 should have ~384-bit field
  curve384 = Secp384r1()
  p384 = curve384.field.P
  assert p384.bit_length() == 384

  # P-521 should have ~521-bit field
  curve521 = Secp521r1()
  p521 = curve521.field.P
  assert p521.bit_length() == 521


if __name__ == "__main__":
  test_secp384r1_basic()
  test_secp384r1_order()
  test_secp384r1_properties()
  test_secp521r1_basic()
  test_secp521r1_order()
  test_secp521r1_properties()
  test_nist_curve_security_levels()
  print("NIST curves tests passed!")
