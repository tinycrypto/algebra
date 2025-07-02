"""Tests for secp256k1 curve (Bitcoin/Ethereum)"""

from algebra.ec.secp256k1 import Secp256k1, Fr


def test_secp256k1_basic():
  """Test secp256k1 basic operations"""
  # Create curve and generator
  G = Secp256k1.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_secp256k1_known_values():
  """Test secp256k1 with known test vectors"""
  G = Secp256k1.generator()

  # Test known doubling result
  expected_2G_x = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
  expected_2G_y = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A

  actual_2G = G.double()
  assert int(actual_2G.x) == expected_2G_x
  assert int(actual_2G.y) == expected_2G_y


def test_secp256k1_order():
  """Test secp256k1 curve order"""
  G = Secp256k1.generator()
  n = Fr.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_secp256k1_properties():
  """Test secp256k1 mathematical properties"""
  G = Secp256k1.generator()

  # Test distributivity: (a + b)*G = a*G + b*G
  a, b = 123, 456
  ab_G = G.scalar_mul(a + b)
  aG_plus_bG = G.scalar_mul(a) + G.scalar_mul(b)
  assert ab_G.equals(aG_plus_bG)

  # Test associativity: (a*b)*G = a*(b*G)
  ab = a * b
  ab_G = G.scalar_mul(ab)
  a_bG = G.scalar_mul(a).scalar_mul(b)  # This won't work directly
  # Instead test: a*(b*G) by computing b*G first
  bG = G.scalar_mul(b)
  a_bG = bG.scalar_mul(a)
  assert ab_G.equals(a_bG)


if __name__ == "__main__":
  test_secp256k1_basic()
  test_secp256k1_known_values()
  test_secp256k1_order()
  test_secp256k1_properties()
  print("secp256k1 tests passed!")
