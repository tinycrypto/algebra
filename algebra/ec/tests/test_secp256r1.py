"""Tests for secp256r1 (P-256) curve"""

from algebra.ec.secp256r1 import Secp256r1, Fr


def test_secp256r1_basic():
  """Test secp256r1 basic operations"""
  # Create curve and generator
  G = Secp256r1.generator()
  assert G.is_on_curve()

  # Test basic operations
  G2 = G.double()
  assert G2.is_on_curve()

  G3 = G + G2
  assert G3.is_on_curve()
  assert G3.equals(G.scalar_mul(3))


def test_secp256r1_known_values():
  """Test secp256r1 with known test vectors"""
  G = Secp256r1.generator()

  # Test known doubling result
  expected_2G_x = 0x7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978
  expected_2G_y = 0x07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1

  actual_2G = G.double()
  assert actual_2G.x.value.item() == expected_2G_x
  assert actual_2G.y.value.item() == expected_2G_y


def test_secp256r1_order():
  """Test secp256r1 curve order"""
  G = Secp256r1.generator()
  n = Fr.P
  nG = G.scalar_mul(n)
  assert nG.is_infinity()


def test_secp256r1_nist_compliance():
  """Test NIST P-256 specific properties"""
  curve = Secp256r1()

  # Verify curve parameters
  assert curve.a.value.item() == -3
  expected_b = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
  assert curve.b.value.item() == expected_b

  # Verify field prime
  expected_p = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
  assert curve.field.P == expected_p


if __name__ == "__main__":
  test_secp256r1_basic()
  test_secp256r1_known_values()
  test_secp256r1_order()
  test_secp256r1_nist_compliance()
  print("secp256r1 tests passed!")
