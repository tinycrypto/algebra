"""Simple tests that don't require heavy computation"""


def test_registry_imports():
  """Test that we can import the registry without computation"""
  from algebra.ec.registry import CURVES, get_curve, list_curves

  # Test basic registry structure
  assert isinstance(CURVES, dict)
  assert len(CURVES) > 0
  assert "secp256k1" in CURVES
  assert "bitcoin" in CURVES

  # Test functions exist
  assert callable(get_curve)
  assert callable(list_curves)

  print("Registry imports successful")


def test_curve_imports():
  """Test that we can import all curve classes"""
  from algebra.ec.secp256k1 import Secp256k1
  from algebra.ec.secp256r1 import Secp256r1
  from algebra.ec.secp384r1 import Secp384r1
  from algebra.ec.secp521r1 import Secp521r1
  from algebra.ec.bn254 import G1 as BN254

  # Test that classes exist
  assert Secp256k1 is not None
  assert Secp256r1 is not None
  assert Secp384r1 is not None
  assert Secp521r1 is not None
  assert BN254 is not None

  print("Curve imports successful")


def test_curve_constants():
  """Test curve constants without instantiation"""
  from algebra.ec.secp256k1 import Fp, Fr

  # Test that field constants are correct
  assert Fp.P == 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
  assert Fr.P == 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141

  print("Curve constants correct")


def test_registry_functionality():
  """Test registry without creating curve instances"""
  from algebra.ec.registry import get_curve, list_curves

  # Test get_curve returns classes
  secp256k1_class = get_curve("secp256k1")
  assert secp256k1_class is not None

  bitcoin_class = get_curve("bitcoin")
  assert bitcoin_class == secp256k1_class

  # Test list_curves
  curves = list_curves()
  assert len(curves) > 5
  assert "secp256k1" in curves
  assert "bitcoin" in curves

  print("Registry functionality working")


if __name__ == "__main__":
  test_registry_imports()
  test_curve_imports()
  test_curve_constants()
  test_registry_functionality()
  print("All simple tests passed!")
