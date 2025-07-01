"""Test runner for EC module tests"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


def run_test_module(module_name):
  """Run a specific test module"""
  try:
    print(f"\n{'=' * 50}")
    print(f"Running {module_name}")
    print("=" * 50)

    module = __import__(f"algebra.ec.tests.{module_name}", fromlist=[module_name])

    # Run the module's main block
    if hasattr(module, "__main__") and callable(getattr(module, "__main__", None)):
      module.__main__()
    else:
      # Try to find and run test functions
      test_functions = [getattr(module, name) for name in dir(module) if name.startswith("test_") and callable(getattr(module, name))]

      for test_func in test_functions:
        try:
          print(f"  Running {test_func.__name__}...")
          test_func()
          print(f"  ✓ {test_func.__name__} passed")
        except Exception as e:
          print(f"  ✗ {test_func.__name__} failed: {e}")
          return False

      if test_functions:
        print(f"✓ All {len(test_functions)} tests in {module_name} passed")
      else:
        print(f"No test functions found in {module_name}")

    return True

  except Exception as e:
    print(f"✗ Failed to run {module_name}: {e}")
    return False


def main():
  """Run all EC tests"""
  print("Running Elliptic Curve Tests")
  print("=" * 50)

  # Test modules to run
  test_modules = [
    "test_registry",
    "test_bn254",
    "test_secp256k1",
    "test_secp256r1",
    "test_curve25519",
    "test_nist_curves",
    "test_interoperability",
  ]

  passed = 0
  failed = 0

  for module in test_modules:
    if run_test_module(module):
      passed += 1
    else:
      failed += 1

  print(f"\n{'=' * 50}")
  print(f"Test Summary: {passed} passed, {failed} failed")
  print("=" * 50)

  if failed == 0:
    print("🎉 All tests passed!")
    return 0
  else:
    print(f"❌ {failed} test modules failed")
    return 1


if __name__ == "__main__":
  sys.exit(main())
