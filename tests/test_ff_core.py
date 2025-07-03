"""Core finite field tests - minimal but comprehensive"""

import pytest
from algebra.ff.m31 import M31
from algebra.ff.babybear import BabyBear
from algebra.ff.bigint_field import BigIntPrimeField
import random


def test_m31_basic_operations():
  """Test M31 field basic operations"""
  # Basic arithmetic
  assert M31(10) + M31(20) == M31(30)
  assert M31(30) - M31(20) == M31(10)
  assert M31(5) * M31(6) == M31(30)

  # Modular reduction
  p = M31.P
  assert M31(p) == M31(0)
  assert M31(p + 1) == M31(1)
  assert M31(p - 1) + M31(2) == M31(1)

  # Division and inverse
  a = M31(7)
  a_inv = a.inv()
  assert a * a_inv == M31(1)

  # Random tests
  random.seed(42)
  for _ in range(10):
    x = random.randint(1, p - 1)
    y = random.randint(1, p - 1)

    a, b = M31(x), M31(y)
    # Test field axioms
    assert a + b == b + a  # Commutative
    assert (a * b) * b.inv() == a  # Division
    assert a - a == M31(0)  # Subtraction


def test_babybear_operations():
  """Test BabyBear field operations"""
  p = BabyBear.P

  # Test with larger values
  a = BabyBear(1000000)
  b = BabyBear(2000000)
  c = a + b
  assert int(c) == 3000000

  # Test modular wrap
  large = BabyBear(p - 100)
  result = large + BabyBear(200)
  assert int(result) == 100

  # Test negative values (implementation may vary)
  neg = BabyBear(-1)
  # Some implementations may not reduce negatives immediately
  # The important thing is that operations work correctly
  assert neg + BabyBear(1) == BabyBear(0)

  # Test power
  base = BabyBear(3)
  assert base**4 == BabyBear(81)
  assert base**p == base  # Fermat's little theorem


def test_bigint_field_operations():
  """Test BigInt field for large primes"""

  # Define secp256k1 field
  class Secp256k1(BigIntPrimeField):
    P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F

  # Test basic ops
  a = Secp256k1(12345)
  b = Secp256k1(67890)

  assert int(a + b) == 80235
  assert int(a * b) == 838102050

  # Test near modulus
  near_p = Secp256k1(Secp256k1.P - 1)
  assert int(near_p + Secp256k1(1)) == 0
  assert int(near_p + Secp256k1(2)) == 1

  # Test inverse for large prime
  x = Secp256k1(0x1234567890ABCDEF)
  x_inv = x.inv()
  assert int(x * x_inv) == 1


def test_field_properties():
  """Test field mathematical properties"""

  # Test with small field for exhaustive check
  class F13(BigIntPrimeField):
    P = 13

  # Check that every non-zero element has inverse
  for i in range(1, 13):
    elem = F13(i)
    inv = elem.inv()
    assert elem * inv == F13(1)

  # Check order of multiplicative group
  g = F13(2)  # Generator
  order = 1
  current = g
  while current != F13(1):
    current = current * g
    order += 1
  assert order == 12  # φ(13) = 12


def test_edge_cases():
  """Test edge cases and error conditions"""

  # Division by zero
  with pytest.raises((AssertionError, ValueError)):
    M31(0).inv()

  # Large exponents
  a = M31(2)
  assert a**100 == M31(pow(2, 100, M31.P))

  # Zero operations
  zero = M31(0)
  assert zero + M31(5) == M31(5)
  assert zero * M31(5) == M31(0)
  assert -zero == zero


if __name__ == "__main__":
  test_m31_basic_operations()
  test_babybear_operations()
  test_bigint_field_operations()
  test_field_properties()
  test_edge_cases()
  print("All core field tests passed!")
