"""Tests for BigInt implementation"""

import pytest
from algebra.bigint import BigInt, mod_inverse, gcd, lcm


def test_basic_arithmetic():
  """Test basic arithmetic operations"""
  a = BigInt(12345)
  b = BigInt(67890)

  # Addition
  assert (a + b).to_int() == 12345 + 67890

  # Subtraction
  assert (b - a).to_int() == 67890 - 12345
  assert (a - b).to_int() == 12345 - 67890

  # Multiplication
  assert (a * b).to_int() == 12345 * 67890

  # With Python ints
  assert (a + 100).to_int() == 12345 + 100
  assert (a - 100).to_int() == 12345 - 100
  assert (a * 10).to_int() == 12345 * 10


def test_large_numbers():
  """Test with large numbers"""
  a = BigInt(10**20)
  b = BigInt(10**20)

  assert (a + b).to_int() == 2 * 10**20
  assert (a * b).to_int() == 10**40

  # Test Karatsuba kicks in
  c = BigInt(2**256 - 1)
  d = BigInt(2**256 + 1)
  result = c * d
  expected = (2**256 - 1) * (2**256 + 1)
  assert result.to_int() == expected


def test_division():
  """Test division and modulo"""
  a = BigInt(100)
  b = BigInt(7)

  q, r = divmod(a, b)
  assert q.to_int() == 14
  assert r.to_int() == 2

  assert (a // b).to_int() == 14
  assert (a % b).to_int() == 2

  # Test with negative numbers
  c = BigInt(-100)
  q, r = divmod(c, b)
  assert q.to_int() * 7 + r.to_int() == -100


def test_power():
  """Test exponentiation"""
  a = BigInt(2)

  assert (a**10).to_int() == 1024
  assert (a**0).to_int() == 1

  # Modular exponentiation
  b = BigInt(3)
  mod = BigInt(7)
  assert pow(b, 4, mod).to_int() == pow(3, 4, 7)


def test_shifts():
  """Test bit shifting"""
  a = BigInt(1234)

  # Left shift
  assert (a << 10).to_int() == 1234 << 10

  # Right shift
  b = a << 10
  assert (b >> 10).to_int() == 1234

  # Edge cases
  assert (a << 0).to_int() == 1234
  assert (a >> 0).to_int() == 1234
  assert (a >> 100).to_int() == 0


def test_comparisons():
  """Test comparison operations"""
  a = BigInt(100)
  b = BigInt(200)
  c = BigInt(100)

  assert a < b
  assert b > a
  assert a <= c
  assert a >= c
  assert a == c
  assert a != b

  # With Python ints
  assert a < 200
  assert a > 50
  assert a == 100


def test_unary_operations():
  """Test unary operations"""
  a = BigInt(100)
  b = BigInt(-100)

  assert (-a).to_int() == -100
  assert (-b).to_int() == 100
  assert abs(a).to_int() == 100
  assert abs(b).to_int() == 100
  assert int(a) == 100


def test_edge_cases():
  """Test edge cases"""
  # Zero
  zero = BigInt(0)
  one = BigInt(1)

  assert (zero + zero).to_int() == 0
  assert (zero * one).to_int() == 0
  assert (one - one).to_int() == 0

  # Division by zero
  with pytest.raises(ValueError):
    one // zero

  # Negative exponent
  with pytest.raises(ValueError):
    one**-1


def test_gcd_lcm():
  """Test GCD and LCM operations"""
  a = BigInt(48)
  b = BigInt(18)

  # GCD
  g = gcd(a, b)
  assert g.to_int() == 6

  # LCM
  lcm_result = lcm(a, b)
  assert lcm_result.to_int() == 144

  # GCD with larger numbers
  c = BigInt(12345678)
  d = BigInt(87654321)
  g2 = gcd(c, d)
  # Verify using Python's math.gcd
  import math

  assert g2.to_int() == math.gcd(12345678, 87654321)


def test_modular_inverse():
  """Test modular inverse"""
  # Test with prime modulus
  a = BigInt(3)
  n = BigInt(7)

  inv = mod_inverse(a, n)
  assert (a * inv % n).to_int() == 1

  # Test with composite modulus (coprime)
  a = BigInt(5)
  n = BigInt(12)

  inv = mod_inverse(a, n)
  assert (a * inv % n).to_int() == 1

  # Test error case
  with pytest.raises(ValueError):
    mod_inverse(BigInt(0), n)
