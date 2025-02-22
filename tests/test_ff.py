from algebra.ff.m31 import M31


def test_mul_power_two_is_correct():
  a = 3
  k = 2
  expected_result = M31(a) * M31(2) ** k
  result = M31(a * 2**k)
  assert result == expected_result


def test_mul_power_two_is_correct_2():
  a = 229287
  k = 4
  expected_result = M31(a) * M31(2) ** k
  result = M31(a * 2**k)
  assert result == expected_result


def test_pow_2_is_correct():
  a = 3
  order = 12
  result = M31(a ** (2**order))
  print(result)
  expected_result = M31(a) ** 4096
  assert result == expected_result


def test_addition():
  a = M31(5)
  b = M31(10)
  expected_result = M31(15)
  assert a + b == expected_result


def test_subtraction():
  a = M31(10)
  b = M31(5)
  expected_result = M31(5)
  assert a - b == expected_result


def test_multiplication():
  a = M31(3)
  b = M31(4)
  expected_result = M31(12)
  assert a * b == expected_result


def test_division():
  a = M31(10)
  b = M31(2)
  expected_result = M31(5)
  assert a / b == expected_result


def test_negation():
  a = M31(5)
  expected_result = M31(2**31 - 6)  # Since M31 is mod 2^31 - 1
  assert -a == expected_result


def test_exponentiation():
  a = M31(2)
  k = 10
  expected_result = M31(1024)  # 2^10
  assert a**k == expected_result


def test_inverse():
  a = M31(3)
  expected_result = M31(1431655765)
  print(a.inv())
  print(expected_result)
  assert a.inv() == expected_result


def test_mul_by_inv():
  x = 3476715743
  assert (M31(x).inv() * M31(x)) == M31(1)


def test_div_1():
  assert (M31(2) / M31(1)) == M31(2)


def test_div_4_2():
  assert (M31(4) / M31(2)) == M31(2)


def test_div_4_3():
  expected_result = M31(1431655766)
  assert (M31(4) / M31(3)) == expected_result


def test_mul_div_2exp_u64():
  # 1 * 2^0 = 1.
  assert M31(1 * 2**0) == M31(1)
  # 5 * 2^2 = 20.
  assert M31(5 * 2**2) == M31(20)
  # 2 * 2^30 = 2^31 = 1.
  assert M31(2 * 2**30) == M31(1)

  # 1 / 2^0 = 1.
  assert M31(1 / 2**0) == M31(1)
  # 2 / 2^0 = 2.
  assert M31(2 / 2**0) == M31(2)
  # 32 / 2^5 = 1.
  assert M31(32 / 2**5) == M31(1)
