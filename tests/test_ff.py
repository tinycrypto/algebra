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
