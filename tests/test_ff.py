# M31 test

import pytest
from algebra.ff.m31 import M31


def test_basic_ops_m31():
  import random

  random.seed(0)
  P = 2147483647  # 2 ** 31 - 1
  for _ in range(100):
    x = random.randint(0, 2**31 - 1)
    y = random.randint(0, 2**31 - 1)
    assert M31((x + y) % P) == M31(x) + M31(y)
    assert M31((x * y) % P) == M31(x) * M31(y)
    if x == 0:
      assert M31(0) == -M31(x)
    else:
      assert M31(P - x) == -M31(x)


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


def test_division():
  a = M31(10)
  b = M31(2)
  expected_result = M31(5)
  assert a / b == expected_result


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


def test_inverse_zero():
  a = M31(0)
  with pytest.raises(AssertionError, match="0 has no inverse"):
    a.inv()


# BabyBear test

from algebra.ff.babybear import BabyBear


def test_basic_ops_babybear():
  import random

  random.seed(0)
  P = 2013265921  # (1 << 31) - (1 << 27) + 1
  for _ in range(100):
    x = random.randint(0, 2**31 - 1)
    y = random.randint(0, 2**31 - 1)
    assert BabyBear((x + y) % P) == BabyBear(x) + BabyBear(y)
    assert BabyBear((x * y) % P) == BabyBear(x) * BabyBear(y)
    if x == 0:
      assert BabyBear(0) == -BabyBear(x)
    else:
      assert BabyBear(P - x) == -BabyBear(x)


def test_mul_power_two_is_correct_babybear():
  a = 3
  k = 2
  expected_result = BabyBear(a) * BabyBear(2) ** k
  result = BabyBear(a * 2**k)
  assert result == expected_result


def test_mul_power_two_is_correct_2_babybear():
  a = 229287
  k = 4
  expected_result = BabyBear(a) * BabyBear(2) ** k
  result = BabyBear(a * 2**k)
  assert result == expected_result


def test_pow_2_is_correct_babybear():
  a = 3
  order = 12  # 2^12 = 4096
  result = BabyBear(a ** (2**order))
  expected_result = BabyBear(a) ** (2**order)
  assert result == expected_result


def test_division_babybear():
  a = BabyBear(10)
  b = BabyBear(2)
  expected_result = BabyBear(5)
  assert a / b == expected_result


def test_exponentiation_babybear():
  a = BabyBear(2)
  k = 10
  expected_result = BabyBear(1024)  # since 2^10 = 1024 (and 1024 < P)
  assert a**k == expected_result


def test_inverse_babybear():
  a = BabyBear(3)
  expected_result = BabyBear(1342177281)
  assert a.inv() == expected_result


def test_mul_by_inv_babybear():
  x = 3476715743
  assert (BabyBear(x).inv() * BabyBear(x)) == BabyBear(1)


def test_div_1_babybear():
  assert (BabyBear(2) / BabyBear(1)) == BabyBear(2)


def test_div_4_2_babybear():
  assert (BabyBear(4) / BabyBear(2)) == BabyBear(2)


def test_div_4_3_babybear():
  # (4 * inv(3)) mod P.
  expected_result = BabyBear(1342177282)
  assert (BabyBear(4) / BabyBear(3)) == expected_result


def test_mul_div_2exp_u64_babybear():
  # 1 * 2^0 = 1.
  assert BabyBear(1 * 2**0) == BabyBear(1)
  # 5 * 2^2 = 20.
  assert BabyBear(5 * 2**2) == BabyBear(20)
  # 2 * 2^30: 2^30 = 1073741824, 2 * 1073741824 = 2147483648.
  # 2147483648 mod P = 2147483648 - P = 2147483648 - 2013265921 = 134217727.
  assert BabyBear(2 * 2**30) == BabyBear(134217727)

  assert BabyBear(1 / 2**0) == BabyBear(1)
  assert BabyBear(2 / 2**0) == BabyBear(2)
  # 32 / 2^5 = 32 / 32 = 1.
  assert BabyBear(32 / 2**5) == BabyBear(1)
