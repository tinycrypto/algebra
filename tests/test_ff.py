from algebra.ff.test_field import MyPrimeField


def test_from_bigint():
  a = MyPrimeField(5)
  b = MyPrimeField(3)
  assert (a + b).value == 8
