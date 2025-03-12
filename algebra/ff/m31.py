from algebra.ff.prime_field import PrimeField


class M31(PrimeField):
  # Mersenne31 prime: 2^31 - 1
  P = 2**31 - 1
  w = 7