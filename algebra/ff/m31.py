from algebra.ff.prime_field import PrimeField
from tinygrad.tensor import Tensor


class M31(PrimeField):
  # Mersenne31 prime: 2^31 - 1
  P = 2147483647

  @classmethod
  def mod_py_obj(cls, inp):
    if isinstance(inp, Tensor):
      return inp % cls.P
    elif isinstance(inp, int):
      return inp % cls.P
    elif isinstance(inp, float):
      return int(inp) % cls.P
    else:
      return [cls.mod_py_obj(x) for x in inp]

  @classmethod
  def modinv_impl(cls, x: Tensor) -> Tensor:
    o = x
    pow_of_x = cls.mul_mod(x, x)
    for _ in range(29):
      pow_of_x = cls.mul_mod(pow_of_x, pow_of_x)
      o = cls.mul_mod(o, pow_of_x)
    return o
