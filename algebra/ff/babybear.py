from algebra.ff.prime_field import PrimeField
from tinygrad.tensor import Tensor


class BabyBear(PrimeField):
  # BabyBear prime: 2^31 - 2^27 + 1
  P = 2013265921

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
    # Compute the modular inverse using Fermat's little theorem:
    #   x^(P-2) mod P.
    return cls.pow_tensor(x, cls.P - 2)
