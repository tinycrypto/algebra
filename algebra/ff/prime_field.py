import numpy as np
from tinygrad.tensor import Tensor


class PrimeField:
  P = None

  def __init__(self, x):
    if isinstance(x, (int, float, list, Tensor)):
      x = self.t32(self.mod_py_obj(x))
    elif isinstance(x, PrimeField):
      x = x.value
    self.value = x

  def __add__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return type(self)(self.add(self.value, other.value))

  def __neg__(self):
    return type(self)(self.P - self.value)

  def __sub__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return type(self)(self.sub(self.value, other.value))

  def __mul__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return type(self)(self.mul_mod(self.value, other.value))

  def __pow__(self, exponent):
    assert isinstance(exponent, int)
    result = type(self)(1)
    base = self
    while exponent:
      if exponent & 1:
        result = result * base
      base = base * base
      exponent //= 2
    return result

  def inv(self):
    assert not self.iszero(self.value), "0 has no inverse"
    return type(self)(self.modinv_impl(self.value))

  def __truediv__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return self * other.inv()

  def __rtruediv__(self, other):
    return self.inv() * other

  __radd__ = __add__
  __rmul__ = __mul__

  def __rsub__(self, other):
    return -(self - other)

  def __repr__(self):
    return f"{self.value.numpy()}"

  def __int__(self):
    return int(self.value)

  def __len__(self):
    return len(self.value)

  def tobytes(self):
    return ((self.value % self.P).numpy()).tobytes()

  def __eq__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return self.eq_t(self.value, other.value)

  # -- Common arithmetic utility methods --
  @classmethod
  def add(cls, a: Tensor, b: Tensor) -> Tensor:
    return a + b

  @classmethod
  def sub(cls, a: Tensor, b: Tensor) -> Tensor:
    return a + (cls.P - b)

  @classmethod
  def mul_mod(cls, a: Tensor, b: Tensor) -> Tensor:
    return (a * b) % cls.P

  @classmethod
  def sum_mod(cls, x: Tensor, axis=None) -> Tensor:
    return (x.sum(axis=axis)) % cls.P

  @classmethod
  def zeros(cls, shape):
    return Tensor.zeros(*shape, dtype=np.float32)

  @staticmethod
  def append(*args, axis=0):
    return Tensor.cat(args, dim=axis)

  @classmethod
  def tobytes_tensor(cls, x: Tensor) -> bytes:
    arr = x.detach().numpy()
    arr = np.mod(arr, cls.P).astype(np.uint32)
    return arr.tobytes()

  @classmethod
  def eq_t(cls, x: Tensor, y: Tensor) -> bool:
    xnp = np.mod(x.detach().numpy(), cls.P)
    ynp = np.mod(y.detach().numpy(), cls.P)
    return np.array_equal(xnp, ynp)

  @classmethod
  def iszero(cls, x: Tensor) -> bool:
    arr = np.mod(x.detach().numpy(), cls.P)
    return np.all(arr == 0)

  @staticmethod
  def zeros_like(x: Tensor):
    return Tensor.zeros(*x.shape, dtype=np.float64)

  @staticmethod
  def t32(x) -> Tensor:
    if isinstance(x, Tensor):
      return x
    return Tensor(np.array(x, dtype=np.int32), requires_grad=False)

  # -- Unique methods: to be implemented by subclasses --
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

  @classmethod
  def pow_tensor(cls, base: Tensor, exponent: int) -> Tensor:
    result = cls.t32(1)
    while exponent:
      if exponent & 1:
        result = cls.mul_mod(result, base)
      base = cls.mul_mod(base, base)
      exponent //= 2
    return result
