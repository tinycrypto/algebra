# m31.py

import numpy as np
from tinygrad.tensor import Tensor

from algebra.ff.bigint import bigints_to_tensor

# The Mersenne31 prime (2 ** 31 - 1)
P = 2147483647


def mod31_py_obj(inp):
  if isinstance(inp, Tensor):
    return inp % P
  elif isinstance(inp, int):
    return inp % P
  elif isinstance(inp, float):
    return int(inp) % P
  else:
    return [mod31_py_obj(x) for x in inp]


class M31:
  """
  The prime field `F_p` where `p = 2^31 - 1`.
  """

  def __init__(self, x):
    if isinstance(x, (int, float, list, Tensor)):
      x = array(mod31_py_obj(x))
    elif isinstance(x, M31):
      x = x.value
    print(x)
    assert (x.numpy() >> 31) == 0, "Value exceeds 32-bit unsigned integer range"
    self.value = x

  def __add__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(add(self.value, other.value))

  def __neg__(self):
    return M31(P - self.value)

  def __sub__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(sub(self.value, other.value))

  def __mul__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(mul_mod(self.value, other.value))

  def __pow__(self, exponent):
    assert isinstance(exponent, int)
    result = M31(1)
    base = self
    while exponent:
      if exponent & 1:
        result = result * base
      base = base * base
      exponent //= 2
    return result

  def inv(self) -> "M31":
    assert not iszero(self.value), "0 has no inverse"
    return M31(modinv(self.value))

  def __truediv__(self, other):
    if isinstance(other, int):
      other = M31(other)
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
    return ((self.value % P).numpy()).tobytes()

  def __eq__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return eq_t(self.value, other.value)


# tensor util


def add(a: Tensor, b: Tensor) -> Tensor:
  return a + b


def sub(a: Tensor, b: Tensor) -> Tensor:
  return a + P - b


def mul_mod(a: Tensor, b: Tensor) -> Tensor:
  return (a * b) % P


def sum_mod(x: Tensor, axis=None) -> Tensor:
  return (x.sum(axis=axis)) % P


def zeros(shape):
  return Tensor.zeros(*shape, dtype=np.float32)


def arange(*args):
  return t32(np.arange(*args, dtype=np.uint32))


def append(*args, axis=0):
  return Tensor.cat(args, dim=axis)


def tobytes(x: Tensor) -> bytes:
  arr = x.detach().numpy()
  arr = np.mod(arr, P).astype(np.uint32)
  return arr.tobytes()


def eq_t(x: Tensor, y: Tensor) -> bool:
  xnp = np.mod(x.detach().numpy(), P)
  ynp = np.mod(y.detach().numpy(), P)
  return np.array_equal(xnp, ynp)


def iszero(x: Tensor) -> bool:
  arr = np.mod(x.detach().numpy(), P)
  return np.all(arr == 0)


def zeros_like(x: Tensor):
  return Tensor.zeros(*x.shape, dtype=np.float64)


def array(pyobj):
  return t32(pyobj)


def modinv(x: Tensor) -> Tensor:
  o = x
  pow_of_x = mul_mod(x, x)
  for _ in range(29):
    pow_of_x = mul_mod(pow_of_x, pow_of_x)
    o = mul_mod(o, pow_of_x)
  return o


def t32(x) -> Tensor:
  if isinstance(x, Tensor):
    return x
  try:
    arr = np.array(x, dtype=np.int64)
  except OverflowError:
    chunks_arr = bigints_to_tensor(x, chunk_bits=32, num_chunks=8)
    return Tensor(chunks_arr, requires_grad=False)
  return Tensor(arr, requires_grad=False)
