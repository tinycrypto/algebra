# m31.py

import numpy as np
from tinygrad.tensor import Tensor

from algebra.ff.bigint import bigints_to_tensor

modulus = 2**31 - 1  # M31


class M31:
  """
  Represents a scalar field (mod 2^31-1) in a tinygrad Tensor
  """

  def __init__(self, x):
      if isinstance(x, M31):
          self.value = x.value
      elif isinstance(x, Tensor):
          self.value = mod31(x)
      elif isinstance(x, int):
          x = x % modulus
          x = np.array(x, dtype=np.uint32)
          self.value = mod31(Tensor(x, requires_grad=False))
      elif isinstance(x, float):
          x = int(x % modulus)
          x = np.array(x, dtype=np.uint32)
          self.value = mod31(Tensor(x, requires_grad=False))
      elif isinstance(x, list):
          x = np.array([xx % modulus for xx in x], dtype=np.uint32)
          self.value = mod31(Tensor(x, requires_grad=False))
      else:
          x = np.array(x, dtype=np.uint32)
          self.value = mod31(Tensor(x, requires_grad=False))

  def __add__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(add_mod(self.value, other.value))

  __radd__ = __add__

  def __neg__(self):
    return M31(mod31(modulus - self.value))

  def __sub__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(sub_mod(self.value, other.value))

  def __mul__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return M31(mul_mod(self.value, other.value))

  __rmul__ = __mul__

  def __pow__(self, exponent):
    result = M31(1)
    base = self
    while exponent:
        if exponent & 1:
            result = result * base
        base = base * base
        exponent //= 2
    return result

  def inv(self):
    return M31(modinv(self.value))

  def __truediv__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return self * other.inv()

  def __eq__(self, other):
    if isinstance(other, int):
      other = M31(other)
    return eq_t(self.value, other.value)

  def tobytes(self):
    return tobytes(self.value)

  def __repr__(self):
    return f"M31(shape={self.value.shape}, data={self.value.numpy()})"


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



def mod31(x: Tensor) -> Tensor:
  """
  x mod (2^31 - 1), done in float64.
  """
  return x % modulus


def add_mod(a: Tensor, b: Tensor) -> Tensor:
  return mod31(a + b)


def sub_mod(a: Tensor, b: Tensor) -> Tensor:
  return mod31(a - b)


def mul_mod(a: Tensor, b: Tensor) -> Tensor:
  return mod31(a * b)


def sum_mod(x: Tensor, axis=None) -> Tensor:
  return mod31(x.sum(axis=axis))


def zeros(shape):
  return Tensor.zeros(*shape, dtype=np.float32)


def arange(*args):
  return t32(np.arange(*args, dtype=np.uint32))


def append(*args, axis=0):
  return Tensor.cat(args, dim=axis)


def tobytes(x: Tensor) -> bytes:
  arr = x.detach().numpy()
  arr = np.mod(arr, modulus).astype(np.uint32)
  return arr.tobytes()


def eq_t(x: Tensor, y: Tensor) -> bool:
  xnp = np.mod(x.detach().numpy(), modulus)
  ynp = np.mod(y.detach().numpy(), modulus)
  return np.array_equal(xnp, ynp)


def iszero(x: Tensor) -> bool:
  arr = np.mod(x.detach().numpy(), modulus)
  return np.all(arr == 0)


def zeros_like(x: Tensor):
  return Tensor.zeros(*x.shape, dtype=np.float64)


def array(pyobj):
  return t32(pyobj)
