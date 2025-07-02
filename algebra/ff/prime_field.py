from tinygrad.tensor import Tensor
from tinygrad import dtypes
from algebra.bigint.bigint import BigInt


class PrimeField:
  P: int = None
  w: int = None

  def __init__(self, x):
    if isinstance(x, (int, float, list, Tensor)):
      # Use BigInt to reduce large values before creating tensor
      if isinstance(x, int) and x >= self.P:
        reduced_val = (BigInt(x) % BigInt(self.P)).to_int()
        x = self.t32(reduced_val)
      else:
        x = self.t32(self.mod_py_obj(x))
    elif isinstance(x, PrimeField):
      x = x.value
    self.value = x

  def __add__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    return type(self)(self.add(self.value, other.value))

  def __neg__(self):
    return type(self)(self.neg(self.value))

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
    zero_tensor = self.iszero(self.value)
    assert not bool(zero_tensor.numpy().item()), "0 has no inverse"
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
    return int(self.value.numpy().item())

  def __len__(self):
    return len(self.value)

  def tobytes(self):
    val = int(self.value.numpy().item()) if self.value.numel() == 1 else int(self.value.numpy()[0])
    result = BigInt(val) % BigInt(self.P)
    return result.to_int().to_bytes((result.to_int().bit_length() + 7) // 8, 'big')

  def __eq__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    result_tensor = self.eq_t(self.value, other.value)
    return bool(result_tensor.numpy().item())

  # -- Common arithmetic utility methods --
  @classmethod
  def add(cls, a: Tensor, b: Tensor) -> Tensor:
    # Use BigInt for large prime arithmetic
    if a.numel() == 1 and b.numel() == 1:
      # Scalar case
      a_val = int(a.numpy().item())
      b_val = int(b.numpy().item())
      result = (BigInt(a_val) + BigInt(b_val)) % BigInt(cls.P)
      return Tensor([result.to_int()], dtype=a.dtype)
    else:
      # Vector case - handle broadcasting for scalar-vector operations
      a_np = a.numpy()
      b_np = b.numpy()
      
      # Handle broadcasting: if one is scalar, broadcast to match the other
      if a_np.ndim == 0:  # a is scalar
        a_np = [a_np.item()] * len(b_np)
      elif b_np.ndim == 0:  # b is scalar  
        b_np = [b_np.item()] * len(a_np)
      
      results = []
      for a_val, b_val in zip(a_np, b_np):
        result = (BigInt(int(a_val)) + BigInt(int(b_val))) % BigInt(cls.P)
        results.append(result.to_int())
      return Tensor(results, dtype=a.dtype)

  @classmethod
  def sub(cls, a: Tensor, b: Tensor) -> Tensor:
    # Use BigInt for large prime arithmetic
    if a.numel() == 1 and b.numel() == 1:
      # Scalar case - use Python's built-in modulo for correct negative handling
      a_val = int(a.numpy().item())
      b_val = int(b.numpy().item())
      result = (a_val - b_val) % cls.P
      return Tensor([result], dtype=a.dtype)
    else:
      # Vector case - handle element-wise subtraction
      a_np = a.numpy()
      b_np = b.numpy()
      results = []
      for a_val, b_val in zip(a_np, b_np):
        result = (int(a_val) - int(b_val)) % cls.P
        results.append(result)
      return Tensor(results, dtype=a.dtype)

  @classmethod
  def neg(cls, a: Tensor) -> Tensor:
    # Use Python's built-in modulo for correct negative handling
    if a.numel() == 1:
      # Scalar case
      a_val = int(a.numpy().item())
      result = (-a_val) % cls.P
      return Tensor([result], dtype=a.dtype)
    else:
      # Vector case - handle element-wise negation
      a_np = a.numpy()
      results = []
      for a_val in a_np:
        result = (-int(a_val)) % cls.P
        results.append(result)
      return Tensor(results, dtype=a.dtype)

  @classmethod
  def mul_mod(cls, a: Tensor, b: Tensor) -> Tensor:
    if a.numel() == 1 and b.numel() == 1:
      # Scalar case
      a_val = int(a.numpy().item())
      b_val = int(b.numpy().item())
      result = (BigInt(a_val) * BigInt(b_val)) % BigInt(cls.P)
      return Tensor([result.to_int()], dtype=a.dtype)
    else:
      # Vector case - handle broadcasting for scalar-vector operations
      a_np = a.numpy()
      b_np = b.numpy()
      
      # Handle broadcasting: if one is scalar, broadcast to match the other
      if a_np.ndim == 0:  # a is scalar
        a_np = [a_np.item()] * len(b_np)
      elif b_np.ndim == 0:  # b is scalar  
        b_np = [b_np.item()] * len(a_np)
      
      results = []
      for a_val, b_val in zip(a_np, b_np):
        result = (BigInt(int(a_val)) * BigInt(int(b_val))) % BigInt(cls.P)
        results.append(result.to_int())
      return Tensor(results, dtype=a.dtype)

  @classmethod
  def sum_mod(cls, x: Tensor, axis=None) -> Tensor:
    # Use BigInt for large prime arithmetic
    x_sum = int(x.sum(axis=axis).numpy().item())
    result = BigInt(x_sum) % BigInt(cls.P)
    return Tensor([result.to_int()], dtype=x.dtype)

  @classmethod
  def zeros(cls, shape):
    return Tensor.zeros(*shape, dtype=dtypes.float32)

  @staticmethod
  def append(*args, axis=0):
    return Tensor.cat(args, dim=axis)

  @classmethod
  def tobytes_tensor(cls, x: Tensor) -> bytes:
    val = int(x.numpy().item()) if x.numel() == 1 else int(x.numpy()[0])
    result = BigInt(val) % BigInt(cls.P)
    return result.to_int().to_bytes((result.to_int().bit_length() + 7) // 8, 'big')

  @classmethod
  def eq_t(cls, x: Tensor, y: Tensor):
    x_val = int(x.numpy().item()) if x.numel() == 1 else int(x.numpy()[0])
    y_val = int(y.numpy().item()) if y.numel() == 1 else int(y.numpy()[0])
    
    x_mod = BigInt(x_val) % BigInt(cls.P)
    y_mod = BigInt(y_val) % BigInt(cls.P)
    return Tensor([1.0 if x_mod == y_mod else 0.0], dtype=dtypes.float32)

  @classmethod
  def iszero(cls, x: Tensor):
    x_val = int(x.numpy().item()) if x.numel() == 1 else int(x.numpy()[0])
    result = BigInt(x_val) % BigInt(cls.P)
    return Tensor([1.0 if result == BigInt(0) else 0.0], dtype=dtypes.float32)

  @staticmethod
  def zeros_like(x: Tensor):
    return Tensor.zeros(*x.shape, dtype=dtypes.float64)

  @staticmethod
  def t32(x) -> Tensor:
    if isinstance(x, Tensor):
      return x
    return Tensor(x, dtype=dtypes.int32)

  # -- Unique methods: to be implemented by subclasses --
  @classmethod
  def mod_py_obj(cls, inp):
    if isinstance(inp, Tensor):
      val = int(inp.numpy().item()) if inp.numel() == 1 else int(inp.numpy()[0])
      result = BigInt(val) % BigInt(cls.P)
      return result.to_int()
    elif isinstance(inp, int):
      result = BigInt(inp) % BigInt(cls.P)
      return result.to_int()
    elif isinstance(inp, float):
      result = BigInt(int(inp)) % BigInt(cls.P)
      return result.to_int()
    else:
      return [cls.mod_py_obj(x) for x in inp]

  @classmethod
  def modinv_impl(cls, x: Tensor) -> Tensor:
    # Compute the modular inverse using BigInt extended GCD
    x_val = int(x.numpy().item()) if x.numel() == 1 else int(x.numpy()[0])
    
    from algebra.bigint.bigint import mod_inverse
    result = mod_inverse(BigInt(x_val), BigInt(cls.P))
    return Tensor([result.to_int()], dtype=x.dtype)

  @classmethod
  def pow_tensor(cls, base: Tensor, exponent: int) -> Tensor:
    # Use BigInt for exponentiation
    base_val = int(base.numpy().item()) if base.numel() == 1 else int(base.numpy()[0])
    
    result = pow(BigInt(base_val), exponent, BigInt(cls.P))
    return Tensor([result.to_int()], dtype=base.dtype)
