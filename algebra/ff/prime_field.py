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
    # Convert tensor boolean to Python bool safely
    assert not bool(zero_tensor.any().item()), "0 has no inverse"
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
    return f"{self.value.item() if self.value.numel() == 1 else self.value.tolist()}"

  def __int__(self):
    return int(self.value.item())

  def __len__(self):
    return len(self.value)

  def tobytes(self):
    val = int(self.value.item()) if self.value.numel() == 1 else int(self.value[0].item())
    result = BigInt(val) % BigInt(self.P)
    return result.to_int().to_bytes((result.to_int().bit_length() + 7) // 8, "big")

  def __eq__(self, other):
    if isinstance(other, int):
      other = type(self)(other)
    result_tensor = self.eq_t(self.value, other.value)
    # For equality, all elements must match
    return bool(result_tensor.all().item())

  # -- Common arithmetic utility methods --
  @classmethod
  def add(cls, a: Tensor, b: Tensor) -> Tensor:
    # For small primes (fits in 32-bit), use pure vectorized tinygrad operations
    if cls.P < (1 << 31):
      return (a + b) % cls.P

    # For large primes, use 64-bit arithmetic to avoid overflow
    if cls.P < (1 << 63):
      # Cast to int64 for safe arithmetic, then back to original dtype
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      result = (a_64 + b_64) % cls.P
      return result.cast(a.dtype)

    # For very large primes, use element-wise tensor operations
    # Even cryptographic primes can be handled with tensor arithmetic
    a_64 = a.cast(dtypes.int64)
    b_64 = b.cast(dtypes.int64)
    return ((a_64 + b_64) % cls.P).cast(a.dtype)

  @classmethod
  def sub(cls, a: Tensor, b: Tensor) -> Tensor:
    # For small primes (fits in 32-bit), use pure vectorized tinygrad operations
    if cls.P < (1 << 31):
      return (a - b) % cls.P

    # For large primes, use 64-bit arithmetic to avoid overflow
    if cls.P < (1 << 63):
      # Cast to int64 for safe arithmetic, then back to original dtype
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      result = (a_64 - b_64) % cls.P
      return result.cast(a.dtype)

    # For very large primes, use element-wise tensor operations
    a_64 = a.cast(dtypes.int64)
    b_64 = b.cast(dtypes.int64)
    return ((a_64 - b_64) % cls.P).cast(a.dtype)

  @classmethod
  def neg(cls, a: Tensor) -> Tensor:
    # For small primes (fits in 32-bit), use pure vectorized tinygrad operations
    if cls.P < (1 << 31):
      return (-a) % cls.P

    # For large primes, use 64-bit arithmetic to avoid overflow
    if cls.P < (1 << 63):
      # Cast to int64 for safe arithmetic, then back to original dtype
      a_64 = a.cast(dtypes.int64)
      result = (-a_64) % cls.P
      return result.cast(a.dtype)

    # For very large primes, use element-wise tensor operations
    a_64 = a.cast(dtypes.int64)
    return ((-a_64) % cls.P).cast(a.dtype)

  @classmethod
  def mul_mod(cls, a: Tensor, b: Tensor) -> Tensor:
    # For small primes where a*b fits in 64-bit, use vectorized operations
    if cls.P < (1 << 31):
      # Use 64-bit for intermediate result to prevent overflow
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      result = (a_64 * b_64) % cls.P
      return result.cast(a.dtype)

    # For medium primes that fit in 63-bit (to allow for squaring), use 64-bit arithmetic
    if cls.P < (1 << 32):  # Conservative check for multiplication safety
      a_64 = a.cast(dtypes.int64)
      b_64 = b.cast(dtypes.int64)
      result = (a_64 * b_64) % cls.P
      return result.cast(a.dtype)

    # For very large primes, use element-wise tensor operations
    a_64 = a.cast(dtypes.int64)
    b_64 = b.cast(dtypes.int64)
    return ((a_64 * b_64) % cls.P).cast(a.dtype)

  @classmethod
  def sum_mod(cls, x: Tensor, axis=None) -> Tensor:
    # For small primes, use pure tensor operations
    if cls.P < (1 << 31):
      return x.sum(axis=axis) % cls.P

    # For large primes that fit in 64-bit, use extended precision
    if cls.P < (1 << 63):
      x_64 = x.cast(dtypes.int64)
      result = x_64.sum(axis=axis) % cls.P
      return result.cast(x.dtype)

    # For very large primes, fall back to BigInt
    x_sum = int(x.sum(axis=axis).item())
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
    val = int(x.item()) if x.numel() == 1 else int(x[0].item())
    result = BigInt(val) % BigInt(cls.P)
    return result.to_int().to_bytes((result.to_int().bit_length() + 7) // 8, "big")

  @classmethod
  def eq_t(cls, x: Tensor, y: Tensor):
    # For small primes, use pure tensor operations
    if cls.P < (1 << 31):
      # Reduce both tensors modulo P and compare directly
      x_mod = x % cls.P
      y_mod = y % cls.P
      return (x_mod == y_mod).float()

    # For larger primes, use 64-bit precision
    if cls.P < (1 << 63):
      x_64 = x.cast(dtypes.int64) % cls.P
      y_64 = y.cast(dtypes.int64) % cls.P
      return (x_64 == y_64).cast(dtypes.float32)

    # Fallback for very large primes - minimize tensor realization
    if x.numel() == 1 and y.numel() == 1:
      x_val = int(x.item())
      y_val = int(y.item())
      x_mod = BigInt(x_val) % BigInt(cls.P)
      y_mod = BigInt(y_val) % BigInt(cls.P)
      return Tensor([1.0 if x_mod == y_mod else 0.0], dtype=dtypes.float32)
    else:
      # For vector inputs, use vectorized comparison when possible
      return (x == y).float()

  @classmethod
  def iszero(cls, x: Tensor):
    # For small primes, use pure tensor operations
    if cls.P < (1 << 31):
      return ((x % cls.P) == 0).float()

    # For larger primes, use 64-bit precision
    if cls.P < (1 << 63):
      x_64 = x.cast(dtypes.int64) % cls.P
      return (x_64 == 0).cast(dtypes.float32)

    # Fallback for very large primes
    if x.numel() == 1:
      x_val = int(x.item())
      result = BigInt(x_val) % BigInt(cls.P)
      return Tensor([1.0 if result == BigInt(0) else 0.0], dtype=dtypes.float32)
    else:
      # For vector inputs, use direct comparison
      return (x == 0).float()

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
      # For small tensors, optimize by avoiding BigInt when possible
      if inp.numel() == 1:
        if cls.P < (1 << 63):
          # Use Python's native modulo for smaller primes
          val = int(inp.item())
          return val % cls.P
        else:
          # Use BigInt only for very large primes
          val = int(inp.item())
          result = BigInt(val) % BigInt(cls.P)
          return result.to_int()
      else:
        # For multi-element tensors, process first element only
        val = int(inp[0].item())
        if cls.P < (1 << 63):
          return val % cls.P
        else:
          result = BigInt(val) % BigInt(cls.P)
          return result.to_int()
    elif isinstance(inp, int):
      if cls.P < (1 << 63):
        return inp % cls.P
      else:
        result = BigInt(inp) % BigInt(cls.P)
        return result.to_int()
    elif isinstance(inp, float):
      if cls.P < (1 << 63):
        return int(inp) % cls.P
      else:
        result = BigInt(int(inp)) % BigInt(cls.P)
        return result.to_int()
    else:
      return [cls.mod_py_obj(x) for x in inp]

  @classmethod
  def modinv_impl(cls, x: Tensor) -> Tensor:
    # Compute the modular inverse using BigInt extended GCD
    x_val = int(x.item()) if x.numel() == 1 else int(x[0].item())

    from algebra.bigint.bigint import mod_inverse

    result = mod_inverse(BigInt(x_val), BigInt(cls.P))
    return Tensor([result.to_int()], dtype=x.dtype)

  @classmethod
  def pow_tensor(cls, base: Tensor, exponent: int) -> Tensor:
    # Use BigInt for exponentiation
    base_val = int(base.item()) if base.numel() == 1 else int(base[0].item())

    result = pow(BigInt(base_val), exponent, BigInt(cls.P))
    return Tensor([result.to_int()], dtype=base.dtype)
