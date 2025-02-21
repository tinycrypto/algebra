from .prime_field import PrimeField, BigInteger


class MyPrimeField(PrimeField["MyPrimeField"]):
  MODULUS = BigInteger(17)
  MODULUS_MINUS_ONE_DIV_TWO = BigInteger(8)
  MODULUS_BIT_SIZE = 5
  TRACE = BigInteger(0)
  TRACE_MINUS_ONE_DIV_TWO = BigInteger(-1)

  def from_bigint(self, _repr: BigInteger) -> "MyPrimeField":
    return MyPrimeField(repr.value % self.MODULUS.value)

  def into_bigint(self) -> BigInteger:
    return BigInteger(self.value)

  def __init__(self, value: int):
    self.value = value % self.MODULUS.value

  def __repr__(self) -> str:
    return f"MyPrimeField({self.value})"

  def __add__(self, other: "MyPrimeField") -> "MyPrimeField":
    if not isinstance(other, MyPrimeField):
      return NotImplemented
    return MyPrimeField((self.value + other.value) % self.MODULUS.value)
