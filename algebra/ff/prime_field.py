from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

T = TypeVar("T", bound="PrimeField")


class BigInteger:
  def __init__(self, value: int):
    self.value = value

  def __add__(self, other: "BigInteger") -> "BigInteger":
    return BigInteger((self.value + other.value))

  def __sub__(self, other: "BigInteger") -> "BigInteger":
    return BigInteger((self.value - other.value))

  def __mul__(self, other: "BigInteger") -> "BigInteger":
    return BigInteger((self.value * other.value))

  def __mod__(self, modulus: int) -> "BigInteger":
    return BigInteger(self.value % modulus)

  def __repr__(self) -> str:
    return f"BigInteger({self.value})"


class PrimeField(ABC, Generic[T]):
  """
  Abstract base class for prime fields.
  This class is influenced by https://docs.rs/ark-ff/latest/ark_ff/fields/trait.PrimeField.html
  """

  @property
  @abstractmethod
  def MODULUS(self) -> BigInteger:
    """The modulus of the field."""
    pass

  @property
  @abstractmethod
  def MODULUS_MINUS_ONE_DIV_TWO(self) -> BigInteger:
    """(MODULUS - 1) / 2."""
    pass

  @property
  @abstractmethod
  def MODULUS_BIT_SIZE(self) -> int:
    """The bit size of the modulus."""
    pass

  @property
  @abstractmethod
  def TRACE(self) -> BigInteger:
    """The trace of the field."""
    pass

  @property
  @abstractmethod
  def TRACE_MINUS_ONE_DIV_TWO(self) -> BigInteger:
    """(TRACE - 1) / 2."""
    pass

  @abstractmethod
  def from_bigint(self, _repr: BigInteger) -> Optional[T]:
    """Convert a BigInteger to a field element."""
    pass

  @abstractmethod
  def into_bigint(self) -> BigInteger:
    """Convert the field element into a BigInteger."""
    pass

  # Provided methods
  @classmethod
  def from_be_bytes_mod_order(cls, _bytes: bytes) -> T:
    """Create a field element from big-endian bytes."""
    # Convert bytes to an integer and reduce modulo MODULUS
    integer_repr = int.from_bytes(bytes, byteorder="big")
    return cls.from_bigint(integer_repr % cls.MODULUS)

  @classmethod
  def from_le_bytes_mod_order(cls, _bytes: bytes) -> T:
    """Create a field element from little-endian bytes."""
    # Convert bytes to an integer and reduce modulo MODULUS
    integer_repr = int.from_bytes(bytes, byteorder="little")
    return cls.from_bigint(integer_repr % cls.MODULUS)
