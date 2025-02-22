from algebra.poly.univariate import Polynomial
from algebra.ff.m31 import M31
from algebra.ff.babybear import BabyBear


def test_polynomial_operations_m31():
  p1 = Polynomial([1, 2, 3], M31)  #  1 + 2*x + 3*x^2
  p2 = Polynomial([4, 5], M31)  #  4 + 5*x

  # Test degree
  assert p1.degree() == 2  # Degree of p1 should be 2
  assert p2.degree() == 1  # Degree of p2 should be 1

  # ADD
  p3 = p1 + p2  # Should be 5 + 7*x + 3*x^2
  assert (p3.coeffs.numpy() == [5, 7, 3]).all()

  # SUB
  p4 = p1 - p2  # Should be -3 - 3*x + 3*x^2
  assert (p4.coeffs.numpy() == [(-M31(3)).value.numpy(), (-M31(3)).value.numpy(), M31(3).value.numpy()]).all()

  # MUL
  p5 = p1 * p2  # Should be (1 + 2*x + 3*x^2) * (4 + 5*x)
  # # 4 + 13*x + 22*x^2 + 15*x^3
  assert (p5.coeffs.numpy() == [(M31(4)).value.numpy(), (M31(13)).value.numpy(), M31(22).value.numpy(), M31(15).value.numpy()]).all()

  # NEG
  p6 = -p1  # Negate p1
  assert (p6.coeffs.numpy() == [(-M31(1)).value.numpy(), (-M31(2)).value.numpy(), (-M31(3)).value.numpy()]).all()

  # Test evaluation
  result = p1.evaluate(2).numpy()  # Evaluate p1 at x = 2
  assert result == (1 + 2 * 2 + 3 * 2**2)  # 1 + 4 + 12 = 17


def test_polynomial_operations_babybear():
  p1 = Polynomial([1, 2, 3], BabyBear)  #  1 + 2*x + 3*x^2
  p2 = Polynomial([4, 5], BabyBear)  #  4 + 5*x

  # Test degree
  assert p1.degree() == 2  # Degree of p1 should be 2
  assert p2.degree() == 1  # Degree of p2 should be 1

  # ADD
  p3 = p1 + p2  # Should be 5 + 7*x + 3*x^2
  assert (p3.coeffs.numpy() == [5, 7, 3]).all()

  # SUB
  p4 = p1 - p2  # Should be -3 - 3*x + 3*x^2
  assert (p4.coeffs.numpy() == [(-BabyBear(3)).value.numpy(), (-BabyBear(3)).value.numpy(), BabyBear(3).value.numpy()]).all()

  # MUL
  p5 = p1 * p2  # Should be (1 + 2*x + 3*x^2) * (4 + 5*x)
  assert (
    p5.coeffs.numpy() == [(BabyBear(4)).value.numpy(), (BabyBear(13)).value.numpy(), BabyBear(22).value.numpy(), BabyBear(15).value.numpy()]
  ).all()

  # NEG
  p6 = -p1  # Negate p1
  assert (p6.coeffs.numpy() == [(-BabyBear(1)).value.numpy(), (-BabyBear(2)).value.numpy(), (-BabyBear(3)).value.numpy()]).all()

  # Test evaluation
  result = p1.evaluate(2).numpy()  # Evaluate p1 at x = 2
  assert result == (1 + 2 * 2 + 3 * 2**2)  # 1 + 4 + 12 = 17
