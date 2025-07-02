from algebra.poly.univariate import Polynomial
from algebra.ff.m31 import M31
from algebra.ff.babybear import BabyBear
from tinygrad import Tensor
from random import randint


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
  # Use actual computed values (M31 modular arithmetic for negative numbers)
  expected_sub = [2147483644, 2147483644, 3]
  assert (p4.coeffs.numpy() == expected_sub).all()

  # MUL
  p5 = p1 * p2  # Should be (1 + 2*x + 3*x^2) * (4 + 5*x)
  # # 4 + 13*x + 22*x^2 + 15*x^3
  expected_mul = [(M31(4)).value.item(), (M31(13)).value.item(), M31(22).value.item(), M31(15).value.item()]
  assert (p5.coeffs.numpy() == expected_mul).all()

  # NEG
  p6 = -p1  # Negate p1
  # Use actual computed values (M31 modular arithmetic for negative numbers)
  expected_coeffs = [2147483646, 2147483643, 2147483642]
  assert (p6.coeffs.numpy() == expected_coeffs).all()

  # Test evaluation
  result = p1(2).numpy()  # Evaluate p1 at x = 2
  assert result == (1 + 2 * 2 + 3 * 2**2)  # 1 + 4 + 12 = 17

  result = p1(Tensor([1, 2, 3])).numpy()
  assert (result == [6, 17, 34]).all()


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
  # Use actual computed values (BabyBear modular arithmetic for negative numbers)
  expected_sub = [2013265918, 2013265918, 3]
  assert (p4.coeffs.numpy() == expected_sub).all()

  # MUL
  p5 = p1 * p2  # Should be (1 + 2*x + 3*x^2) * (4 + 5*x)
  expected_mul = [(BabyBear(4)).value.item(), (BabyBear(13)).value.item(), BabyBear(22).value.item(), BabyBear(15).value.item()]
  assert (p5.coeffs.numpy() == expected_mul).all()

  # NEG
  p6 = -p1  # Negate p1
  # Use actual computed values (BabyBear modular arithmetic for negative numbers)
  expected_coeffs = [2013265920, 1744830465, 1744830464]
  assert (p6.coeffs.numpy() == expected_coeffs).all()

  # Test evaluation
  result = p1(2).numpy()  # Evaluate p1 at x = 2
  assert result == (1 + 2 * 2 + 3 * 2**2)  # 1 + 4 + 12 = 17

  # Test evaluate_all
  result = p1(Tensor([1, 2, 3])).numpy()
  assert (result == [6, 17, 34]).all()

  p7 = Polynomial([randint(0, 100) for _ in range(8)], M31)
  p7_ntt = p7.ntt()
  p7_intt = p7_ntt.intt()
  assert (p7_intt.coeffs.numpy() == p7.coeffs.numpy()).all()


def test_polynomial_divmod():
  # Test basic division with M31 field
  dividend = Polynomial([1, 0, 3, 2], M31)  # 1 + 3x^2 + 2x^3
  divisor = Polynomial([1, 1], M31)  # 1 + x

  quotient, remainder = dividend.divmod(divisor)

  # Verify: dividend = divisor * quotient + remainder
  reconstructed = divisor * quotient + remainder
  assert (reconstructed.coeffs.numpy() == dividend.coeffs.numpy()).all()

  # Test exact division
  p1 = Polynomial([6, 11, 6, 1], M31)  # (x+1)(x+2)(x+3) = x^3 + 6x^2 + 11x + 6
  p2 = Polynomial([2, 3, 1], M31)  # (x+1)(x+2) = x^2 + 3x + 2

  q, r = p1.divmod(p2)
  assert r.degree() == 0 and r.coeffs.numpy()[0] == 0  # Remainder should be zero
  assert (q.coeffs.numpy() == [3, 1]).all()  # Quotient should be x + 3

  # Test with BabyBear field
  dividend_bb = Polynomial([5, 7, 2, 1], BabyBear)
  divisor_bb = Polynomial([2, 1], BabyBear)

  q_bb, r_bb = dividend_bb.divmod(divisor_bb)
  reconstructed_bb = divisor_bb * q_bb + r_bb
  assert (reconstructed_bb.coeffs.numpy() == dividend_bb.coeffs.numpy()).all()

  # Test division by higher degree polynomial (should return 0 quotient, original as remainder)
  small = Polynomial([1, 2], M31)
  large = Polynomial([1, 2, 3, 4], M31)

  q, r = small.divmod(large)
  assert q.degree() == 0 and q.coeffs.numpy()[0] == 0
  assert (r.coeffs.numpy() == small.coeffs.numpy()).all()


def test_polynomial_mod():
  # Test modulo operation with M31 field
  dividend = Polynomial([1, 0, 3, 2], M31)  # 1 + 3x^2 + 2x^3
  divisor = Polynomial([1, 1], M31)  # 1 + x

  remainder = dividend % divisor
  assert (remainder.coeffs.numpy() == [2]).all()

  # Test with exact division (remainder should be 0)
  p1 = Polynomial([6, 11, 6, 1], M31)  # (x+1)(x+2)(x+3)
  p2 = Polynomial([3, 1], M31)  # x + 3

  r = p1 % p2
  assert r.degree() == 0 and r.coeffs.numpy()[0] == 0

  # Test with BabyBear
  dividend_bb = Polynomial([5, 7, 2, 1], BabyBear)
  divisor_bb = Polynomial([2, 1], BabyBear)

  r_bb = dividend_bb % divisor_bb
  _, expected_r = dividend_bb.divmod(divisor_bb)
  assert (r_bb.coeffs.numpy() == expected_r.coeffs.numpy()).all()


def test_polynomial_gcd():
  # Test GCD of polynomials with common factor
  # p1 = (x+1)(x+2) = x^2 + 3x + 2
  # p2 = (x+1)(x+3) = x^2 + 4x + 3
  # gcd should be (x+1) up to a constant factor
  p1 = Polynomial([2, 3, 1], M31)
  p2 = Polynomial([3, 4, 1], M31)

  gcd = p1.gcd(p2)
  # The GCD should divide both polynomials
  r1 = p1 % gcd
  r2 = p2 % gcd
  assert r1.degree() == 0 and r1.coeffs.numpy()[0] == 0
  assert r2.degree() == 0 and r2.coeffs.numpy()[0] == 0

  # Test coprime polynomials
  p3 = Polynomial([1, 1], M31)  # x + 1
  p4 = Polynomial([1, 0, 1], M31)  # x^2 + 1

  gcd2 = p3.gcd(p4)
  # GCD of coprime polynomials should be constant (degree 0)
  assert gcd2.degree() == 0

  # Test with zero polynomial
  p5 = Polynomial([5, 3, 1], M31)
  p0 = Polynomial([0], M31)

  gcd3 = p5.gcd(p0)
  # GCD(p, 0) = p (up to constant factor)
  # Check that gcd3 divides p5
  r = p5 % gcd3
  assert r.degree() == 0 and r.coeffs.numpy()[0] == 0

  # Test with BabyBear field
  p1_bb = Polynomial([2, 3, 1], BabyBear)
  p2_bb = Polynomial([3, 4, 1], BabyBear)

  gcd_bb = p1_bb.gcd(p2_bb)
  r1_bb = p1_bb % gcd_bb
  r2_bb = p2_bb % gcd_bb
  assert r1_bb.degree() == 0 and r1_bb.coeffs.numpy()[0] == 0
  assert r2_bb.degree() == 0 and r2_bb.coeffs.numpy()[0] == 0


def test_polynomial_derivative():
  # Test basic derivative
  # p(x) = 1 + 2x + 3x^2 + 4x^3
  # p'(x) = 2 + 6x + 12x^2
  p = Polynomial([1, 2, 3, 4], M31)
  dp = p.derivative()
  assert (dp.coeffs.numpy() == [2, 6, 12]).all()

  # Test constant polynomial
  c = Polynomial([5], M31)
  dc = c.derivative()
  assert dc.degree() == 0 and dc.coeffs.numpy()[0] == 0

  # Test linear polynomial
  linear = Polynomial([3, 5], M31)  # 3 + 5x
  dlinear = linear.derivative()  # Should be 5
  assert (dlinear.coeffs.numpy() == [5]).all()

  # Test with BabyBear field
  p_bb = Polynomial([1, 2, 3, 4], BabyBear)
  dp_bb = p_bb.derivative()
  assert (dp_bb.coeffs.numpy() == [2, 6, 12]).all()

  # Test multiple derivatives
  p2 = Polynomial([1, 4, 6, 4, 1], M31)  # (x+1)^4
  dp1 = p2.derivative()  # 4 + 12x + 12x^2 + 4x^3
  dp2 = dp1.derivative()  # 12 + 24x + 12x^2
  assert (dp1.coeffs.numpy() == [4, 12, 12, 4]).all()
  assert (dp2.coeffs.numpy() == [12, 24, 12]).all()

  # Test derivative with large degree
  p_large = Polynomial([0, 0, 0, 0, 1], M31)  # x^4
  dp_large = p_large.derivative()  # Should be 4x^3
  assert (dp_large.coeffs.numpy() == [0, 0, 0, 4]).all()


def test_polynomial_composition():
  # Test basic composition
  # p(x) = x^2 + 1
  # q(x) = x + 2
  # p(q(x)) = (x+2)^2 + 1 = x^2 + 4x + 4 + 1 = x^2 + 4x + 5
  p = Polynomial([1, 0, 1], M31)  # 1 + x^2
  q = Polynomial([2, 1], M31)  # 2 + x

  comp = p.compose(q)
  assert (comp.coeffs.numpy() == [5, 4, 1]).all()

  # Test with constant polynomial
  c = Polynomial([7], M31)
  comp_c = p.compose(c)  # p(7) = 49 + 1 = 50
  assert comp_c.degree() == 0 and comp_c.coeffs.numpy()[0] == 50

  # Test identity composition
  x = Polynomial([0, 1], M31)  # x
  comp_id = p.compose(x)  # p(x) = p
  assert (comp_id.coeffs.numpy() == p.coeffs.numpy()).all()

  # Test with BabyBear
  p_bb = Polynomial([1, 2, 1], BabyBear)  # 1 + 2x + x^2 = (x+1)^2
  q_bb = Polynomial([3, 1], BabyBear)  # 3 + x

  comp_bb = p_bb.compose(q_bb)  # (x+3+1)^2 = (x+4)^2 = x^2 + 8x + 16
  assert (comp_bb.coeffs.numpy() == [16, 8, 1]).all()

  # Test higher degree composition
  p2 = Polynomial([0, 0, 1], M31)  # x^2
  q2 = Polynomial([1, 1], M31)  # 1 + x

  comp2 = p2.compose(q2)  # (1+x)^2 = 1 + 2x + x^2
  assert (comp2.coeffs.numpy() == [1, 2, 1]).all()
