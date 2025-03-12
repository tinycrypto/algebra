from algebra.ff.m31 import M31
from algebra.poly.univariate import Polynomial
from random import randint
from tinygrad import Tensor, dtypes

p1 = Polynomial(Tensor([randint(0, 10) for _ in range(10)], dtype=dtypes.uint32), M31)
print(f'p1: {p1.coeffs.numpy()}')

p1_ntt = p1.ntt()
print(f'p1_ntt: {p1_ntt.coeffs.numpy()}')

p1_intt = p1_ntt.intt()
print(f'p1_intt: {p1_intt.coeffs.numpy()}')

assert (p1_intt.coeffs.numpy() == p1.coeffs.numpy()).all()

