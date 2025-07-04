from algebra.extensions.extension_field import ExtensionField
from algebra.ff.gf2 import GF2


class GF2n(ExtensionField):
    base_field = GF2
    degree = None
    irreducible_poly = None
    
    def __add__(self, other):
        if isinstance(other, int):
            other = type(self)([GF2(other)])
        
        result = [self.coeffs[i] + other.coeffs[i] for i in range(self.degree)]
        return type(self)(result)
    
    def __mul__(self, other):
        if isinstance(other, int):
            other = type(self)([GF2(other)])
        
        result = [GF2(0)] * (2 * self.degree - 1)
        for i in range(len(self.coeffs)):
            for j in range(len(other.coeffs)):
                result[i + j] += self.coeffs[i] * other.coeffs[j]
        
        return type(self)(self._reduce(result))
    
    def _reduce(self, coeffs):
        while len(coeffs) > self.degree and coeffs[-1] == GF2(1):
            for i, coeff in enumerate(self.irreducible_poly):
                idx = len(coeffs) - len(self.irreducible_poly) + i
                if idx >= 0:
                    coeffs[idx] += coeff
            coeffs.pop()
        
        return coeffs[:self.degree] + [GF2(0)] * (self.degree - len(coeffs))


class GF4(GF2n):
    degree = 2
    irreducible_poly = [GF2(1), GF2(1), GF2(1)]


class GF8(GF2n):
    degree = 3
    irreducible_poly = [GF2(1), GF2(1), GF2(0), GF2(1)]