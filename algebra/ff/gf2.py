from algebra.ff.prime_field import PrimeField

class GF2(PrimeField):
    P = 2
    
    def __init__(self, x):
        self.value = self.t32(int(x) & 1)
    
    def __add__(self, other):
        return GF2(self.value.item() ^ GF2(other).value.item())
    
    def __mul__(self, other):
        return GF2(self.value.item() & GF2(other).value.item())
    
    __radd__ = __add__
    __rmul__ = __mul__