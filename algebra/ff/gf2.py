class GF2:
    def __init__(self, x):
        self.value = int(x) & 1
    
    def __add__(self, other):
        return GF2(self.value ^ other.value)
    
    def __mul__(self, other):
        return GF2(self.value & other.value)
    
    def __repr__(self):
        return str(self.value)
    
    def __int__(self):
        return self.value
    
    def __eq__(self, other):
        if isinstance(other, GF2):
            return self.value == other.value
        return self.value == (int(other) & 1)
    
    __radd__ = __add__
    __rmul__ = __mul__