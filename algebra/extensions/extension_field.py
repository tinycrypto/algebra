from abc import ABC, abstractmethod
from typing import List, Union


class ExtensionField(ABC):
    """Abstract base class for field extensions"""
    
    base_field = None
    degree = None
    
    def __init__(self, coeffs: Union[List, 'ExtensionField']):
        """Initialize with coefficients or copy from another element"""
        if isinstance(coeffs, ExtensionField):
            self.coeffs = coeffs.coeffs[:]
        else:
            self.coeffs = list(coeffs)
            # Pad with zeros if needed
            while len(self.coeffs) < self.degree:
                self.coeffs.append(self.base_field(0))
    
    @abstractmethod
    def __add__(self, other):
        """Addition in the extension field"""
        pass
    
    @abstractmethod
    def __mul__(self, other):
        """Multiplication in the extension field"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.coeffs})"
    
    def __eq__(self, other):
        if not isinstance(other, ExtensionField):
            return False
        return self.coeffs == other.coeffs
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)