from algebra.extensions.gf2n import GF4, GF8
from algebra.ff.gf2 import GF2


def test_gf4():
    # Test elements in GF(4)
    a = GF4([GF2(1), GF2(0)])  # 1
    b = GF4([GF2(0), GF2(1)])  # x
    c = GF4([GF2(1), GF2(1)])  # 1 + x
    
    # Test addition (XOR)
    assert a + b == c
    assert b + b == GF4([GF2(0), GF2(0)])  # x + x = 0
    
    # Test multiplication
    # x * x = x^2, but x^2 = x + 1 in GF(4) with irreducible x^2 + x + 1
    result = b * b
    expected = GF4([GF2(1), GF2(1)])  # x + 1
    assert result == expected


def test_gf8():
    # Test elements in GF(8)
    a = GF8([GF2(1), GF2(0), GF2(0)])  # 1
    b = GF8([GF2(0), GF2(1), GF2(0)])  # x
    
    # Test addition
    result = a + b
    expected = GF8([GF2(1), GF2(1), GF2(0)])  # 1 + x
    assert result == expected
    
    # Test that x + x = 0
    assert b + b == GF8([GF2(0), GF2(0), GF2(0)])

