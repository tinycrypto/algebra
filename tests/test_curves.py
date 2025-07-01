"""Comprehensive tests for all elliptic curve implementations"""

import pytest
from algebra.ec.bn254 import G1 as BN254
from algebra.ec.secp256k1 import Secp256k1
from algebra.ec.secp256r1 import Secp256r1
from algebra.ec.curve25519 import Curve25519, Ed25519
from algebra.ec.secp384r1 import Secp384r1
from algebra.ec.secp521r1 import Secp521r1


def test_bn254():
    """Test BN254 curve"""
    # Create curve and generator
    G = BN254.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test order
    from algebra.ec.bn254 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_secp256k1():
    """Test secp256k1 curve (Bitcoin)"""
    # Create curve and generator
    G = Secp256k1.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test known doubling result
    expected_2G_x = 0xC6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5
    expected_2G_y = 0x1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A
    
    actual_2G = G.double()
    assert actual_2G.x.value.item() == expected_2G_x
    assert actual_2G.y.value.item() == expected_2G_y
    
    # Test order
    from algebra.ec.secp256k1 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_secp256r1():
    """Test secp256r1 curve (P-256)"""
    # Create curve and generator
    G = Secp256r1.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test known doubling result
    expected_2G_x = 0x7CF27B188D034F7E8A52380304B51AC3C08969E277F21B35A60B48FC47669978
    expected_2G_y = 0x07775510DB8ED040293D9AC69F7430DBBA7DADE63CE982299E04B79D227873D1
    
    actual_2G = G.double()
    assert actual_2G.x.value.item() == expected_2G_x
    assert actual_2G.y.value.item() == expected_2G_y
    
    # Test order
    from algebra.ec.secp256r1 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_curve25519():
    """Test Curve25519"""
    # Create curve and generator
    G = Curve25519.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test order
    from algebra.ec.curve25519 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_ed25519():
    """Test Ed25519"""
    # Create curve and generator
    G = Ed25519.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))


def test_secp384r1():
    """Test secp384r1 curve (P-384)"""
    # Create curve and generator
    G = Secp384r1.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test order
    from algebra.ec.secp384r1 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_secp521r1():
    """Test secp521r1 curve (P-521)"""
    # Create curve and generator
    G = Secp521r1.generator()
    assert G.is_on_curve()
    
    # Test basic operations
    G2 = G.double()
    assert G2.is_on_curve()
    
    G3 = G + G2
    assert G3.is_on_curve()
    assert G3.equals(G.scalar_mul(3))
    
    # Test order
    from algebra.ec.secp521r1 import Fr
    n = Fr.P
    nG = G.scalar_mul(n)
    assert nG.is_infinity()


def test_scalar_multiplication_consistency():
    """Test that scalar multiplication is consistent across curves"""
    curves_and_generators = [
        (BN254.generator(), "BN254"),
        (Secp256k1.generator(), "secp256k1"),
        (Secp256r1.generator(), "secp256r1"),
        (Curve25519.generator(), "Curve25519"),
        (Secp384r1.generator(), "secp384r1"),
        (Secp521r1.generator(), "secp521r1"),
    ]
    
    for G, name in curves_and_generators:
        # Test that k*G = G + G + ... + G (k times) for small k
        k = 5
        kG_scalar = G.scalar_mul(k)
        kG_addition = G
        for _ in range(k - 1):
            kG_addition = kG_addition + G
        
        assert kG_scalar.equals(kG_addition), f"Scalar multiplication inconsistent for {name}"
        
        # Test that (a + b)*G = a*G + b*G
        a, b = 7, 11
        ab_G = G.scalar_mul(a + b)
        aG_plus_bG = G.scalar_mul(a) + G.scalar_mul(b)
        
        assert ab_G.equals(aG_plus_bG), f"Scalar multiplication distributivity failed for {name}"


def test_multi_scalar_multiplication():
    """Test multi-scalar multiplication on different curves"""
    curves_and_generators = [
        (BN254.generator(), "BN254"),
        (Secp256k1.generator(), "secp256k1"),
        (Secp256r1.generator(), "secp256r1"),
        (Curve25519.generator(), "Curve25519"),
    ]
    
    for G, name in curves_and_generators:
        # Test MSM with small values
        points = [G, G.double(), G.scalar_mul(3)]
        scalars = [2, 3, 4]
        
        # Expected: 2*G + 3*(2*G) + 4*(3*G) = 2*G + 6*G + 12*G = 20*G
        expected = G.scalar_mul(20)
        actual = G.multi_scalar_mul(points, scalars)
        
        assert expected.equals(actual), f"MSM failed for {name}"


def test_point_compression():
    """Test basic point operations that might be used in compression"""
    curves_and_generators = [
        (BN254.generator(), "BN254"),
        (Secp256k1.generator(), "secp256k1"),
        (Secp256r1.generator(), "secp256r1"),
    ]
    
    for G, name in curves_and_generators:
        # Test that -(-P) = P
        neg_G = -G
        neg_neg_G = -neg_G
        assert G.equals(neg_neg_G), f"Double negation failed for {name}"
        
        # Test that P + (-P) = O
        sum_with_neg = G + (-G)
        assert sum_with_neg.is_infinity(), f"P + (-P) != O for {name}"
        
        # Test that 2*P = P + P
        double_P = G.double()
        add_P = G + G
        assert double_P.equals(add_P), f"Double != Add for {name}"


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_bn254,
        test_secp256k1,
        test_secp256r1,
        test_curve25519,
        test_ed25519,
        test_secp384r1,
        test_secp521r1,
        test_scalar_multiplication_consistency,
        test_multi_scalar_multiplication,
        test_point_compression,
    ]
    
    for test_func in test_functions:
        print(f"Running {test_func.__name__}...")
        test_func()
        print(f"✓ {test_func.__name__} passed")
    
    print("\nAll curve tests passed!")