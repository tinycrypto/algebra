"""Elliptic Curve Cryptography Module

This module provides implementations of popular elliptic curves used in cryptography,
including Bitcoin's secp256k1, NIST curves, and pairing-friendly curves.
"""

from .curve import EllipticCurve, ECPoint
from .bn254 import G1 as BN254, Fq as BN254_Fq, Fr as BN254_Fr
from .secp256k1 import Secp256k1, Fp as Secp256k1_Fp, Fr as Secp256k1_Fr
from .secp256r1 import Secp256r1, Fp as Secp256r1_Fp, Fr as Secp256r1_Fr
from .curve25519 import Curve25519, Ed25519, Fp as Curve25519_Fp, Fr as Curve25519_Fr
from .secp384r1 import Secp384r1, Fp as Secp384r1_Fp, Fr as Secp384r1_Fr
from .secp521r1 import Secp521r1, Fp as Secp521r1_Fp, Fr as Secp521r1_Fr
from .registry import get_curve, list_curves

__all__ = [
    # Base classes
    "EllipticCurve",
    "ECPoint",
    
    # Curve implementations
    "BN254",
    "Secp256k1",
    "Secp256r1", 
    "Curve25519",
    "Ed25519",
    "Secp384r1",
    "Secp521r1",
    
    # Field implementations
    "BN254_Fq", "BN254_Fr",
    "Secp256k1_Fp", "Secp256k1_Fr",
    "Secp256r1_Fp", "Secp256r1_Fr",
    "Curve25519_Fp", "Curve25519_Fr",
    "Secp384r1_Fp", "Secp384r1_Fr",
    "Secp521r1_Fp", "Secp521r1_Fr",
    
    # Registry and utilities
    "get_curve",
    "list_curves",
]