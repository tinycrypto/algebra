"""Elliptic Curve Cryptography Module

This module provides implementations of popular elliptic curves used in cryptography,
including Bitcoin's secp256k1, NIST curves, and pairing-friendly curves.

Available curves:
- secp256k1 (Bitcoin/Ethereum)
- secp256r1/P-256 (NIST)
- secp384r1/P-384 (NIST)
- secp521r1/P-521 (NIST)
- BN254 (ZK proofs/pairings)

Tests are located in the tests/ subdirectory.
"""

from .curve import EllipticCurve, ECPoint
from .bn254 import G1 as BN254, Fq as BN254_Fq, Fr as BN254_Fr
from .secp256k1 import Secp256k1, Fp as Secp256k1_Fp, Fr as Secp256k1_Fr
from .secp256r1 import Secp256r1, Fp as Secp256r1_Fp, Fr as Secp256r1_Fr
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
  "Secp384r1",
  "Secp521r1",
  # Field implementations
  "BN254_Fq",
  "BN254_Fr",
  "Secp256k1_Fp",
  "Secp256k1_Fr",
  "Secp256r1_Fp",
  "Secp256r1_Fr",
  "Secp384r1_Fp",
  "Secp384r1_Fr",
  "Secp521r1_Fp",
  "Secp521r1_Fr",
  # Registry and utilities
  "get_curve",
  "list_curves",
]
