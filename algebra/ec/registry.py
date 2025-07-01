"""Simple elliptic curve registry"""

from algebra.ec.bn254 import G1 as BN254
from algebra.ec.secp256k1 import Secp256k1
from algebra.ec.secp256r1 import Secp256r1
from algebra.ec.curve25519 import Curve25519, Ed25519
from algebra.ec.secp384r1 import Secp384r1
from algebra.ec.secp521r1 import Secp521r1


# Simple curve registry
CURVES = {
    "bn254": BN254,
    "secp256k1": Secp256k1,
    "secp256r1": Secp256r1,
    "p256": Secp256r1,
    "curve25519": Curve25519,
    "ed25519": Ed25519,
    "secp384r1": Secp384r1,
    "p384": Secp384r1,
    "secp521r1": Secp521r1,
    "p521": Secp521r1,
    # Aliases
    "bitcoin": Secp256k1,
    "ethereum": Secp256k1,
}


def get_curve(name: str):
    """Get a curve class by name"""
    return CURVES.get(name.lower())


def list_curves():
    """List available curves"""
    return list(CURVES.keys())