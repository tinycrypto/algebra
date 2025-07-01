# Elliptic Curve Tests

This directory contains comprehensive tests for all elliptic curve implementations.

## Test Organization

- **`test_simple.py`** - Basic import and registry tests (safe to run)
- **`test_registry.py`** - Tests for curve registry functionality
- **`test_bn254.py`** - Tests for BN254 pairing-friendly curve
- **`test_secp256k1.py`** - Tests for secp256k1 (Bitcoin/Ethereum curve)  
- **`test_secp256r1.py`** - Tests for secp256r1 (NIST P-256 curve)
- **`test_curve25519.py`** - Tests for Curve25519 and Ed25519
- **`test_nist_curves.py`** - Tests for P-384 and P-521 curves
- **`test_interoperability.py`** - Cross-curve consistency tests

## Running Tests

### Simple Tests (Recommended)
```bash
python algebra/ec/tests/test_simple.py
```

### Full Test Suite
```bash
python algebra/ec/tests/run_tests.py
```

### Individual Test Modules
```bash
python algebra/ec/tests/test_registry.py
python algebra/ec/tests/test_secp256k1.py
# etc.
```

## Test Categories

1. **Import Tests** - Verify all modules can be imported
2. **Registry Tests** - Test curve selection and aliases
3. **Basic Operations** - Point addition, doubling, scalar multiplication
4. **Known Values** - Test against standard test vectors
5. **Mathematical Properties** - Verify curve mathematics
6. **Interoperability** - Cross-curve consistency checks

## Notes

- Some tests may require significant computation time due to large field operations
- Simple tests can be run quickly without heavy computation
- All curves are tested for mathematical correctness and standard compliance