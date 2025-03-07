# tinycrypto/algebra [![CI](https://github.com/tinycrypto/algebra/actions/workflows/ci.yml/badge.svg)](https://github.com/tinycrypto/algebra/actions/workflows/ci.yml)

A [tinygrad](https://github.com/tinygrad/tinygrad) based algebra implementation of algebra. Influenced by [arkworks-rs/algebra](https://github.com/arkworks-rs/algebra/tree/master).

## Directory structure

* [`ff`](algebra/ff): Generic abstractions for, and implementations of various kinds of finite fields
  * - [ ] Mersenne31
    * - [x] basic
    * - [ ] extension
  * - [ ] BabyBear
    * - [x] basic
    * - [ ] extension
* [`ec`](algebra/fec): Generic abstractions for prime-order groups, and implementations of various kinds of (pairing-friendly and standard) elliptic curves
* [`poly`](algebra/poly): Interfaces for univariate, multivariate, and multilinear polynomials, and FFTs over finite fields
  * - [X] univariate
  * - [ ] multivariate
* [`linalg`](algebra/linalg): Interfaces for linear algebra operations, and implementations of various kinds of linear algebra operations
  * - [X] matrix inverse
  * - [X] LU decomposition