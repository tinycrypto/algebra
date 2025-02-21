# tinycrypto/algebra

A tinygrad based algebra implementation of algebra. Influenced by [arkworks-rs/algebra](https://github.com/arkworks-rs/algebra/tree/master).

## Directory structure

* [`ff`](algebra/ff): Generic abstractions for, and implementations of various kinds of finite fields
* [`ec`](algebra/fec): Generic abstractions for prime-order groups, and implementations of various kinds of (pairing-friendly and standard) elliptic curves
* [`poly`](algebra/poly): Interfaces for univariate, multivariate, and multilinear polynomials, and FFTs over finite fields