# tinycrypto

tinycrypto is a collection of cryptographic primitives implemented in [tinygrad](https://github.com/tinygrad/tinygrad). Because certain cryptographic schemes(e.g lattice cryptography, polynomial operations) require quite of an matrix operations, we began experimenting with the idea of leveraging tinygrad’s hyper-optimized `Tensor` data structure. Could this approach lead to a fast yet user-friendly Python cryptography library?

## Components

- **[`ff`](tinycrypto/ff)**: Generic abstractions and implementations of various finite fields:
  - Mersenne31
    - [x] Basic
    - [ ] Extension
  - BabyBear
    - [x] Basic
    - [ ] Extension

- **[`poly`](tinycrypto/poly)**: Interfaces for univariate, multivariate, and multilinear polynomials, as well as FFTs over finite fields:
  - [x] Univariate
  - [ ] Multivariate