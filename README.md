# mxx: Lattice-Based Cryptography Library

This library is intended to support primitive-level lattice-based cryptography developed by Machina-iO. `mxx` is a primitive library upon which we may build more complex constructions, potentially for academic papers.

Applications of `mxx` include:
- [Diamond iO](https://github.com/MachinaIO/diamond-io)

### Prerequisites
- [OpenFHE](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html) (System install required in `/usr/local/lib`), make sure to install our [fork](https://github.com/MachinaIO/openfhe-development/tree/feat/improve_determinant) in `/feat/improve_determinant` branch

## Overview

### Matrix Element

In the LWE setting, matrix elements are integers, and in the RLWE setting, matrix elements are polynomials. Since we want matrix elements to be generic over types, we define basic common functionality as a trait.

### Matrix

The core of lattice operations is represented as a matrix. Both disk-based and memory (RAM)-based storage are supported as two options.

### Sampler

For basic matrix sampling, there are:

1. Hash sampler  
2. Uniform sampler  

The trapdoor sampler is used for lattice trapdoor sampling techniques (detailed in the algorithm described in [Implementing Token-Based Obfuscation under (Ring) LWE](https://eprint.iacr.org/2018/1222.pdf)).

### BGG+ Encoding

An encoding scheme introduced in [Fully Key-Homomorphic Encryption, Arithmetic Circuit ABE, and Compact Garbled Circuits](https://eprint.iacr.org/2014/356.pdf), known as BGG+ encoding.

### Circuit

Arithmetic circuits that can be evaluated homomorphically through arithmetic gate operations.

## License

<sup>
Licensed under the [MIT license](LICENSE).
</sup>
