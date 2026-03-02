# AGR16 Recursive Auxiliary Chain

## Purpose

This document defines the long-lived design invariant for AGR16 public evaluation in this repository: multiplication must use a recursive auxiliary chain, not a fixed number of auxiliary fields, so circuits with multiplication depth greater than two can preserve Equation 5.1-style ciphertext consistency checks.

## Scope

This design applies to:

- `src/agr16/public_key.rs`
- `src/agr16/encoding.rs`
- `src/agr16/sampler.rs`
- `src/circuit/evaluable/agr16.rs`
- `src/agr16/mod.rs` tests

## Recursive state model

For each wire encoding:

- `c_times_s_pubkeys[level]` is the public-key label for `E(c * s^(level+1))`.
- `c_times_s_encodings[level]` is the ciphertext-side auxiliary value that must satisfy the same recursive relation as the public key.
- `s_power_pubkeys[level]` and `s_power_encodings[level]` provide advice encodings for `E(s^(level+2))`.

All homomorphic operations are public and must not use secret-key material directly.

## Multiplication design rule

AGR16 multiplication uses Section 5 Eq. 5.24 / 5.25 style recursion level-by-level:

- Base wire output `c` is computed from level-0 auxiliary terms.
- Level `l` auxiliary output requires level `l+1` inputs and a convolution over levels `0..l`.

Because each level depends on `l+1`, multiplication consumes one available auxiliary level from the chain. This is intentional and matches the recursion in the paper.

## Depth sizing rule

Let `L` be initial auxiliary chain length produced by samplers and `D` be multiplication depth on a circuit path.

- To preserve level-0 post-multiplication auxiliary invariants through depth `D`, use `L >= D + 1`.

Tests for depth>=3 must therefore use a sampler depth greater than or equal to 4.

## Security/behavior constraints

- `Agr16Encoding` arithmetic remains secret-independent.
- Equation 5.1 checks are performed against ciphertext/public-key relation, not plaintext-only checks.
- Sampler-generated recursive chains must keep key/encoding depth aligned.
