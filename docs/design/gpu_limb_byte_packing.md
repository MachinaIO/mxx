# Design: GPU Limb Byte Packing

## Purpose

Define a long-lived GPU memory representation invariant that reduces VRAM usage without changing polynomial arithmetic semantics.

## Core Decision

GPU matrix/poly limb storage uses byte-packed coefficients instead of fixed-width `u64` slots.

- For each modulus `q_i`, limb coefficient width is:
  - `coeff_bytes_i = ceil(bit_width(q_i) / 8)`.
- Persisted limb buffers in CUDA memory are `uint8_t` buffers.
- Kernels load/store limb values through packed helpers that reconstruct `u64` temporaries from per-limb byte width.

## Required Invariants

1. Stored value invariant:
   - Each stored residue for limb `i` must fit in `coeff_bytes_i` bytes.
2. Access invariant:
   - Every kernel path that reads/writes persisted limb data must use per-limb metadata (`stride_bytes`, `coeff_bytes`) instead of assuming `sizeof(u64)`.
3. Layout invariant:
   - Poly stride is tracked in bytes (`bytes_per_poly`), and limb offsets inside a poly are byte offsets.
4. Compatibility invariant:
   - External host-facing RNS batch APIs keep the existing `u64`-per-coefficient byte contract; conversion happens at the CUDA boundary.

## Why This Design

- VRAM reduction: small moduli no longer consume 8 bytes per coefficient.
- Behavioral stability: arithmetic logic still operates on `u64` values after packed load, preserving existing modular math semantics.
- Migration safety: keeping host RNS batch format unchanged avoids broad Rust API and test churn while still achieving GPU memory savings.

## Trade-offs

- Additional packed load/store operations add minor per-access overhead.
- Serde paths need explicit conversion between host `u64` layout and packed GPU layout.
- Kernel signatures carry more metadata (`stride_bytes`, `coeff_bytes`), increasing plumbing complexity.

