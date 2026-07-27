# Workspace architecture

This repository is a virtual Cargo workspace. It intentionally has no root package or compatibility facade: consumers depend on the crate that owns the required abstraction.

## Dependency layers

```text
Level 1: mxx-primitives
             ^
             |
Level 2: mxx-gadgets
             ^
             |
Level 3: mxx-func-enc   mxx-we   mxx-io
```

Dependencies may only point upward in this diagram:

- `mxx-primitives` has no dependency on another workspace crate.
- `mxx-gadgets` depends only on `mxx-primitives`.
- `mxx-func-enc`, `mxx-we`, and `mxx-io` depend on `mxx-primitives` and `mxx-gadgets`.
- Application crates must not depend on one another.

Code shared by two application crates belongs in the lowest layer that can express it naturally. For example, the native polynomial-matrix benchmark interface and common iO/WE error-simulation helpers live in `mxx-gadgets`, while CUDA matrix measurements live beside the reusable benchmark interface instead of in `mxx-io`.

## Directory ownership

### `mxx-primitives`

`crates/primitives/` owns operations that are broadly useful across lattice-cryptography schemes:

| Path | Responsibility |
| --- | --- |
| `crates/primitives/src/element/` | Ring element abstractions. |
| `crates/primitives/src/poly/` | Polynomial traits, DCRT parameters, CPU polynomials, and GPU polynomial wrappers. |
| `crates/primitives/src/matrix/` | Generic matrices, DCRT matrices, memory/disk backing, and GPU matrices. |
| `crates/primitives/src/sampler/` | Uniform, hash, trapdoor, and GPU samplers plus generic sampling bounds. |
| `crates/primitives/src/rlwe_enc.rs` | Low-level RLWE encryption helpers. |
| `crates/primitives/src/env.rs` | Primitive-operation configuration. |
| `crates/primitives/cuda/` | Native CUDA headers and kernels. |
| `crates/primitives/build.rs` | OpenFHE and CUDA compilation/linking. |
| `crates/primitives/benches/` | Primitive CPU/GPU matrix and preimage benchmarks. |

GPU implementations of primitive operations stay in this crate even when a higher-level crate is their main caller.

### `mxx-gadgets`

`crates/gadgets/` owns reusable cryptographic components assembled from primitives:

| Path | Responsibility |
| --- | --- |
| `crates/gadgets/src/bgg/` | BGG public keys, encodings, samplers, and vector/polynomial variants. |
| `crates/gadgets/src/circuit/` | Circuit IR, serialization, analysis, and evaluators. |
| `crates/gadgets/src/circuit_gadgets/` | Arithmetic, convolution, FHE/Ring-GSW, PRG, NTT, and other circuit gadgets. |
| `crates/gadgets/src/lookup/` | LWE and GGH15 lookup mechanisms. |
| `crates/gadgets/src/decoder/` | Decoder artifacts, mask circuits, PRG helpers, simulation, and estimates. |
| `crates/gadgets/src/noise_refresh/` | Reusable noise-refresh circuits and models. |
| `crates/gadgets/src/input_injector/` | Diamond input injection. |
| `crates/gadgets/src/slot_transfer/` | Public-key and encoding slot transfer. |
| `crates/gadgets/src/commit/` | Commitment components, including WEE25. |
| `crates/gadgets/src/bench_estimator/` | Reusable circuit and native-operation benchmark models. |
| `crates/gadgets/src/simulator/` | Norm, lattice-security, and shared application simulation utilities. |
| `crates/gadgets/src/storage/` | Artifact storage helpers. |

The former circuit-gadget directory is now `crates/gadgets/src/circuit_gadgets/` so that the crate name `mxx-gadgets` describes the full layer rather than one nested directory.

### Application crates

| Crate | Paths | Responsibility |
| --- | --- | --- |
| `mxx-func-enc` | `crates/func-enc/src/` | Functional-encryption trait and constructions. The AKY24 module remains disabled as it was before the split. |
| `mxx-we` | `crates/we/src/diamond_we*` | Witness-encryption trait, Diamond WE, simulation, and estimates. |
| `mxx-io` | `crates/io/src/aky24_io*`, `crates/io/src/diamond_io*` | iO trait, AKY24 iO models, Diamond iO, simulations, and estimates. |

## Features and native build ownership

All five crates define `disk` and `gpu` features. Upper crates only forward those features to their dependencies. `mxx-primitives` is the sole owner of:

- OpenFHE linkage;
- `cc` build dependencies;
- CUDA source compilation;
- `libc` and `memmap2` for primitive matrix backing.

Use workspace-wide commands when checking a cross-cutting change:

```sh
cargo check --workspace
cargo check --workspace --features disk
cargo check --workspace --features gpu
```

## Tests

Unit tests remain beside the implementation they test. Integration tests are grouped by owner:

- `crates/gadgets/tests/` for lookup, circuit-gadget, input-injector, and related regression tests;
- `crates/we/tests/` for Diamond WE;
- `crates/io/tests/` for AKY24 iO and Diamond iO.

Integration tests are not part of routine validation and must only be run when a task explicitly requests them. Compiling their targets with `cargo check --workspace --tests` is permitted for boundary validation.
