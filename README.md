# mxx

`mxx` is a Rust and CUDA workspace for lattice-cryptography research and implementation work at Machina iO. It contains low-level polynomial, matrix, sampling, and GPU operations; reusable cryptographic gadgets; and application-level FE, WE, and iO constructions.

The repository includes:

- [BGG+ encodings](https://eprint.iacr.org/2014/356.pdf) in `crates/gadgets/src/bgg/`.
- [WEE25 matrix commitment](https://eprint.iacr.org/2025/509.pdf) in `crates/gadgets/src/commit/`.
- [Lookup-table evaluation over BGG+ encodings](https://eprint.iacr.org/2025/1870.pdf) in `crates/gadgets/src/lookup/`.
- Evaluation and decryption of [GSW-FHE](https://eprint.iacr.org/2013/340.pdf) over BGG+ encodings, following [this construction](https://eprint.iacr.org/2015/029.pdf), in `crates/gadgets/src/circuit_gadgets/fhe/`.
- Diamond witness encryption in `crates/we/`.
- Benchmark estimation for pseudorandom obfuscation based on [AKY24](https://eprint.iacr.org/2024/1720.pdf) in `crates/io/src/aky24_io/`.
- Benchmark estimation for [Diamond iO](https://eprint.iacr.org/2025/236.pdf) in `crates/io/src/diamond_io/`.

## Workspace layout

The repository is a virtual Cargo workspace with no root facade crate:

| Crate | Responsibility |
| --- | --- |
| `mxx-ir-core` | Executable typed graph IR, exact compile expressions, concrete validation, and artifact manifests. |
| `mxx-ir-symbolic` | Optional symbolic-term elaboration, rewrite, and cross-graph identity operations over Graph IR. |
| `mxx-noise-simulator` | Numerical noise analysis over elaborated symbolic IR using the existing polynomial and matrix norm rules. |
| `mxx-runtime` | CPU/GPU execution, reproducible sampling transcripts, liveness, and indexed artifact-family persistence. |
| `mxx-bench-estimator` | Binding-sensitive measured graph-cost composition, critical paths, parallel waves, and memory peaks. |
| `mxx-primitives` | Polynomial and matrix representations, samplers, analytical sampling bounds, OpenFHE integration, and all native CUDA kernels and wrappers. |
| `mxx-gadgets` | BGG encodings, circuits, circuit gadgets, lookup, decoding, noise refresh, input injection, slot transfer, commitments, storage, simulation, and benchmark models. |
| `mxx-bgg` | Graph compilers for BGG+ wire bundles and `PolyCircuit`, with explicit scheme contexts for lookup and slot-transfer lowering. |
| `mxx-func-enc` | Functional-encryption interfaces and constructions. AKY24 remains disabled until its shared-decoder migration is complete. |
| `mxx-we` | Witness-encryption interfaces and Diamond WE. |
| `mxx-io` | Indistinguishability-obfuscation interfaces, AKY24 iO estimation, and Diamond iO. |

The principal dependency directions are shown with consumers on the left:

```text
mxx-runtime          -> mxx-ir-core, mxx-primitives
mxx-ir-symbolic    -> mxx-ir-core
mxx-noise-simulator -> mxx-ir-symbolic, mxx-primitives
mxx-bench-estimator  -> mxx-ir-core, mxx-runtime
mxx-gadgets          -> mxx-primitives, mxx-noise-simulator
mxx-bgg              -> mxx-ir-core, mxx-gadgets
mxx-func-enc/we/io   -> lower layers, never one another
```

The three application crates do not depend on one another. See `docs/architecture.md` for the detailed directory map and boundary rules.

## Requirements

- Rust with edition 2024 support.
- OpenFHE C++ libraries installed in the system location expected by `crates/primitives/build.rs`. Follow the [OpenFHE installation guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html), but use [MachinaIO/openfhe-development](https://github.com/MachinaIO/openfhe-development) instead of the upstream repository.
- OpenMP support through the system C/C++ toolchain.
- For GPU builds, a CUDA toolkit with `nvcc` and the CUDA runtime libraries.

## Cargo features

Crates with concrete storage or device behavior expose the relevant opt-in
feature names. Application crates forward them to their lower-level
dependencies.

| Feature | Effect |
| --- | --- |
| `disk` | Enables disk-backed primitive matrix storage through `libc` and `memmap2`. |
| `gpu` | Enables CUDA-backed primitive operations and GPU implementations in dependent crates. Native CUDA compilation is owned by `mxx-primitives`. |

GPU build configuration is handled by `crates/primitives/build.rs` and `crates/primitives/cuda/`.

| Variable | Purpose |
| --- | --- |
| `CUDA_ARCH` | CUDA SM architecture passed to `nvcc`. |
| `CUDA_HOME` | CUDA installation root used to locate `nvcc` and libraries. |
| `CUDA_LIB_DIR` | CUDA library directory linked by Cargo. |
| `NVCC` | Explicit CUDA compiler path. |

Runtime parallelism and batching helpers are split between `crates/primitives/src/env.rs` for primitive operations and `crates/gadgets/src/env.rs` for gadget-level workloads.

## Common commands

Type-check the complete CPU workspace:

```sh
cargo check --workspace
```

Type-check all crates with disk-backed storage:

```sh
cargo check --workspace --features disk
```

Type-check all crates with CUDA support:

```sh
cargo check --workspace --features gpu
```

Run a targeted unit test in its owning crate:

```sh
cargo test -p mxx-gadgets --lib <test_name>
```

Format Rust code:

```sh
cargo +nightly fmt --all
```

Run primitive benchmarks explicitly:

```sh
cargo bench -p mxx-primitives --bench bench_matrix_mul_cpu
cargo bench -p mxx-primitives --bench bench_preimage_cpu
cargo bench -p mxx-primitives --features gpu --bench bench_matrix_mul_gpu
cargo bench -p mxx-primitives --features gpu --bench bench_preimage_gpu
```

Integration tests live under each owning crate's `tests/` directory. Run them only when the task explicitly requires them.

## License

The workspace crates are licensed under `MIT OR Apache-2.0`. See `LICENSE`.
