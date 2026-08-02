# mxx

`mxx` is a Rust and CUDA workspace for lattice-cryptography research and implementation work at Machina iO. It contains low-level polynomial, matrix, sampling, and GPU operations; reusable cryptographic gadgets; and application-level FE, WE, and iO constructions.

The repository includes:

- [BGG+ encodings](https://eprint.iacr.org/2014/356.pdf) as declarative DSL programs in `crates/bgg/`.
- Evaluation and decryption of [GSW-FHE](https://eprint.iacr.org/2013/340.pdf) over BGG+ encodings, following [this construction](https://eprint.iacr.org/2015/029.pdf), in `crates/gadgets/src/circuit_gadgets/fhe/`.
- [Diamond witness encryption](https://eprint.iacr.org/2025/375) as declarative encryption and decryption graphs, with symbolic noise simulation, automatic parameter search, and CPU/GPU execution, in `crates/we/`. Indistinguishability-obfuscation implementations remain disabled pending their DSL cutover.

## Workspace layout

The repository is a virtual Cargo workspace with no root facade crate:

| Crate | Responsibility |
| --- | --- |
| `mxx-ir-core` | Executable typed graph IR, exact compile expressions, concrete validation, and artifact manifests. |
| `mxx-dsl` | Typed declarative construction API over immutable core DAG nodes. |
| `mxx-ir-symbolic` | Typed symbolic-expression elaboration, targeted rewrite, and cross-graph identity operations over Graph IR. |
| `mxx-noise-simulator` | Numerical noise analysis over elaborated symbolic IR using the existing polynomial and matrix norm rules. |
| `mxx-runtime` | CPU/GPU execution, reproducible sampling transcripts, sessions, and in-memory artifacts. |
| `mxx-bench-estimator` | Binding-sensitive measured graph-cost composition, critical paths, parallel waves, and memory peaks. |
| `mxx-primitives` | Polynomial and matrix representations, samplers, analytical sampling bounds, OpenFHE integration, and all native CUDA kernels and wrappers. |
| `mxx-gadgets` | BGG-independent circuits and circuit gadgets. |
| `mxx-bgg` | Declarative BGG+ public keys, encodings, samplers, polynomial/naive families, circuit evaluation, masked decoding, slot transfer, artifacts, and noise refresh. |
| `mxx-func-enc` | Functional-encryption interfaces. AKY24 functional encryption is disabled pending a separate specification of its raw-mask semantics. |
| `mxx-we` | Diamond witness encryption: declarative preprocessing, encryption and decryption graphs, symbolic noise simulation, parameter search, cost estimation, and CPU/GPU runtime integration. |
| `mxx-io` | Indistinguishability-obfuscation interfaces. AKY24 iO and Diamond iO are disabled pending separate application cutovers. |

The principal dependency directions are shown with consumers on the left:

```text
mxx-runtime          -> mxx-ir-core, mxx-primitives
mxx-ir-symbolic      -> mxx-ir-core
mxx-dsl              -> mxx-ir-core, mxx-ir-symbolic
mxx-noise-simulator -> mxx-ir-core, mxx-ir-symbolic, mxx-primitives
mxx-bench-estimator  -> mxx-ir-core, mxx-runtime
mxx-gadgets          -> mxx-dsl, mxx-ir-core, mxx-primitives
mxx-bgg              -> mxx-dsl, mxx-gadgets, mxx-ir-core
mxx-func-enc/we/io   -> lower layers, never one another
```

The three application crates do not depend on one another. See `docs/architecture.md` for the detailed directory map and boundary rules.

## Requirements

- Rust with edition 2024 support.
- OpenFHE C++ libraries installed in the system location expected by `crates/primitives/build.rs`. Follow the [OpenFHE installation guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html), but use [MachinaIO/openfhe-development](https://github.com/MachinaIO/openfhe-development) instead of the upstream repository.
- OpenMP support through the system C/C++ toolchain.
- For GPU builds, a CUDA toolkit with `nvcc` and the CUDA runtime libraries.

## Cargo features

Crates with device-specific behavior expose the relevant opt-in feature name.
Application crates forward it to their lower-level dependencies.

| Feature | Effect |
| --- | --- |
| `gpu` | Enables CUDA-backed primitive operations and GPU implementations in dependent crates. Native CUDA compilation is owned by `mxx-primitives`. |

GPU build configuration is handled by `crates/primitives/build.rs` and `crates/primitives/cuda/`.

| Variable | Purpose |
| --- | --- |
| `CUDA_ARCH` | CUDA SM architecture passed to `nvcc`. |
| `CUDA_HOME` | CUDA installation root used to locate `nvcc` and libraries. |
| `CUDA_LIB_DIR` | CUDA library directory linked by Cargo. |
| `NVCC` | Explicit CUDA compiler path. |

Primitive-operation environment configuration lives in
`crates/primitives/src/env.rs`. Graph scheduling and batching are configured
through the explicit runtime and estimator APIs rather than gadget-specific
environment variables.

## Common commands

Type-check the complete CPU workspace:

```sh
cargo check --workspace
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
