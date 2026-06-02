# mxx

`mxx` is a Rust and CUDA library for lattice-cryptography research and implementation work at Machina iO. It provides primitive-level building blocks for polynomial and matrix arithmetic, trapdoor sampling, key-homomorphic encodings, and higher-level constructions of lattice-based schemes.

Specifically, this repository provides implementations of the following works:
- [BGG+ encodings](https://eprint.iacr.org/2014/356.pdf), available at `src/bgg/`
- [WEE25 matrix commitment](https://eprint.iacr.org/2025/509.pdf), available at `src/commit/`.
- [Lookup-table evaluation over BGG+ encodings](https://eprint.iacr.org/2025/1870.pdf) available at `src/lookup/`.
- Evaluation and decryption of [GSW-FHE](https://eprint.iacr.org/2013/340.pdf) over BGG+ encodings, following [this construction](https://eprint.iacr.org/2015/029.pdf), available at `src/gadgets/fhe/`.
- Benchmark estimation for pseudorandom obfuscation based on [AKY24](https://eprint.iacr.org/2024/1720.pdf), available at `src/io/aky24_io/`.
- Benchmark estimation for [Diamond iO](https://eprint.iacr.org/2025/236.pdf), available at `src/io/diamond_io/`.


## Requirements

- Rust with support for edition 2024.
- OpenFHE C++ libraries installed in the system location expected by `build.rs`. Follow the [OpenFHE installation guide](https://openfhe-development.readthedocs.io/en/latest/sphinx_rsts/intro/installation/installation.html), but run those steps against our fork, [MachinaIO/openfhe-development](https://github.com/MachinaIO/openfhe-development), rather than the upstream OpenFHE repository. The build script links `OPENFHEpke`, `OPENFHEbinfhe`, and `OPENFHEcore`.
- OpenMP support through the system C/C++ toolchain.
- For GPU builds, a CUDA toolkit with `nvcc` and the CUDA runtime libraries.

## Cargo Features

The default feature set is CPU-only and keeps matrix backing storage in memory.

| Feature | Effect |
| --- | --- |
| `disk` | Enables disk-backed matrix storage through `libc` and `memmap2`. Without this feature, `src/matrix/base/memory.rs` is used. |
| `gpu` | Enables CUDA-backed polynomial/matrix paths, GPU samplers, GPU lookup and BGG paths, GPU-aware circuit evaluation, and native CUDA compilation from `cuda/`. |

GPU builds use these environment variables. Their defaults are defined in `build.rs`.

| Variable | Purpose |
| --- | --- |
| `CUDA_ARCH` | CUDA SM architecture passed to `nvcc`. |
| `CUDA_HOME` | CUDA installation root used to locate `nvcc` and libraries. |
| `CUDA_LIB_DIR` | CUDA library directory linked by Cargo. |
| `NVCC` | Explicit CUDA compiler path. |

Runtime parallelism and batching are also controlled by environment helpers in `src/env.rs`, including `MXX_CIRCUIT_PARALLEL_GATES`, `LUT_PREIMAGE_CHUNK_SIZE`, `GGH15_GATE_PARALLELISM`, `BGG_POLY_ENCODING_SLOT_PARALLELISM`, `SLOT_TRANSFER_SLOT_PARALLELISM`, `AUX_SAMPLING_CHUNK_WIDTH`, and `MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH`.

<!-- ## Repository Layout

The public Rust surface is organized from `src/lib.rs`:

| Path | Main responsibility |
| --- | --- |
| `src/poly/` | Polynomial traits and DCRT polynomial implementations backed by OpenFHE parameters and CRT decomposition. |
| `src/matrix/` | Generic matrix traits, integer and DCRT-polynomial matrices, memory/disk storage, and GPU DCRT matrix support behind `gpu`. |
| `src/sampler/` | Hash, uniform, and trapdoor samplers, including OpenFHE-backed Gaussian routines and GPU trapdoor paths. |
| `src/bgg/` | BGG-style encodings, public keys, polynomial encodings, digit conversion, and sampler integration. |
| `src/lookup/` | Public lookup-table machinery, LWE and GGH15 lookup encodings, commit/eval helpers, and debug tooling. |
| `src/circuit/` | Arithmetic gate definitions, evaluable polynomial objects, serialized circuit support, and polynomial-circuit construction/evaluation. |
| `src/gadgets/` | Arithmetic, convolution multiplication, FHE/RingGSW, Goldreich PRG, and secret inner-product gadgets. |
| `src/func_enc/` | Functional-encryption workflows, currently including AKY24 key generation, decryption benchmarking, and error simulation helpers. |
| `src/io/` | Diamond iO and AKY24 IO workflows, simulations, benchmark estimators, and circuit utilities. |
| `src/we/` | Witness-encryption workflows, including Diamond WE simulation and benchmark estimation. |
| `src/decoder/` | Decoder artifacts, masked high-bit decoding, mask circuits, PRG support, simulation, and benchmark support. |
| `src/noise_refresh/` | Circuit decrypt, merge, PRG, naive-vector, and simulation components for noise refresh. |
| `src/input_injector/` | Input injection and Diamond GPU injection paths. |
| `src/slot_transfer/` | Slot-transfer BGG public keys, polynomial encodings, and polynomial-vector helpers. |
| `src/commit/` | Commitment schemes, including WEE25. |
| `src/bench_estimator/` | Estimators for BGG encodings, BGG public keys, naive vectors, and GPU-aware costs. |
| `src/simulator/` | Lattice estimation, norm estimation, and evaluation-error simulation tools. |
| `src/storage/` | Binary read/write helpers for repository data artifacts. |
| `cuda/` | CUDA runtime, ChaCha, and matrix kernels for arithmetic, NTT, decomposition, sampling, trapdoor, and serialization paths. |
| `benches/` | CPU/GPU matrix multiplication and preimage benchmarks declared in `Cargo.toml`. |
| `tests/` | Integration-style regression and GPU tests. Run these only when a task explicitly calls for them. | -->

## Common Commands

CPU-only type checking:

```sh
cargo check
```

Type check with disk-backed storage:

```sh
cargo check --features disk
```

Type check with CUDA support:

```sh
cargo check --features gpu
```

Run a targeted unit test by name when validating a narrow change:

```sh
cargo test <test_name>
```

Format Rust code when Rust files are changed:

```sh
cargo +nightly fmt --all
```

Run benchmarks explicitly by bench target:

```sh
cargo bench --bench bench_matrix_mul_cpu
cargo bench --bench bench_preimage_cpu
cargo bench --features gpu --bench bench_matrix_mul_gpu
cargo bench --features gpu --bench bench_preimage_gpu
```

## License

The Cargo manifest declares `MIT OR Apache-2.0`. See `LICENSE` for the checked-in license text.
