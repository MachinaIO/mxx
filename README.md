# mxx

**Write a lattice-based cryptographic protocol in a Rust-based DSL, much as you would write it on paper, and
run it on GPUs, with no CUDA and no hand-tuning of GPU memory or parallelism.**

> **Status:** research code under active development. It has not been audited.

## Why mxx

Lattice-based schemes spend most of their time on polynomial matrix arithmetic, gadget
decomposition, and preimage sampling, which is work that GPUs do well. Running such a scheme on a
GPU usually requires three kinds of work:

- hand-writing CUDA kernels for each operation,
- splitting the work into batches small enough to fit in GPU memory, and
- retuning those batch sizes whenever the parameters, the protocol, or the GPU changes.

With mxx you write only the protocol. You state its steps in terms of ring elements, matrices,
and samples, and mxx executes them on the GPU. It also chooses the degree of parallelism that fits
in GPU memory and runs fastest.

## What it looks like

The example below encrypts one bit `m` with Ring-LWE and decrypts it on the GPU. It computes
`b = a·s + e + floor(Q/2)·m`, where `a` is a public element derived from a seed, `s` is a
Gaussian secret, and `e` is Gaussian noise. It then recovers `m` by rounding the constant term
of `b - a·s`. Every parameter, such as the ring's CRT width and depth, the noise
width `sigma`, and the gadget base, is a named variable, and it gets a value only when the
program is bound for a run. The full program is
[`crates/backends/examples/rlwe_encrypt.rs`](crates/backends/examples/rlwe_encrypt.rs); run it
with `cargo run -r -p mxx-backends --example rlwe_encrypt --features gpu`.

```rust
use bigdecimal::BigDecimal;
use mxx_backends::{
    GpuRuntime, RuntimeValue, backend::poly_gpu::gpu_backend, poly::dcrt::gpu::GpuDCRTPolyParams,
    sampler::bounds::hard_cutoff_from_sigma_bound,
};
use mxx_dsl::{BuiltGraph, DslContext, DslError, HashTag, Int, IntType, Ring};
use mxx_ir_core::{IntExpr, ParamEnv, Rational, RealExpr, generate_crt_basis};
use num_bigint::BigInt;
use std::{collections::BTreeMap, sync::Arc};

/// Describes the protocol. Named parameters stay symbolic until they are bound.
fn rlwe_program(ring_dimension: u32) -> Result<BuiltGraph, DslError> {
    // R_Q = Z_Q[X]/(X^N + 1), where Q is a product of `crt_depth` primes of `crt_bits` bits.
    let ring = Ring::new(
        IntExpr::Var("crt_bits".into()),
        IntExpr::Var("crt_depth".into()),
        ring_dimension,
    );
    // Gaussian parameters: the width sigma and the bound on every sampled coefficient.
    let sigma = RealExpr::Var("sigma".into());
    let cutoff = IntExpr::Var("cutoff".into());
    let context = DslContext::new("rlwe-encrypt")
        .int_parameter("crt_bits")
        .int_parameter("crt_depth")
        .int_parameter("gadget_base_bits")
        .int_parameter("cutoff")
        .real_parameter("sigma");

    // Inputs supplied at run time: a 32-byte public seed and the message bit (0 or 1).
    let seed = ring.bytes_input("seed", 32);
    let bit: Int = context.input("bit", IntType)?;

    // The public element a is derived from the seed, so another party can recompute it.
    let a = ring.hash_matrix(seed, HashTag::from(b"rlwe-example/a".as_slice()), (1, 1));
    let s = ring.gaussian((1, 1), sigma.clone(), cutoff.clone());
    let e = ring.gaussian((1, 1), sigma, cutoff);

    // Encrypt the bit as the constant term: b = a*s + e + floor(Q/2) * bit.
    let delta = ring.polynomial([ring.modulus().floor_div(2)]);
    let message = bit.lift_to_constant_polynomial(ring.matrix_type((1, 1)));
    let b = &a * &s + e + &delta * &message;

    // Decrypt: round the constant term of b - a*s to the nearest multiple of Q/2.
    let decrypted = (b.clone() - &a * &s).threshold_decode_bools(2, 1).remove(0);

    context.output("ciphertext", b)?.output("decrypted", decrypted)?.build()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ring_dimension = 4096;
    let (crt_bits, crt_depth, gadget_base_bits) = (60, 3, 20);
    let sigma = 4;
    let cutoff = hard_cutoff_from_sigma_bound(&BigDecimal::from(sigma)); // 6.5 sigma

    // Bind the parameters, then check the whole program under them.
    let bindings = ParamEnv {
        integers: BTreeMap::from([
            ("crt_bits".to_owned(), BigInt::from(crt_bits)),
            ("crt_depth".to_owned(), BigInt::from(crt_depth)),
            ("gadget_base_bits".to_owned(), BigInt::from(gadget_base_bits)),
            ("cutoff".to_owned(), BigInt::from(cutoff)),
        ]),
        reals: BTreeMap::from([("sigma".to_owned(), Rational::from_integer(BigInt::from(sigma)))]),
        ..ParamEnv::default()
    };
    let program = rlwe_program(ring_dimension)?.validate(&bindings)?;

    // Register the same ring with a GPU backend.
    let moduli = generate_crt_basis(ring_dimension, crt_depth, crt_bits)?;
    let gpu_params = GpuDCRTPolyParams::new(ring_dimension, moduli, gadget_base_bits, None);
    let mut runtime = GpuRuntime::new(gpu_backend([gpu_params]))?;

    let inputs = |seed: u8, bit: bool| {
        BTreeMap::from([
            ("seed".to_owned(), RuntimeValue::Bytes(Arc::from([seed; 32]))),
            ("bit".to_owned(), RuntimeValue::Int(BigInt::from(bit))),
        ])
    };

    // Planning picks the parallelism and memory schedule that fit this GPU. It needs inputs of
    // the right types, so it gets dummy ones.
    let mut plan = runtime.plan(program, &inputs(0, false))?;

    // The plan then runs on real inputs without being planned again.
    for (seed, bit) in [(1, false), (2, true)] {
        let result = runtime.execute(&mut plan, inputs(seed, bit))?;
        assert_eq!(runtime.download_bool(&result["decrypted"])?, bit);
        println!("bit {bit} decrypted correctly");
    }
    Ok(())
}
```

Running `rlwe_program` does not compute anything. It records the protocol as a program: a graph
whose nodes are primitive operations, such as a matrix product, a Gaussian sample, or a
coefficient decoding, and whose edges carry values between them. Computation happens only when
a backend executes the graph. This is why one description can run on a GPU or on a CPU. Export
of the same graph to Lean, for machine-checked correctness proofs, is a work in progress.

## What mxx does for you

### Describing a protocol

- **Lattice objects as values.** You work with matrices of polynomials over RNS rings
  `Z_Q[X]/(X^N + 1)`, where Q is a product of word-sized primes. You also work with matrices of
  small (bounded) coefficients, lattice trapdoors and their preimages, integers, Booleans, byte
  strings, and indexed collections of these. `+`, `-`, and `*` behave as in the math.
- **Standard lattice operations.** mxx provides NTT-based polynomial arithmetic, gadget
  decomposition, ring automorphisms, multiplication by `X^k`, modulus switching and reduction,
  RNS basis extension and reduction (as used in hybrid key switching), CRT recomposition,
  coefficient extraction and packing, and threshold decoding.
- **Sampling built in.** You can draw uniform, interval, and discrete Gaussian samples, generate
  lattice trapdoors, and sample trapdoor preimages. You write a sample where the protocol needs
  it, and every run draws fresh randomness automatically. A sample written outside a loop is
  drawn once and shared, and a sample written inside a loop is drawn fresh in every iteration.
- **Seeds instead of large public matrices.** A matrix that is sampled from public random coins
  can instead be derived from a fixed-size random seed with a hash function (a random oracle).
  One party then sends only the 32-byte seed, not the matrix, and the other parties recompute the
  same matrix from it, which cuts communication. The derived values are identical on the CPU and
  the GPU.
- **Loops, choices, and reusable pieces.** You can loop over independent items (`parallel`), or
  repeat a step that updates a state, such as rounds of an evaluation (`iterate`). You can pick
  one of several computed values using a value known only at run time (`select`), and define a
  named block once and call it many times (`Subgraph`). Loops are recorded once, not copied per
  iteration, so a protocol with thousands of iterations stays a small program.
- **Parameters that vary.** Dimensions, loop counts, moduli, and sampler widths can be named
  parameters, so one protocol description serves many parameter sets.
- **Errors caught before running.** Before execution, mxx checks every step for matching shapes,
  rings, and bounds under the chosen parameters.

### Running it on GPUs

- **One description for GPU and CPU.** The same program runs on the GPU or on the CPU, so there
  is no separate GPU implementation to keep in sync.
- **Automatic tuning under a memory budget.** On a GPU, more parallelism, for example batching
  more matrix operations together, lowers latency but needs more GPU memory. mxx automatically
  chooses the parallelism with the lowest latency that fits within the memory of the machine it
  runs on.
- **Plan once, run many times.** Before the first run, mxx builds a *plan* for the program. The
  plan fixes the parallelism described above and a schedule for allocating and freeing GPU
  memory. Once the plan exists, you can run the program any number of times on new inputs with
  those tuned settings, without planning again.
- **Several GPUs.** A single plan can spread work over several GPUs. At present only large
  matrix products, preimage sampling, and independent loop iterations are spread; other steps
  run on one GPU. Multi-GPU execution has so far been tested only with several virtual devices on
  one physical GPU.
- **(Advanced) Your own CUDA kernel when it matters.** If part of a protocol needs a specialized
  kernel for better performance, you write only that kernel. The rest of the protocol stays in the
  DSL, and the automatic tuning of parallelism and memory scheduling still applies around your
  kernel, so you do not reimplement it. TFHE blind rotation uses this option.

### Protocols with several parties

- **Using other parties' outputs.** Real protocols run in stages, often by different parties:
  one party runs setup and publishes public keys, another encrypts, and a third decrypts. Each
  stage is its own program. A stage can export its results, and another party's stage can
  import them as inputs, from memory or from disk, loading only the parts it uses.

### Constructions included

| Crate | What you get |
| --- | --- |
| `mxx-fhe` | TFHE with NAND bootstrapping, and leveled BGV with SIMD slots, rotations, relinearization, hybrid RNS key switching, and noise tracking. |
| `mxx-bgg` | BGG+ public keys and encodings, circuit evaluation, LWE lookup tables, slot transfer, Tall encodings, and WEE25 commitments. |
| `mxx-gadgets` | Circuit models and reusable building blocks: nested-RNS arithmetic, NTT circuits, Ring-GSW, a Goldreich PRG, and noise refresh. |
| `mxx-we` | Diamond witness encryption with a Lean-checked parameter search. Temporarily disabled; see [Workspace layout](#workspace-layout). |

For the design in depth, including how the GPU runtime plans and runs a program and its current
limitations, start with `docs/architecture.md`.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `mxx-ir-core` | Executable graph IR, protocol declarations, validation, artifact manifests, and Lean claim generation. |
| `mxx-dsl` | Typed graph construction and sampler-free ideal and predicate builders. |
| `mxx-backends` | Polynomial and matrix operations, samplers, the CPU executor, the GPU runtime and native CUDA, transcripts, sessions, and artifacts. |
| `mxx-gadgets` | BGG-independent circuits and reusable circuit gadgets. |
| `mxx-bgg` | BGG+ keys, encodings, sampling, evaluation, decoding, lookup, slot transfer, and refresh. |
| `mxx-fhe` | DSL-based TFHE with NAND bootstrapping, and leveled BGV with SIMD, rotations, and ciphertext noise tracking. |
| `mxx-we` | Witness-encryption interfaces and parameterized dynamic-circuit Diamond WE. Temporarily disabled: it is excluded from the workspace until its protocol family is redesigned, and builds only with `--manifest-path crates/we/Cargo.toml`. |
| `mxx-func-enc`, `mxx-io` | Functional-encryption and iO interfaces; protocol implementations have been removed. |

There is no symbolic IR and no probabilistic noise simulator. Correctness uses enforced integer
coefficient cutoffs and deterministic worst-case bounds. CPU and GPU samplers enforce the same
cutoffs: Gaussian draws are resampled until they meet the bound, and preimage sampling retries
whole candidates that exceed it. Lattice-security estimation intentionally continues to model
the corresponding ordinary untruncated distributions separately.

## Requirements and building

- Rust with edition 2024 support.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature (`CUDA_ARCH` defaults to `89`).

```sh
cargo test -r --workspace --lib
cargo test -r --workspace --lib --features gpu
```
