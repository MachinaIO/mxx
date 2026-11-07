# Exponent-LUT integration tests

The integration suite contains exactly three GPU-gated targets. Each target
contains exactly one sequential test and runs the complete production flow:
parameter search with noise simulation, benchmark estimation, then an
artifact-backed round trip (preprocess, encrypt, and online evaluation). Benchmark estimation is the GPU runtime's
planning step: every stage graph is planned with `GpuRuntime` immediately
before it executes, the planner measures its wave-width and column-tile
candidates on that exact graph, and the selected `GpuWarmupReport` is logged
at `info` level (run with `RUST_LOG=info` to see it).

| File | Test | Application |
| --- | --- | --- |
| `nand_circuit.rs` | `nand_circuit_end_to_end` | NAND circuit |
| `public_ip.rs` | `public_ip_end_to_end` | Public inner product |
| `turboshake128.rs` | `turboshake128_label_hash_round_trip` | Counter-mode TurboSHAKE128 label hash over public sign encodings |

The workflows use the public RNS-BGG application API (the TurboSHAKE128 target
searches only its CRT depth at a toy ring dimension, without a lattice-security gate), and all three
take the `dcrt_runtime` sequential lock. They require the `gpu` feature and a
GPU-visible environment. Before execution, the NAND target searches its
configured CRT-depth and ring-dimension candidates, validates the production
noise report, and retains only a candidate that reaches its configured positive
lattice-security target through `lattice-estimator-cli`. The estimator is
required for this security gate; the only mode that skips it is the NAND
noiseless diagnostic described below, which makes no security claim.

Parameter controls are namespaced per target rather than shared globally:
`MXX_EXPONENT_LUT_NAND_*` controls the NAND workflow,
`MXX_EXPONENT_LUT_PUBLIC_IP_*` controls the public inner product, and
`MXX_EXPONENT_LUT_TURBOSHAKE128_*` the label hash (their rustdoc lists the
variables). Their ring-dimension settings specify inclusive logarithmic bounds
and their security settings specify the required target. Search uses 40 CRT
bits by default and derives the gadget base as `ceil(CRT_BITS/2)`, subject to the
backend limits. Malformed, non-positive, or inconsistent values are rejected.

Parameter search uses the average-case noise model by default. For each
target, setting its namespaced `*_NOISE_MODEL=worst` selects the worst-case
model. In average-case mode, the matching namespaced
`*_FAILURE_EXPONENT` controls the failure-event budget. These settings are
part of the accepted noise report and are not silently changed by the
round-trip stage.

The NAND target runs the Static RHS lifecycle by default. Setting
`MXX_EXPONENT_LUT_NAND_RHS_MODE=dynamic` selects the Dynamic lifecycle:
preprocessing receives no secret and no RHS plaintexts, the single secret is
sampled afterwards, and encryption binds the RHS plaintexts with a fresh batch
identifier. The noise model applies to both modes. For quick iterations, fix
small parameters and keep the stage artifacts in GPU memory with
`*_STORAGE=device`.

Compile all three targets without running them:

```sh
cargo test -r -p mxx-exponent-lut --features gpu \
  --test nand_circuit --test public_ip --test turboshake128 --no-run
```

Run one target when GPU execution is explicitly authorized:

```sh
cargo test -r -p mxx-exponent-lut --features gpu \
  --test nand_circuit nand_circuit_end_to_end -- --nocapture
cargo test -r -p mxx-exponent-lut --features gpu \
  --test public_ip public_ip_end_to_end -- --nocapture
cargo test -r -p mxx-exponent-lut --features gpu \
  --test turboshake128 turboshake128_label_hash_round_trip -- --nocapture
```

The NAND target builds a complete binary NAND tree whose depth is controlled
by `MXX_EXPONENT_LUT_NAND_DEPTH`; its LUT-round budget is checked before
compilation. Because one round trip takes minutes at production parameters,
each run evaluates a single assignment whose every input bit is drawn at
random (it is printed as `NAND input assignment`). The output is compared
with an independent Boolean NAND oracle, and the same output must be rejected
as the opposite bit. `MXX_EXPONENT_LUT_NAND_ERROR_SIGMA`
sets the encoding-error width (default 4); `0` is a noiseless diagnostic
mode that samples and adds no encoding error and skips the lattice-estimator
gate, so it checks correctness only.

The 128-bit, depth-1 NAND round trip was verified on one 16 GiB GPU with the
following command, which selected `n=16384`, CRT depth 8, and two primes per
drop. With three fixed assignments it took about 30 minutes, most of it spent
planning the preprocess graph once per assignment:

```sh
MXX_GPU_MEASUREMENT_WARMUPS=0 MXX_GPU_MEASUREMENT_ITERATIONS=1 \
MXX_EXPONENT_LUT_NAND_MIN_LOG_RING_DIMENSION=14 \
MXX_EXPONENT_LUT_NAND_MAX_LOG_RING_DIMENSION=14 \
MXX_EXPONENT_LUT_NAND_DEPTH=1 MXX_EXPONENT_LUT_NAND_SECURITY_BITS=128 \
MXX_EXPONENT_LUT_NAND_CRT_BITS=40 \
cargo test -r -p mxx-exponent-lut --features gpu \
  --test nand_circuit nand_circuit_end_to_end -- --nocapture
```

GPU memory controls that affect these targets are `MXX_GPU_MEMORY_FRACTION`
(share of device memory for a plan's persistent allocations, default 0.8) and
`MXX_GPU_SMALL_RHS_CHUNK_COLUMNS` (small-RHS multiplication chunk width,
default 16); see `docs/architecture.md` section 6.1.

The public inner-product target draws `a` and the hash seed of `t` from
`MXX_EXPONENT_LUT_PUBLIC_IP_SEED` (reported when fresh), computes `<a, t> mod Q`
in the clear, verifies the public output against it and rejects its successor,
and checks the output with `decode_public_output`, whose decoded residual count
must be zero. Correctness-oracle calculations remain outside the production
graphs.

There is no alternate scheme selector or stage-skip switch. Internal
structural regressions belong to the crate's unit tests; these external
targets exercise the public, artifact-backed workflow and its noise/security
acceptance gates.

## Parameter IDs for the Lean correctness proofs

With `MXX_EXPONENT_LUT_NAND_PARAMETER_ID` or `MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID` set to a
parameter ID, `nand_circuit.rs` and `public_ip.rs` run without searching at the parameters of that
row of `lean/<test>/parameters.csv` (the noise model, the RNS-BGG parameters, the error width, and
for the public inner product the input modulus, the dimension, and the seed). They check that the
recorded lattice security is reproduced, verify the round trip and the decrypted output, and export
the graphs of the executed stages as a closed protocol to `lean/<test>/<parameter id>/generated`.
See `lean/README.md` for the meaning of each column and the proofs of the exported claims. A NAND
parameter ID runs the RHS lifecycle of its row's `rhs_mode` and ignores
`MXX_EXPONENT_LUT_NAND_RHS_MODE`; a Dynamic row exports the stages `payload`, `encrypt`,
`online_eval`, and `decode`. A public inner-product parameter ID exports the stages `mask`,
`encrypt`, `online_eval`, and `decode`: with public inputs only, online evaluation derives every
public key from the hash key, so no stage reads the preprocessing artifacts or the payload secret.

```sh
MXX_EXPONENT_LUT_NAND_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test nand_circuit -- --nocapture
MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID=average-n32-crt2x40 \
  cargo test -r -p mxx-exponent-lut --features gpu --test public_ip -- --nocapture
```
