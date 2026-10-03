# Exponent-LUT integration tests

The integration suite contains exactly three GPU-gated targets. Each target
contains exactly one sequential test and runs the complete production flow:
noise simulation, benchmark estimation, then an artifact-backed preprocess,
encrypt, and online round trip. Benchmark estimation is the GPU runtime's
planning step: every stage graph is planned with `GpuRuntime` immediately
before it executes, the planner measures its wave-width and column-tile
candidates on that exact graph, and the selected `GpuWarmupReport` is logged
at `info` level (run with `RUST_LOG=info` to see it).

| File | Test | Application |
| --- | --- | --- |
| `nand_circuit.rs` | `nand_circuit_end_to_end` | NAND circuit |
| `lwr_prf.rs` | `lwr_prf_end_to_end` | Sparse-LWR PRF |
| `noise_refresh.rs` | `noise_refresh_end_to_end` | Noise refreshing |

All three workflows use the public RNS-BGG application API and the
`dcrt_runtime` sequential lock. They require the `gpu` feature and a
GPU-visible environment. Before execution, the NAND and Sparse-LWR targets
search their configured CRT-depth and ring-dimension candidates, validate the
production noise report, and retain only a candidate that reaches its
configured positive lattice-security target through `lattice-estimator-cli`.
The refresh target likewise checks its production noise/transcript contracts
before verifying the refreshed output. The estimator is required for these
security gates; the only mode that skips it is the NAND noiseless diagnostic
described below, which makes no security claim.

Parameter controls are namespaced per target rather than shared globally:
`MXX_EXPONENT_LUT_NAND_*` controls the NAND workflow and
`MXX_EXPONENT_LUT_LWR_PRF_*` controls Sparse-LWR, while
`MXX_EXPONENT_LUT_NOISE_REFRESH_*` controls noise refreshing. Their
ring-dimension settings specify inclusive logarithmic bounds and their
security settings specify the required target. Search uses 40 CRT bits by
default and derives the gadget base as `ceil(CRT_BITS/2)`, subject to the
backend limits. Malformed, non-positive, or inconsistent values are rejected.

Parameter search uses the average-case noise model by default. For each
target, setting its namespaced `*_NOISE_MODEL=worst` selects the worst-case
model. In average-case mode, the matching namespaced
`*_FAILURE_EXPONENT` controls the failure-event budget. These settings are
part of the accepted noise report and are not silently changed by the
round-trip stage.

Compile all three targets without running them:

```sh
cargo test -r -p mxx-exponent-lut --features gpu \
  --test nand_circuit --test lwr_prf --test noise_refresh --no-run
```

Run one target when GPU execution is explicitly authorized:

```sh
cargo test -r -p mxx-exponent-lut --features gpu \
  --test nand_circuit nand_circuit_end_to_end -- --nocapture
cargo test -r -p mxx-exponent-lut --features gpu \
  --test lwr_prf lwr_prf_end_to_end -- --nocapture
cargo test -r -p mxx-exponent-lut --features gpu \
  --test noise_refresh noise_refresh_end_to_end -- --nocapture
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

The PRF target uses a fixed canonical label and independently computes the
sequential `D*T` exponent and readout before verifying the caller-owned
output. The refresh target independently computes
the clear PRF state/readout, checks the three production graphs in order, and
verifies the refreshed output with caller-owned secrets. Correctness-oracle
calculations remain outside the production graphs.

There is no alternate scheme selector or stage-skip switch. Internal
structural regressions belong to the crate's unit tests; these external
targets exercise the public, artifact-backed workflow and its noise/security
acceptance gates.

## Recorded Sparse-LWR parameters

`lwr_prf_parameters.csv` records parameter sets found for `lwr_prf.rs`, one
row per set, with the date and the commit the simulation ran on. Every row
passed the simulation-only run (`MXX_EXPONENT_LUT_LWR_PRF_SIMULATION_ONLY=1`)
at a 100-bit target: the average-case noise audit and the encoding lattice
security. None of them has run on a GPU. The columns are:

| Column | Meaning |
| --- | --- |
| `ring_dimension`, `crt_bits`, `base_bits`, `crt_depth`, `primes_per_drop`, `levels` | RNS-BGG encoding parameters of the accepted candidate |
| `encoding_security_bits` | Lattice security of the encoding from `lattice-estimator-cli` |
| `noise_model`, `failure_exponent` | Noise model of the audit and, for `average`, its failure budget `2^-failure_exponent` |
| `lwr_dimension`, `support_weight`, `input_modulus`, `output_modulus` | PRF parameters ν, h, Q, P |
| `prf_min_estimator_bits` | Minimum over the eight attacks of `sparse_lwr_estimator.sage.py` |
| `prf_mitm_bits`, `prf_binding_attack` | Estimator `LWE.mitm` with the split dimension rounded to an integer (it raises `TypeError` on sparse secrets otherwise), and the cheapest attack overall |
| `level0_matrix_gb`, `stored_artifacts_tb` | One level-0 `(1, columns)` matrix, and the RHS, input-encoding, and public-key artifacts the implementation at `git_commit` stores, estimated from matrix sizes |

## Parameter IDs for the Lean correctness proofs

With `MXX_EXPONENT_LUT_NAND_PARAMETER_ID` or `MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID` set to a
parameter ID, `nand_circuit.rs` and `lwr_prf.rs` run without searching at the parameters of that row
of `lean/<test>/parameters.csv` (the noise model, the RNS-BGG parameters, the error width, and for
the PRF the LWR profile and the key seed). They check that the recorded lattice security is
reproduced, verify the round trip and the decrypted output, and export the graphs of the executed
stages as a closed protocol to `lean/<test>/<parameter id>/generated`. See `lean/README.md` for the
meaning of each column and the proofs of the exported claims.

`lwr_prf.rs` with `MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY=1` as well builds and validates the same
stage graphs without a round trip: only the secret sampling executes, on the CPU, and the claim is
exported. It exports parameter sets whose round trip does not fit the local GPU; the stage graphs,
and so the claim, do not depend on whether the stages executed.

```sh
MXX_EXPONENT_LUT_NAND_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test nand_circuit -- --nocapture
MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID=average-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test lwr_prf -- --nocapture
```
