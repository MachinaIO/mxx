# Exponent-LUT integration tests

The integration suite contains exactly three GPU-gated targets. Each target
contains exactly one sequential test and runs the complete production flow:
noise simulation, benchmark estimation, then an artifact-backed preprocess,
encrypt, and online round trip.

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
security gates; there is no successful skip-Sage mode.

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
compilation. It compares every selected assignment with an independent
Boolean NAND oracle. The PRF target uses a fixed canonical label and
independently computes the sequential `D*T` exponent and readout before
verifying the caller-owned output. The refresh target independently computes
the clear PRF state/readout, checks the three production graphs in order, and
verifies the refreshed output with caller-owned secrets. Correctness-oracle
calculations remain outside the graphs passed to the estimator.

There is no alternate scheme selector or stage-skip switch. Internal
structural regressions belong to the crate's unit tests; these external
targets exercise the public, artifact-backed workflow and its noise/security
acceptance gates.
