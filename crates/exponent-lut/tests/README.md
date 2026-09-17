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
GPU-visible environment. The fixtures are intentionally small and validate
functional dataflow; they do not establish lattice-security claims.

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

The NAND test compares the returned encoding with an independent Boolean
NAND oracle. The PRF test uses a fixed canonical label and independently
computes the sequential `D*T` exponent and readout before verifying the
caller-owned output. The refresh test independently computes the clear PRF
state/readout, checks the three production graphs in order, and verifies the
refreshed output with caller-owned secrets. Correctness-oracle calculations
are kept outside the graphs passed to the estimator.

No stage-selection environment variable or alternate scheme selector is
supported by these targets. Internal structural regressions belong to the
crate's unit tests; these external targets exercise only the public,
artifact-backed workflow.
