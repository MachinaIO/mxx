# Fused BGV RNS conversion

`RnsModUp` and `RnsModDown` preserve the centered CRT sums of ePrint 2021/204,
Appendix B.2.3. Each input is inverse-transformed once; CRT contributions are
accumulated before each output digit is forward-transformed. CPU plans cache
conversion constants. CUDA caches normalization/division constants on the host
and computes cofactor weights with a setup kernel; all matrix coefficients and
temporary conversion tables remain on the device. Kernel arguments are copied
by value, and stream events protect source and output lifetimes.

The change also replaces upstream `GetMatrixElement` with a direct native
polynomial copy. The upstream accessor interpolated CRT coefficients and used
process-global NTT caches; a repeated CPU oracle test crashed in that transform
path. Direct copies preserve the ordered basis and representation format and
remove that extra work without serializing matrix operations.

## Measurements

Baseline: the composed Hybrid RNS implementation before this optimization.
Final: fused conversion plus the direct matrix-entry copy. Results measure their
combined effect, not fusion alone. Machine: AMD Ryzen 9 7950X3D, release build.
Ciphertext basis: three 30-bit CRT primes; one 60-bit auxiliary prime and three
digits. These are test configurations, not security-selected parameters.

The production key-switch evaluator reuses generated keys and an input matrix.
Each time includes runtime execution and output materialization, excludes key
generation, graph construction, and backend creation, and follows one warmup.
Medians below are computed from 20 samples, averaging the two middle values.

| Ring dimension | Before (seconds) | After (seconds) | Speedup |
| ---: | ---: | ---: | ---: |
| 8 | 0.005471305 | 0.000764703 | 7.15x |
| 1024 | 0.016813041 | 0.002826498 | 5.95x |
| 8192 | 0.124363243 | 0.021494726 | 5.79x |

Reproduce final measurements with:

```sh
FHE_TEST_RING_DIMENSION=1024 cargo test -r -p mxx-fhe --lib test_cpu_key_switch_evaluation_timing -- --ignored --nocapture
```

`FHE_BENCH_REPEATS` sets the sample count. Raw logs, samples, and the
programmatically reconciled CSV are under `test_data/bgv-rns-fusion/` (ignored).

## Validation

- Workspace release unit-test compilation, CPU and GPU: warning-free.
- CPU matrix module: 25 tests passed; direct-entry format/basis test passed.
- CPU FHE: 15 tests passed; manual timing fixture ignored in the normal suite.
- CPU fused-versus-composed oracle: 300 consecutive executions, zero failures.
- RNS IR/exporter tests: four matching tests passed; runtime basis-order rejection passed.
- RTX 4080 SUPER: fused GPU oracle and three BGV tests each passed three executions.
- `lake build RuntimeMatrixOps`: passed. This checks the operational definitions;
  it is not a cryptographic security proof or a new BGV noise theorem.
- Integration tests were not run.
