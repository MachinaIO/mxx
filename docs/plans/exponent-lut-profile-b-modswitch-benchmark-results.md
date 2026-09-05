# Profile B modulus-switched refresh: H200 benchmark estimate

## Technical summary

The September 5, 2026 estimate completed successfully on two H200 GPUs. Against
the previous successful H200 release-fence repeat, the selected five-tower
refresh reduces modeled preprocessing work by 86.14% and online work by 85.74%.
Ideal unlimited-resource DAG latency decreases by 23.10% and 7.54%, respectively.
These are GPU-calibrated estimates, not measured complete-program execution times.

## Fixed environment and comparison basis

Both runs use two H200 GPUs, CUDA 12.8.93, Rust/Cargo 1.98.1,
80% VRAM policy, one warmup and two measurement iterations. The new run uses
OpenFHE 1.4.2; the historical OpenFHE version has not been independently verified.
Full original
dimensions are retained: `n=65536`, 27 CRT towers of 50 bits, decomposition base
bits 25, `Q=4096`, native PRF base `p=32`, `v=8192`, `h=18`, and 108 source
columns. No manual fixed wave caps are applied.

The new replay uses the accepted average-case noise report with 100-bit failure
and mask-smudging parameters, five retained towers per slot, six distinct
250-bit P groups, uniform `d_m=38`, and shared `d_e=1`. Exact q and ordered P
values are recorded in the accepted replay JSON and
`docs/plans/exponent-lut-profile-b-mask-search-results.md`.

The historical measured-estimator baseline used `d_m=250`, not 249. The latter
is the separate current-simulator P=q comparator from the mask-search report;
it is not substituted for the historical benchmark here.

| Accounting quantity | Historical H200 repeat | Switched H200 run |
|---|---:|---:|
| Mask digits per slot | 250 | 38 |
| Native PRF outputs | 47,782,821,888 | 7,268,990,976 |
| Native PRF information bits (5 per output) | 238,914,109,440 | 36,344,954,880 |
| Decoder preimage artifact bytes | 149,824,734,561 | 32,105,301,345 |

The output-count ratio is 0.152126. This accounts for much of the reduction in
total work; it does not predict the same improvement in dependency depth.

## Total work falls more than ideal DAG latency

Work sums calibrated GPU work over modeled invocations. Cumulative waves sum
fleet-wave latencies over those invocations. Ideal DAG latency is the estimator's
critical path under unlimited scheduling resources, not the time required to
execute the entire DAG on these two GPUs. All values below are seconds.

| Stage / metric | Historical H200 repeat | Switched H200 run | New / old |
|---|---:|---:|---:|
| Preprocess work | 1.683494065996354e16 | 2334001729508941.0 | 0.138640 |
| Preprocess cumulative waves | 8431229306404981.0 | 1168642814281283.5 | 0.138609 |
| Preprocess ideal DAG | 52.530028878000024 | 40.396357977000086 | 0.769015 |
| Online work | 1.8862675868487988e16 | 2689404757964997.5 | 0.142578 |
| Online cumulative waves | 9453895049055412.0 | 1353100409743131.8 | 0.143126 |
| Online ideal DAG | 27.605426201 | 25.524083569999966 | 0.924604 |

Preprocessing includes select setup, the public-key program, and decoder
preimages. Online is logged as `oneline_eval`. Neither whole-stage total nor
whole-DAG latency is divided by PRF label count to claim per-bit cost.

## PRF-only per-bit accounting improves modestly

These denominators use native PRF information bits, and the numerators include
only the corresponding PRF accumulation and value-readout roles. They exclude
unrelated refresh/decoder work. Cumulative-wave seconds per bit are amortized
accounting, not isolated-bit latency or an achievable full-program throughput.

| PRF-only metric, seconds per information bit | Historical H200 repeat | Switched H200 run |
|---|---:|---:|
| Public-key work | 68791.7868986067 | 62517.35312095739 |
| Public-key cumulative waves | 34450.20453857375 | 31297.513487140503 |
| Encoding work | 48562.69326102998 | 44987.89728017567 |
| Encoding cumulative waves | 24329.260119644354 | 22567.127256955057 |

Per-bit work is approximately 9.12% lower for public-key PRF and 7.36% lower
for encoding PRF. Calibration variation is not separated from implementation
effects by this single new estimate; these modest differences should not be
attributed solely to modulus switching.

## Low-P decoder products fit in one calibrated wave

The generic high-q small-RHS shapes retain their previous fleet capacities:
`(2 x 108) * (108 x 108)` uses capacity 20 and six waves; the analogous
one-row product uses capacity 42 and three waves. Their measured fleet-wave
latencies change from 0.529145344 to 0.4803505275 seconds and from
0.7505945005 to 0.695152106 seconds, respectively.

The old decoder shape `(1 x 112) * (112 x 108)` at q uses fleet capacity 40,
three waves, 0.8637670875 seconds per fleet wave, and 2.5913012625 cumulative
wave seconds per modeled product. The new six P-specific decoder shapes
`(1 x 24) * (24 x 108)` each use capacity 1022 and one wave. Their measured
latencies are 0.0895189995, 0.089303719, 0.1578163, 0.089508793,
0.0890656835, and 0.0894423415 seconds. The slower third sample is retained,
not discarded. These are distinct ring-group measurements, not six repetitions
of one operation. Shape and ring changes preclude pairing nodes by numerical ID.

## Evidence, reproducibility, and limits

The new log, relative to the external main checkout's repository root rather than this
worktree, is
`logs/runpod/2026-09-05/h200-modswitch-B-e7bdd72b1/65s4t43qg05bbo-2gpu-e7bdd72b1-20260905-B-modswitch-estimate.log`.
The historical log is
`logs/runpod/2026-09-05/h200-B-a8d55ce0a/ckvq51jj993o7x-2gpu-a8d55ce0a-20260905-B-h200-release-fence-repeat.log`
in the historical checkout's retained logs. Both record a passed estimate test
and `BENCH_EXIT=0`. The new estimate test took 216.12 seconds; the wrapper took
336 seconds including compilation. Neither is modeled DAG execution time.
The periodic GPU monitor observed a peak of 114510 MiB on each GPU; this is a
sampled peak, not an exact allocation high-water mark. This report compares one
new run (one warmup, two measurement iterations), not repeated new benchmark runs.
The orchestration owner reports pod
`65s4t43qg05bbo` stopped with status `EXITED` after evidence retrieval.

The source was an uncommitted archive of branch `codex/exponent-lut-modswitch`,
with baseline metadata `e7bdd72b1c4c594f7aa282cc655546cb6891ec78`; that commit
does not identify the dirty source. The actual uploaded source archive SHA-256
is `3978181559778acad841a4b1216bee89d8b7787c3e08eae4035441dc97fe5088`.
Accepted replay files are `B-modswitch.phase2.json` and `B-modswitch.report.json`,
with SHA-256 values `4cd5091b3f4a44534b9d9ebd76597344b2254eb048daf847d9fb2293d125b807`
and `81fe98518f8082dc93ecfc3c12a58fb881d54882048693fe8926228f0906ec28`.

The invoked target was `test_gpu_exponent_lut_benchmark_estimate` in
`crates/exponent-lut/tests/benchmark_estimate_gpu_tests.rs`, release mode with
GPU enabled and `--ignored --exact --nocapture`. It replays the accepted report
and measures actual GPU primitives, including switch transforms. Parameter
simulation and lattice estimation were not rerun. No complete-program runtime,
round-trip, or joint-security proof is claimed. Some memory-accounting fields
contain `UINT64_MAX` saturation sentinels and must not be read as measured VRAM.

The useful conclusion is lower modeled total work with smaller decoder waves,
not a demonstrated sevenfold end-to-end runtime speedup. Repeated independent
estimates would be needed to quantify calibration variability; full execution
would require a separate authorized experiment. The previously documented
shared-secret/joint-view security limitation remains unchanged.
