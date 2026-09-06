# Universal approximate gadget: H200x2 estimate

The September 6, 2026 run completed successfully. Against the previous Profile B
modulus-switched H200 estimate, preprocessing work increased **57.71%** and
online work increased **31.57%**. This is not an isolated approximation comparison:
the new full modulus has 33 CRT towers rather than 27, while ordinary gadget
width changes from 108 to 104 columns.

## Parameters and rerun noise validation

The selected configuration is N=65536, 33 primes of 50 bits, base bits 25,
seven omitted gadget towers (26 retained), Q=4096, p=32, v=8192, h=18,
mask digits 38, fresh digits 1, and 100-bit failure/mask targets.
Ordinary gadgets and decompositions use 104 columns throughout PRF and refresh.
Only trapdoor preimages retain exact sampler rows: each five-tower decoder
uses B of shape 2x24 and K of shape 24x104. There is no 104-to-132 bridge.

Noise simulation was rerun **after** implementing the universal approximate
routing equations. Public even residuals R congruent to each route scalar C
modulo the omitted-prime product define the actual target C-R. This prevents
half-integer native payloads from causing modular half-wrap. Exact retained
digit gains and deterministic per-slot payload bias are included.

The revised AverageCase model passed with maximum F = **850377501249489271**.
The same CPU test validated both production graphs and their setup identity.
The previous maximum-seven-dropped result applied to the former bridge model;
this rerun establishes acceptance of seven, **not its maximality** in the
universal model. Conventional full-q RLWE rough security is floor 102 bits and
the pinned sparse-LWR estimate is 103.186 bits; correlated joint-view security
remains unresolved.

## GPU-calibrated stage accounting

All values are seconds. Work sums GPU work; cumulative waves sum calibrated
fleet-wave latencies over modeled invocations. Ideal DAG is the critical path
with unlimited scheduling resources, **not achievable complete-program runtime
on these two GPUs**.

| Stage / metric | Previous 27-tower exact gadget | New 33-tower approximate gadget | New / previous |
|---|---:|---:|---:|
| Preprocess work | 2334001729508941.0 | 3681018493415230.0 | 1.57713 |
| Preprocess cumulative waves | 1168642814281283.5 | 1846302424673399.8 | 1.57987 |
| Preprocess ideal DAG | 40.396357977000086 | 46.75858355049968 | 1.15750 |
| Online work | 2689404757964997.5 | 3538323666630709.5 | 1.31565 |
| Online cumulative waves | 1353100409743131.8 | 1776261968616454.5 | 1.31273 |
| Online ideal DAG | 25.524083569999966 | 27.87102673950018 | 1.09195 |

PRF label count is 8553758720 = 65536 x 104 x (33 x 38 + 1), or
42768793600 information bits at five bits per native output: 17.675% more
labels than the prior 7268990976. The following numerators contain only
PRF accumulation and value readout, not unrelated refresh work.

| PRF-only seconds per information bit | Previous | New |
|---|---:|---:|
| Public-key work | 62517.35312095739 | 83788.0744599583 |
| Public-key cumulative waves | 31297.513487140503 | 42025.21135528996 |
| Encoding work | 44987.89728017567 | 50256.66268121681 |
| Encoding cumulative waves | 22567.127256955057 | 25229.514310935043 |

These are amortized accounting metrics, not isolated-bit latency.

## Calibrated shapes and validation scope

At full q, (1x104)(104x104) has fleet capacity 36, three waves,
0.7787048235 seconds per wave, 4.656615747 work seconds, and
2.3361144705 cumulative-wave seconds. The prior (1x108)(108x108)
used capacity 42 and three waves at 0.695152106 seconds per wave.

The two-row (2x104)(104x104) product uses capacity 16 and seven waves
at 0.5518334135 seconds per wave, versus capacity 20 and six waves for
the prior (2x108)(108x108). The seven new low-P decoder groups
(1x24)(24x104) each fit one wave, with wave latency between
0.0884419265 and 0.08958271400000001 seconds. Shapes and rings differ;
nodes are not paired merely by numeric ID.

Both runs use two H200 GPUs, 80% VRAM policy, one warmup and two iterations.
All six explicit-short-count GPU primitive checks passed (three on each GPU).
The estimate test passed in 238.54 seconds; the wrapper took 401 seconds,
including compilation/tests, and recorded BENCH_EXIT=0. The orchestration
owner confirmed the pod stopped with status EXITED. UINT64_MAX memory fields
are saturation sentinels, not measured VRAM or usable capacity.

Local validation also passed: warning-free release workspace library
compilation with and without GPU, GPU benchmark compilation, 25 simulation
unit tests, one route-parity test, and one CPU mixed-layout test.
No complete refresh execution, round trip, or joint-security proof is claimed.
One calibration run does not isolate implementation effects from measurement
variation.

## Evidence

- Noise and source-generated benchmark input:
  `test_data/universal_approximate_selected/noise.json` and
  `test_data/universal_approximate_selected/benchmark-input.json`.
- Machine-parsed comparison, raw values and log line references:
  `test_data/universal_approximate_selected/h200-comparison.json`.
- CSV, checked against parsed log values:
  `docs/plans/exponent-lut-universal-approximate-h200-results.csv`.
- Prior report:
  `docs/plans/exponent-lut-profile-b-modswitch-benchmark-results.md`.
- New retained log, relative to the external main checkout:
  `logs/runpod/2026-09-06/h200-universal-approx-a125812f0/65s4t43qg05bbo-h200x2-a125812f0-20260906-universal-approx-estimate.log`.

Source delivery was an uncommitted archive of `codex/exponent-lut-encode`,
base metadata `a125812f0c4af7405f5639fa3f57cdd0599c26c7`.
The archive, not that base commit, identifies the measured source:
SHA-256 `d5507d79775dd3c0d70fe5b53f621210982f2e3cef0ce9ad1ff6e468465cb538`.
No commit or push was performed.

## Follow-up: minimum-depth exact gadget, depth 26/drop 0

The subsequent September 6 run benchmarked the freshly simulated exact
baseline at depth 26, with 104 ordinary gadget columns and no approximation
residual. Its CPU noise/graph export passed with maximum F=
1306070000453953074; conventional BGG security floor 143 and pinned sparse-LWR
floor 103 remain above the 100-bit target. The exact metadata in
`test_data/exact_minimum_depth_selected/benchmark-input.json` is authoritative;
the shared export helper's older "universal approximate" banner is cosmetic.

Work and cumulative waves are accounting totals, not elapsed complete-program
runtime. The ideal DAG column retains the unlimited-resource meaning above.

| Stage / metric, seconds | Exact 26/drop 0 | Exact26 / approx33 | Exact26 / prior exact27 |
|---|---:|---:|---:|
| Preprocess work | 2128709018008008.8 | 0.578293 | 0.912043 |
| Preprocess cumulative waves | 1068314436844513.3 | 0.578624 | 0.914150 |
| Preprocess ideal DAG | 41.06335061899996 | 0.878199 | 1.01651 |
| Online work | 2794472899242602.5 | 0.789773 | 1.03907 |
| Online cumulative waves | 1407045751844575.3 | 0.792139 | 1.03987 |
| Online ideal DAG | 26.039940829000077 | 0.934301 | 1.02021 |

The exact26 label count is 6740770816 = 65536 x 104 x (26 x 38 + 1),
or 33703854080 information bits. Only PRF accumulation and value-readout
work enters these amortized per-output and per-bit numerators:

| PRF-only accounting, seconds | Public-key PRF | Encoding PRF |
|---|---:|---:|
| Work per native output | 308211.4814566578 | 258463.5196995486 |
| Cumulative waves per native output | 154677.6689888892 | 129458.39411460972 |
| Work per information bit | 61642.29629133156 | 51692.70393990972 |
| Cumulative waves per information bit | 30935.533797777844 | 25891.678822921946 |

The two-row full-q product uses fleet capacity 22 and five waves at
0.5709481935 seconds per wave; the one-row product uses capacity 46 and
three waves at 0.8022274115 seconds per wave. Its six low-P decoder groups
each fit one wave. This comparison changes CRT depth and, versus exact27,
gadget width. The new host exposes 32 CPU threads versus 48 on the preceding
host; CUDA 12.8.93, H200x2, VRAM 80%, one warmup and two iterations are
unchanged. Driver version is 580.178.04. These single estimates do not isolate
an approximation effect or control host/calibration variability.

All six GPU primitive checks passed. The benchmark passed in 166.67 seconds,
the wrapper took 347 seconds, and BENCH_EXIT=0 was recorded. The orchestration
owner confirmed pod `8irrfv3qhe2noh` stopped with status EXITED. No complete
refresh execution or round-trip claim is added.

The raw log, relative to the external main checkout, is
`logs/runpod/2026-09-06/h200-exact26-a125812f0/8irrfv3qhe2noh-h200x2-a125812f0-20260906-exact26-estimate.log`.
Parsed comparisons are
`test_data/exact_minimum_depth_selected/h200-vs-approx33.json` and
`test_data/exact_minimum_depth_selected/h200-vs-exact27.json`.
The existing companion CSV is appended without changing its columns;
new stage prefixes `exact26_vs_approx33:` and `exact26_vs_exact27:` identify
the comparator. Every appended value and ratio was checked against parsed
completed logs. Original unprefixed CSV rows remain approx33 versus exact27.

Measured source archive SHA-256:
`a472f3f03f969c96f66667185a9c71538eb9163f7e6f198074c832e5323a57a3`.
Exact26 benchmark input SHA-256:
`c6f68bfd10fe0c7ad264e00e5cd9e3aaaf20f9392682d043f3ba4eb081cd81e4`.
The source remains an uncommitted archive of the same branch/base metadata;
no source change, commit or push occurred during this measured run.
