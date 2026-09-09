# RNS-BGG refresh parameter search, 2026-09-09

## Status and scope

The 40-bit CRT candidate below passes the current average-case correctness
model, including the joint PRF/refresh event budget and a subsequent LUT on
the refreshed state. It also exceeds the 100-bit target under the previous
BGG search policy of conventional component estimates: the new LWE rough
estimate is **146.18687920368833 classical bits**, and the existing pinned
sparse-LWR estimate is **103.18615790798486 classical bits**. Their minimum
is 103.18615790798486 bits (integer floor 103).

This report uses the same conventional component-security interpretation as
the earlier BGG searches. It does not prove joint security of correlated
PRF outputs, masks, and cross-modulus views. Average-case correctness also
retains the existing conditional residual-independence and tail heuristics;
it is not a worst-case correctness theorem.

## Component security evidence

The new LWE run uses the exact 1,080-bit modulus from the numerical
candidate, not a power-of-two surrogate, with unlimited samples. Sage 10.7
and official estimator commit `53da5982597709ba0fdf94ea37a84d822310fd84`
returned all three attacks selected by `LWE.estimate.rough`:

| Attack | Classical log2 cost |
| --- | --- |
| Primal uSVP | 185.712 |
| Dual hybrid | 185.71200000000002 |
| Arora-GB | 146.18687920368833 |

The minimum is Arora-GB, with solving degree 5. Both lattice attacks select
BKZ block size 636. The dual-hybrid optimization returns no guessed/FFT
coordinates (`zeta=0`, `t=0`). This run completed; no missing attack result
was treated as a security bound.

The LWR evidence is the existing reviewed `(8192,18,4096,32)` tuple under
MATZOV/GSA with unlimited samples. Its stored-value consistency test passed;
this task did not rerun that external estimate. These two components use
the same different estimator policies as the earlier BGG parameter searches.
The full eight-attack LWE/MATZOV run has not completed and is **not** used
as evidence. The result is a conventional rough-policy parameter estimate,
not a claim that the full attack suite or whole-construction security has
been certified.

## Selected numerical candidate

| Parameter | Value |
| --- | --- |
| Encoding ring dimension | 65,536 |
| Encoding secret | Uniform ternary, `{-1,0,1}` |
| Fresh encryption/decoder error | `CBD(eta=2)`, variance 1 |
| Initial CRT depth and prime width | 27 primes, each 40 bits |
| Initial encoding modulus | 1,080 bits; exact primes in companion JSON |
| Gadget base | `2^20` |
| LWR dimension / secret weight | 8,192 / 18 |
| LWR input / output modulus | 4,096 / 32 |
| PBC buckets / private products | 21 / 21 |
| Exponent layout | `stride_log2=5`, `width_log2=11` |
| PRF-end modulus | 6 primes, approximately 240 bits |
| Refreshed modulus | 5 primes, approximately 200 bits |
| Next LUT output modulus | 4 primes, approximately 160 bits |
| State widths: initial / PRF end / refreshed / next LUT | 104 / 20 / 16 / 12 polynomials |
| Correctness failure exponent | 100, average-case model |
| Joint mask statistical-security target | 100 bits |

The chain uses nested prefixes with depths `27,26,...,1`. The PRF consumes
21 transitions, `27 -> 6`; refresh consumes `6 -> 5`. The output is a normal
16-polynomial encoding, retaining the chain `5 -> 4 -> 3 -> 2 -> 1`.
It is not replaced by a terminal two-polynomial row. We explicitly checked
one subsequent LUT; we did not certify an arbitrary future application.
The squared-input-modulus/two-column optimization remains deferred.

The numerical search uses the production private-product recurrence and
replays the same PRF accumulation Program through `rns_bgg::trace_prf`.
Exact rational variances must agree. It counts all selector ciphertext
errors, including the unselected members of each one-hot group. The saved
PBC layout has 1,131--1,240 active selectors per bucket; its seed and counts
are in the JSON.

## Refresh workload and correctness checks

Each refreshed polynomial has 65,536 coordinates. Across 16 polynomials,
there are 1,048,576 coordinates and five CRT mask channels. The digit plan
uses base 32 and weights `32^j`:

- 31 mask digits per channel, with support `32^31 = 2^155`.
- One fresh-error digit per coordinate, shared across channels, with the
  shifted interval `[-15,16]`, mean `1/2`, and variance `341/4`.
- Distinct PRF outputs: `(5 * 31 + 1) * 1,048,576 = 163,577,856`.
- Terminal digit emissions, counting the shared fresh digit in each channel:
  `(5 * 31 + 5) * 1,048,576 = 167,772,160`.

Refresh data is assigned the error envelope of the completed PRF, rather
than assumed to be a freshly encrypted input. The final exposed error bound
is 231,900,862 in every channel. The smallest CRT spacing is
1,461,469,928,716,547,646,209,826,899,121,853,740,468,082,573,313.
The planner checks both mask support/statistical distance and room within
this spacing, including fresh-error support.

The joint budget counts 5,242,880 exposed coordinates and all coordinates
of every PRF state for all distinct outputs: 14,622,405,143,035,904 events.
The resulting squared tail multiplier is `5,371,897 / 25,000 = 214.87588`.
The simulation audits every PRF gate against its modulus and message scale
using this joint multiplier. The next LUT is additionally checked with the
existing worst-case bound: error 67,165,186, versus an output half-scale of
approximately `2^39`. This continuation assumes a norm-one plaintext and
2,048 LUT branches.

Correctness does not make this a practical full-refresh configuration.
The current planner estimates **57,684,173,139,801,538,560 bytes** of terminal
artifacts alone (about 57.7 decimal exabytes). This is persistent storage,
not peak GPU memory or measured runtime, and excludes earlier PRF stages.
No full-refresh round trip or GPU benchmark was run for this search.

## Nearby candidates

All rows keep `N=65536`, the LWR tuple, and the two 100-bit targets fixed.
These are numerical results, not separate security certifications.

| CRT bits / base bits | Initial modulus bits | Refreshed columns | Distinct PRF outputs | Numerical acceptance |
| --- | --- | --- | --- | --- |
| 50 / 25 | 1,350 | 16 | 163,577,856 | Yes |
| 40 / 20 | 1,080 | 16 | 163,577,856 | Yes |
| 32 / 16 | 896 | 20 | 245,104,640 | Yes |
| 28 / 14 | 810 | 24 | 342,884,352 | Yes |
| 26 / 13 | — | — | — | No, subsequent worst-case LUT check fails |
| 25 / 12 | — | — | — | No, subsequent worst-case LUT check fails |

The 40-bit candidate lowers the initial modulus relative to the 50-bit
candidate without increasing widths or output count. Smaller primes need
more remaining CRT limbs to accommodate the masks. Rejection here applies
to the tested pools and continuation policy, not to all possible RNS-BGG
parameters. The search is not a global runtime optimization.

## Reproduction and evidence

The companion JSON contains the selected exact chain, PBC seed, digit plan,
raw serialized planner output, source hashes, and security evidence. The
CSV records the numerical comparison. Big integers in the serialized Rust
planner use little-endian 32-bit limbs.

Build and run the host-only numerical entry in the existing integration
file (the GPU feature is required by the file, but this entry creates no
GPU graph or ciphertext):

```sh
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh --no-run
MXX_EXPONENT_LUT_REFRESH_CRT_BITS=40 \
MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH=27 \
MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=100 \
MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT=test_data/rns-bgg-refresh-search.json \
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh \
  test_rns_bgg_average_refresh_noise_simulation -- --exact --nocapture
```

The PBC seed is random on each run, so selector counts and tail bounds can
vary slightly. Inspect `correctness_accepted`; successful test execution
alone does not imply an accepted parameter set. The archived JSON records
the exact layout used for the values above.

Implementation references:

- `crates/exponent-lut/tests/noise_refresh.rs`: `test_rns_bgg_average_refresh_noise_simulation`.
- `crates/exponent-lut/src/rns_bgg/noise.rs`: `trace_prf`, `plan_refresh`, and gate error propagation.
- `crates/exponent-lut/src/noise_refresh/prf.rs`: the actual accumulation Program.
- `crates/exponent-lut/tests/noise_refresh.rs`: `reviewed_primary_tuples`, pinned LWR evidence.

### LWE estimator reproduction

Use Sage and official `lattice-estimator` commit
`53da5982597709ba0fdf94ea37a84d822310fd84` on `PYTHONPATH`.
The LWE distribution is passed explicitly; the old Gaussian-error envelope
is not reused for `CBD(2)`. The conventional comparison uses the official
`LWE.estimate.rough` policy (ADPS16 Core-SVP/GSA and its documented attack
selection). This is distinct from the full MATZOV attack suite used for
the pinned sparse-LWR evidence.

```python
import json
from sage.all import Integer, oo, log
from estimator import LWE, ND, gb

record = json.load(open("docs/plans/exponent-lut-rns-bgg-refresh-security100.json"))
params = LWE.Parameters(
    n=65536,
    q=Integer(record["selected_candidate"]["initial_modulus"]),
    Xs=ND.Uniform(-1, 1), Xe=ND.CenteredBinomial(2), m=oo,
)
original = gb.gb_cost

def adaptive(n, D, omega=2, prec=None):
    if prec is not None:
        return original(n, D, omega, prec)
    limit = 2 * n
    precision = min(8, limit)
    while True:
        result = original(n, D, omega, precision)
        if result["dreg"] != oo or precision == limit:
            return result
        precision = min(2 * precision, limit)

gb.gb_cost = adaptive
results = LWE.estimate.rough(params, jobs=3, catch_exceptions=False)
assert {"usvp", "dual_hybrid", "arora-gb"} <= results.keys()
print({name: float(log(cost["rop"], 2)) for name, cost in results.items()})
```

The adaptive wrapper changes only exact series expansion precision.
`gb_cost` stops at the first negative Hilbert-series coefficient. The
coefficients below a cutoff are unchanged by omitted higher terms. If no
negative coefficient is found, the wrapper expands further, up to the
original `2*n` limit. The bounded-error branch for the selected tuple finds
its first negative coefficient at degree 5. Small checks at dimensions
8, 16, 32, and 64 gave exactly the original cost dictionaries. This is not
an assertion that the original degree-131072 computation completed and was
compared, nor that the full estimator's other branches finish at degree 5.
