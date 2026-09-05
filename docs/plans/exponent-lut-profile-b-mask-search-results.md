# Profile B reduced-mask parameter search (2026-09-05)

## Result and scope

Choose five retained CRT towers per slot and `d_m=38`, with `d_e=1`.
This is the minimum uniform mask-digit count in the completed contiguous-group
search over **all 27 retained widths**, holding the original comparison
profile fixed. Six retained towers tie at 38 digits; the declared tie-break
chooses the smaller retained width and narrower per-slot decoder. This is not
an exhaustive search of arbitrary subsets or ragged digit vectors.

The selected point passes the existing **AverageCase** noise authority with
failure exponent 100, tail correction 0, and mask statistical-security target
100. Existing marginal lattice estimates also exceed 100 bits, as detailed
below. The cross-modulus joint-security argument remains unresolved: these
are model-qualified candidate parameters, not a proven 100-bit deployment.

Fixed profile: `n=65536`, 27 50-bit CRT primes, `log2(beta)=25`, `Q=4096`,
`p_PRF=32`, sparse-LWR `v=8192,h=18`, encoding sigma `4`, decoder sigma
`4.578`, default official preimage cutoff, one refresh/execution. Original
public/encoding width is `m=108`. The program/layout and noise formulas were
unchanged throughout the sweep.

## Same-profile comparison

The comparator is `P=q` under the **same current simulator**, not a recovered
historical benchmark mask count. No historical mask-count artifact was found,
and no GPU benchmark or integration test was run for this search.

| Quantity | Selected: 5 towers | Tied: 6 towers | Same-model `P=q`: 27 towers |
| --- | ---: | ---: | ---: |
| Bits of each `P` | 250 | 300 | 1350 |
| Uniform `d_m` | 38 | 38 | 249 |
| `sum_t d_m,t` | 1026 | 1026 | 6723 |
| Fresh `d_e` | 1 | 1 | 1 |
| Mask `M` | `2^190` | `2^190` | `2^1245` |
| Centered bound `B_m` | `2^189` | `2^189` | `2^1244` |
| Distinct anchors | 6 | 5 | 1 |
| Columns of each `B_P` / rows of each `K_t` | 24 | 28 | 112 |
| Columns of every `K_t` | 108 | 108 | 108 |
| PRF labels | 7,268,990,976 | 7,268,990,976 | 47,591,718,912 |
| Maximum simulated `F_avg` | 850377501249489267 | 1975968368272671393 | approximately `1.0568047e335` |

The exact count is `n*m*(sum_t d_m,t+d_e)`. Thus the selected point has
**84.7264% fewer PRF labels** than the same-model `P=q` comparator
(a `6.5472`-fold count reduction). This is a count comparison, not a measured
runtime speedup. The six-tower tie uses fewer anchors, but wider per-slot
decoders; no performance superiority is asserted for either.

All widths 1–4 were rejected by exact preparation/noise conditions. The first
accepted uniform digit count at each width 5–27 was:

```text
retained: 5  6  7  8  9 10 11 12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27
d_m:     38 38 48 58 68 78 89 99 109 119 129 139 149 159 169 179 189 199 209 219 229 239 249
```

## Exact selected moduli and checks

The original modulus is:

```text
q = 24575012695138087385929126897750909656272258522909779006351874311661121711669962466811957617902595081528153873688517552616364202863895535255835360511239178629152398019130383467745651051744701995127286122317461995823907819630788470303452222277876040173987573103242519525568461033912964253821904328182996249953838254840703474844125956014277342873416614777086975400735779396431017302877712465764655546425802753
```

Zero-based slot groups and their exact switching moduli:

| Slots | Retained source tower indices | `P_t` |
| --- | --- | --- |
| 0–4 | 0–4 | 1809251283755169804643792595030325745566728194278478234920304281948304834561 |
| 5–9 | 5–9 | 1809251207930325016022886889611584165212105705678103291926203402066066800641 |
| 10–14 | 10–14 | 1809251120731756287638325423094560278549504412161526380292198786363058421761 |
| 15–19 | 15–19 | 1809251018157598974248277523328659399802009492139169810353294299171281174529 |
| 20–24 | 20–24 | 1809250902945974011043096294392820043706360432234152287429651025113362333697 |
| 25–26 | 22–26 | 1809250879987899561990856885660925020771157995355245904313539848666809630721 |

Every slot satisfies exact `q_t | P_t | q`, with `J_t=q/P_t` and
`Delta_t=P_t/q_t` (200-bit spacing). The complete decimal `q_t,J_t,Delta_t`
and per-slot error/margin values are in the persisted search JSON.
For the selected point:

- `M=1569275433846670190958947355801916604025588861116008628224`.
- Minimum strict rounding margin
  `Delta_t-2(B_m+F_avg,t)`:
  `1605368404408516041782639623211186277883973868608376545242395 > 0`.
- Minimum smudging margin
  `M-2^101*(27*65536*108)*F_avg,t`:
  `1157264469336541567343474824793365005706215453443007447040 > 0`.
- Fresh support bound is `31`; its smallest prime-factor margin is
  `1125899841175522 > 0`.
- Joint tail event count is `7268990976`, with exact `z_squared=11610229/62500`.
- Official and selected decoder preimage cutoffs both equal `12437291109817`.

Average-case operational variances retain the accepted TFHE-style heuristic
and conservative correlated `Add`; deterministic paired rounding is additive
and the low-P decoder error is not divided by `J`. Payload support is not
counted as operational variance. Passing these checks is not worst-case
correctness acceptance.

## Security interpretation

The unchanged original high-q ternary/Gaussian marginal was freshly estimated
at floor **136 bits** (rough uSVP/dual-hybrid approximately `2^136.4`,
`n=65536`, sigma `4`, unlimited scalar samples). The reviewed sparse-LWR tuple
`(4096,32,8192,18)` has the pinned conservative floor **103 bits**
(`103.18615790798486`). Each selected 250-bit anchor modulus has the separately
checked rough scalar-LWE marginal floor **1202 bits**, using one unknown
ternary ring polynomial after subtracting the fixed augmented public row,
sigma `4`, and `24*n=1572864` scalar equations for one execution.

These authorities are distinct from noise/smudging acceptance and from a
joint-view proof. Marginal estimates do not establish security of correlated
ring equations, low-P anchors sharing a secret, public switched targets and
preimages, or the complementary `J` projections. The six-tower tie is reported
here as a noise-accepted alternative; its marginal estimate is not reused
from the five-tower selection. Estimator provenance and finite-Sage reproduction
for the selected anchor moduli are recorded in
`docs/plans/exponent-lut-modulus-switched-refresh.md`, §8; the CLI overflow
sentinel is not used as a security value.

## Reproduction and persisted evidence

Run the explicit component search (not a backend/integration execution):

```sh
cargo test --release -p mxx-exponent-lut --test parameter_search_unit \
  profile_b_minimum_mask_retained_width_search -- --ignored --exact --nocapture
```

The completed unit passed in 305.16 seconds. It checks every retained width,
tests mask counts in ascending order within exact safe limits, materializes
the first accepted candidate's full production report, and selects
lexicographically by `(sum_mask_digits, retained_width)` without a hardcoded
result. Compilation and nightly formatting passed; no production code or
noise formula changed for this search.

This run's repository-relative evidence directory is
`test_data/profile_b_minimum_mask_search/1788588283213876339/`:

- `search.json`: full grid, exact decimal summaries and selected score.
- `widths.json`: all 27 accepted/rejected width records.
- `retained-5-full-report.json`: complete selected production noise report.
- `retained-6-full-report.json`: tied alternative's report.
- `retained-27-full-report.json`: exact same-model unswitched comparison.

Reports for every other accepted width are retained in the same directory.
Fresh runs create a new timestamped directory rather than overwrite evidence.
