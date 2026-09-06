# Universal approximate gadget: minimum exact depth and bounded drop search

## Result

The September 6, 2026 CPU-only rerun found **minimum exact depth d0=26**.
With full CRT depth at most d0+3=29, the **maximum accepted omitted count is
three**, attained at depth 29 with 26 retained towers and 104 gadget columns.

| Full depth | Maximum omitted towers | Retained towers | Gadget columns | Maximum AverageCase F |
|---:|---:|---:|---:|---:|
| 26 | 0 | 26 | 104 | 1306070000453953074 |
| 27 | 1 | 26 | 104 | 850377501249489269 |
| 28 | 2 | 26 | 104 | 850377501249489269 |
| 29 | 3 | 26 | 104 | 850377501249489269 |

At depth 29, this omits 10.3448% of ordinary gadget columns: 116 becomes 104.
It does not reduce column count below the minimum-depth exact baseline.

## Fixed scope and complete coverage

The sweep fixes N=65536, 50-bit CRT primes, base bits 25, sparse-LWR
(v,h,Q,p)=(8192,18,4096,32), mask/fresh digits 38/1, five-tower exact decoder
moduli, and 100-bit AverageCase failure and mask-smudging targets. The public
PBC layout and source-generated ordered primes are the same deterministic
profile used by the preceding simulation and benchmark work.

This uses the **current universal approximate production model**: ordinary
gadgets/decompositions are shortened throughout PRF and refresh, exact
rectangular trapdoor preimages retain their sampler row count, and public
even-corrected routing targets include both actual retained-digit gains and
deterministic half-payload residual bias. There is no short-to-full bridge and
no reuse of the obsolete bridge-model frontier or stale mask-gain shortcut.

- Exact baseline: depths 1–4 cannot accommodate the fixed five-tower switch;
  every depth 5–25 was simulated and rejected, while depth 26 passed.
  At depth 25, maximum F was 690390709822359186247793024787212.
- Window: all 110 pairs with depth 26–29 and omitted count 0 through depth−1
  were simulated, without monotonicity assumptions.
- Lower-depth check: all 231 pairs at depths 5–25 with omitted count at least
  four were additionally simulated and rejected. Thus failure of an exact
  configuration was not assumed to imply failure of its approximations.
  A supplemental run also evaluated all 63 lower-depth pairs with one through
  three omissions; all were rejected. Depth 29 is therefore the only depth
  in the explored domain attaining three accepted omissions.

All 425 distinct feasible pairs at depths 5–29 are covered, with **zero
preparation errors** (426 evaluations include the repeated exact depth-26
baseline). Four structural exclusions are recorded separately. The primary
CPU test passed in 54.22 seconds and the supplemental test in 14.87 seconds.
Four independent snapshots were processed concurrently to
bound memory. Maximum claims apply only to this fixed-parameter domain, not
to different ring dimensions, bases, mask digits, or decoder layouts.

## Acceptance margins and security

For the selected depth-29/drop-3 case, all hard-authority, correctness and
combined acceptance flags are true. Its minimum strict rounding margin is
1605368392622968876904248787003490853407078941433388488092951, and minimum
mask-smudging margin is
1143135149565961023693244030820353540475038023006956814336.
All per-slot thresholds, variances, moduli and bias terms are saved in the
raw report. Large rejected unreduced tail bounds are model outputs, not
observed modular decryption failures.

The conventional full-q RLWE estimate was rerun directly on the selected
depth-29 modulus, with N=65536, ternary secret, Gaussian sigma=4, and unlimited
samples. The rough usvp/dual-hybrid estimates are approximately 123.2 bits
(CLI floor 123), using estimator commit
`e35f45b7976a90a79c3c6625a45bbc344c1abc67`. The unchanged pinned sparse-LWR
estimate is 103.186 bits, so the conventional component estimates remain
above 100 bits. This does not prove security of correlated joint views.
AverageCase residual independence remains an explicit modeling assumption.

## Reproduction and evidence

Run the local simulation with:

```sh
cargo test --release -p mxx-exponent-lut --test parameter_search_unit universal_approximate_minimum_depth_window -- --ignored --nocapture
cargo test --release -p mxx-exponent-lut --test parameter_search_unit universal_approximate_complete_lower_depths -- --ignored --nocapture
```

The completed evidence directory is
`test_data/universal_approximate_depth_window/1788660855725121579/`:
`complete-search.json` contains the full enumeration, `depth-29-drop-3.json` contains
the selected report, `simulation.log` records completion, and
`security-depth29.log` records the direct estimator output and
`security-provenance.md` records its exact-q command and estimator commit. The exact-q CLI
invocation used the `full_modulus` field of that selected report and
`--s-dist '{"name":"Ternary"}' --e-dist '{"name":"DiscreteGaussian","stddev":4.0}'`.

The companion
`docs/plans/exponent-lut-universal-approximate-depth-window-results.csv`
contains every exact-baseline result and the four accepted frontiers;
its values were checked programmatically against `search.json`.
No GPU, benchmark, complete refresh execution, commit, or push was performed
for this search.
