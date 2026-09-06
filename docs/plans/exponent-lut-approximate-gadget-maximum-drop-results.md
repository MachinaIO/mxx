# Fixed-N approximate-gadget maximum-drop search

The completed local AverageCase search selects **7 dropped CRT towers out of
33**, with `N=65536`, 50-bit CRT primes and 25-bit gadget base. This reduces
the two-row gadget from **132 to 104 columns**, a **21.21%** reduction. The
full ciphertext modulus retains all 33 primes (1650 bits). The retained
gadget has 26 towers; the five-tower refresh switching groups are a separate
parameter.

This is acceptance of the explicitly specified noise model, including the
even-residual bridge below. That bridge is not implemented in the runtime
backend. No runtime correctness or joint-security proof is claimed.

## Selected parameters and checks

| Quantity | Selected value |
| --- | ---: |
| Ring dimension | 65536 |
| Full / retained / dropped CRT towers | 33 / 26 / 7 |
| CRT prime / gadget base bits | 50 / 25 |
| Full / retained gadget columns | 132 / 104 |
| Native sparse-LWR `Q,p,v,h` | 4096, 32, 8192, 18 |
| Refresh switching width | 5 towers per slot |
| Mask / fresh digits | 38 / 1 |
| Failure exponent / mask security target | 100 / 100 |
| Tail correction bits | 0 |
| Joint tail event count | 10856693760 |
| Exact squared tail factor | 4678749 / 25000 |
| Maximum refresh `F_avg` | 856291577040878475 |
| PRF / bridge-inclusive readout tail-bound bit lengths | 1283 / 1421 |

The minimum strict rounding margin is
`1605368353711956442992508959171823471431036302826486897656957`.
The minimum smudging margin is
`949521865682183247284666582272694483406604524717315457024`.
Both are positive. Fresh support is the unchanged 31, below every 50-bit
plaintext prime. Bit lengths describe unreduced modeled bounds, not sampled
noise or measured decryption failure rates.

At `N=65536`, a fresh rough uSVP/dual-hybrid estimate against the actual full
modulus gives floor **102 bits** at depth 33 and **97 bits** at depth 34.
The estimator commit is `e35f45b7976a90a79c3c6625a45bbc344c1abc67`, with
ternary secret, Gaussian sigma 4, and unlimited scalar samples. These are
rough marginal estimates, not an exhaustive attack evaluation. The unchanged
sparse-LWR tuple retains its pinned estimate `103.18615790798486` bits.
The previously documented cross-modulus joint-view security obligation
remains unresolved; marginal estimates do not establish security of shared
secret anchors, preimages, and correlated public views.

## Noise model and bridge

For dropped-prime product `P`, primitive coefficient error is bounded by
`k*floor(P/2)`. AverageCase assumes independent uniform centered dropped CRT
residues. Their variance is
`sum_r (P/r)^2*(r^2-1)/12`. An augmented ternary secret weights this by
`1+2N/3`; the simulator stores **four times physical variance**. The worst
secret-weighted coefficient bound is `(N+1)*k*floor(P/2)`.

The existing program interpreter propagates one such residual at each private
or public multiplication and two at each LUT branch. Add retains its existing
correlated transfer. The readout trace weights residuals by the existing
plaintext-target gains. Target and normalization scalar multiples of the
shortened gadget have zero discarded residual. Retained direct matrices use
the balanced-digit cap `2^24` with their actual retained row count.

The final bridge converts the shortened-gadget relation to the full gadget
used by the unchanged refresh routes and exact decoder. For each full gadget
column, compute its scalar CRT correction `R`. If `R` is odd, replace it by
the smaller-magnitude of `R+P` and `R-P`. The resulting even residual preserves
all dropped residues. Retained columns are the balanced decomposition of the
corrected retained residues. This is an explicitly modeled custom bridge,
not the default primitive decomposition. Its selected column gains are
`L1=497893694` and `L2_squared=5867243521787205`.

Native `p=32` readout payloads have magnitude at most `31/2`. Even residuals
therefore give an integer bridge bias `31*max|R|/2`, without inverse-of-two
wraparound. The bridge scales stochastic variance by `L2_squared`; its fixed
bias is transported by **L1** through every refresh route and added outside
the statistical tail. Earlier pre-bridge comparison results must not be
interpreted as this bridge-inclusive calculation.

## Maximality scope

The incumbent was found by testing all retained counts at depth 33 with mask
counts 38 and 39. To exclude every larger dropped count, the completed grid
checked all **325** pairs `9 <= depth <= 33`, `8 <= dropped < depth`.
Depths at most 8 cannot improve the incumbent because at least one tower must
remain. The security-qualified depth cap is 33 under the marginal screen
above. Switching groups remain the existing contiguous five-tower layout;
arbitrary alternative switching layouts are outside this maximum claim.

Every pair was rejected by an optimistic bound valid for all mask counts
3 through 39: omit all nonnegative mask-route variance and bias, use the
minimum joint tail factor at 3 digits, but allow the maximum mask modulus
`32^39`. Keep fresh, state, decoder, paired-rounding, and fresh-route bridge
bias terms. Each persisted witness violates either the smudging inequality
or the strict rounding inequality. No pair required the implemented exhaustive
mask-count fallback. Counts below 3 cannot cover `Q=4096`; counts at least 40
cannot fit the five-tower spacing, which is strictly below `2^200`.

## Evidence and reproduction

The selected full decimal parameters, per-step contributions and margins are
in `test_data/profile_b_approximate_gadget_noise/1788656461319031162/retained-26.json`.
The complete proof grid and copied raw logs are in
`test_data/approximate_gadget_maximum_drop/1788656642909207371/`.
`search.json` records all 325 rejection witnesses; the copied
`mxx-approximate-n16-exact-security-cap.log` contains actual q values and raw
security estimates.

```sh
cargo test --release -p mxx-exponent-lut --test parameter_search_unit approximate_gadget_maximum_drop_pilot -- --ignored --exact --nocapture
cargo test --release -p mxx-exponent-lut --test parameter_search_unit approximate_gadget_maximum_drop_frontier -- --ignored --exact --nocapture
cargo test --release -p mxx-exponent-lut --lib noise::bridge_tests -- --nocapture
cargo test --release -p mxx-exponent-lut --lib noise_refresh::simulation::tests -- --nocapture
cargo +nightly fmt --all
```

The pilot passed in 5.60 seconds; the 325-pair frontier passed in 39.02
seconds. The scalar bridge parity/algebra unit and all 25 existing simulation
units passed. Release compilation was warning-free. These are local CPU
component simulations and unit checks; no GPU or runtime integration test
was executed.
