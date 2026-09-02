# PBC diagnostics

`pbc_diagnostics` is an opt-in measurement tool for the sparse-LWR PBC. It
does not select production parameters, change a profile, estimate operational
noise, or create an encrypted artifact. The output is a reproducible report
schema for a particular invocation and should be retained with the parameters
used to obtain it.

Run a named profile with explicit dimensions and trial count:

```text
cargo run -p mxx-exponent-lut --example pbc_diagnostics -- \
  --nu 256 --h 32 --trials 100 --profile Conservative
cargo run -p mxx-exponent-lut --example pbc_diagnostics -- \
  --nu 256 --h 32 --trials 100 --profile PaperEvaluation --format json
```

`--width-limit N` is optional for either named profile. A custom profile must
specify every scheduling parameter explicitly:

```text
cargo run -p mxx-exponent-lut --example pbc_diagnostics -- \
  --nu 256 --h 32 --trials 100 --profile Custom \
  --c 3 --k 35 --max-seed-attempts 128 --width-limit 8
```

The command samples an exact-weight support first, then obtains a fresh
operating-system-backed root seed for each trial. It retries only the layout
seed derived from that root seed. It never resamples the support and never
silently changes `c`, `k`, the retry limit, or the selected profile.

The accepted seed is conditioned on successful scheduling for the sampled
support. Version 1 therefore supports an honestly generated support fixed
before the seed, not an adversarial support chosen after public hashes are
known. The paper-evaluation profile's often cited `<1 bit` retry-leakage
argument is only a concrete rationale for its tested parameters, not a
universal robustness proof; use a later robust-PBC construction when that
guarantee is required.

Reports include first-attempt and cumulative retry success, failure causes,
bucket-width and rectangular-padding percentiles, real and dummy selector
package counts, matching/layout timings, and structural cell/reduction
counts. Counts and timings are totals over accepted trials; `accepted_trials`
is included so per-layout averages can be derived. Width and padding
percentiles are per-layout accepted samples. The stable performance category
names are:

```text
pbc_layout_hashing
pbc_matching
pbc_real_selector_packages
pbc_dummy_selector_packages
pbc_bucket_cell_evaluations
pbc_bucket_reductions
pbc_padding_overhead
```

The timing and structural metrics contain no operational-noise recurrence;
noise bounds continue to be derived by the generic DSL correctness machinery.
