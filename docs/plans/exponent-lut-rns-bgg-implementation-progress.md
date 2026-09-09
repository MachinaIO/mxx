# RNS-BGG implementation progress

Implementation follows `docs/plans/exponent-lut-rns-bgg-implementation-plan.md`.
The implementation and the user-approved validation scope are complete.
The user accepted one successful noisy PRF GPU round trip as sufficient and
requested no further full-refresh round-trip validation. The running full
refresh and queued benchmark were stopped. Lean and the special squared-modulus
two-column entrance remain deferred.

## Implemented

- The former BFV backend is now `rns_bgg` / `RnsBggScheme`, with the external
  selector `rns-bgg`. Its five files remain `mod.rs`, `encoding.rs`,
  `public_key.rs`, `noise.rs`, and `gpu.rs`. BGG remains a separate backend.
  Historical BFV simulation reports are unchanged.
- The existing fixed-size `Blocks` carrier accepts rectangular `Mat` entries
  and bounded `Preimage` entries, using the same graph schema implementation.
  State keys and encodings have `2D_i` columns and satisfy
  `c_i = s*A_i - p_i*mu_i*t*G_(i+1) + e_i`. Only the terminal chain endpoint
  retains two polynomial columns.
- Private products prepare `U = Dec_low(C)` once and transport its bounded
  coefficients to the input modulus. Online evaluation performs the prepared
  row product and ModDown. LUT alignment also prepares
  `B = ModDown(sigma(A)*U)` and `V = Dec_low(B)`. Branch caches retain U, V,
  the helper public matrix and sigma, without unused raw C/B matrices.
- Public and online LUT/terminal branch sums occur before the outer ModDown.
  Payload targets, positional digit weights and normalization are applied
  before fresh RHS error sampling. Input and helper preparation keep the
  payload secret in preprocessing and the mask secret in online preparation.
- Scalar and Family cryptographic operations use the same scalar arithmetic.
  Family wrappers select elements and preserve shared graph fields. Generic
  Program, PRF, terminal and direct-refresh orchestration remain common;
  application crates do not acquire backend arithmetic or new node kinds.
- Prepared scalar, Family, terminal and Program artifact schemas retain bounded
  actions and actual matrix widths. Online imports consume prepared artifacts
  without re-decomposition or public aggregation/switching.
- Existing `CenteredExtend` transports bounded graph values through the DSL,
  IR and runtime. GPU transport reuses compact allocation and asynchronous
  `gpu_small_matrix_copy`. The only CUDA change permits contexts on the same
  device instead of requiring the identical context. Existing readiness and
  consumer events remain. No kernel, CPU coefficient transfer, or global
  synchronization was added.
- Worst-case digit action uses `N*D*b`; conditional second moments use
  `N*D*b^2/2`, with the corresponding drop factors. Selector ciphertext errors
  are all included. Existing average-case independence heuristics remain.
  Program event counts use each value's level width; refresh tail accounting
  sums actual preterminal state coordinate counts for the emitted labels.
- Parameter planning searches prime-prefix chains using the digit-based error
  transfer and a Pareto frontier of error and estimated peak storage. The
  manifest retains each level's radix. Storage estimates account for wide
  matrices and compact digits, excluding allocator/NTT workspaces.
- Refresh outputs retain all `2D_ref` columns. Data targets contain
  `H*p_ref*mu*t*G_ref`, with `H = q_ref/prime`. Channels retain all output primes
  in a common preterminal modulus. Channel residues are rounded before CRT
  recomposition, and the returned encoding can enter the next LUT directly.
- Mask/fresh digit planning includes every output coefficient in its fixed
  point. Masks are channel-specific; fresh digits are shared across channels.
  The integration oracle reconstructs every fresh digit with its base-p weight
  and checks every output column after artifact reopen and continuation.
- Architecture and test documentation describe the new representation. Refresh
  reports include per-level modulus bits, CRT depth, radix and state columns,
  plus algebraic product counts for reused radix digits and the paper's
  one-prime digits on the same chain. These are not kernel-launch counts or
  end-to-end timing ratios.

## Verified results

| Validation | Result |
| --- | --- |
| Workspace CPU library build | Passed without warnings |
| Workspace GPU library build | Passed without warnings |
| RNS-BGG CPU unit sweep | 36 passed, 16.02 s |
| RNS-BGG GPU unit sweep | 17 passed, 580.71 s |
| Updated all-column GPU refresh/preprocessing checks | 2 passed, 233.04 s |
| GPU bounded-digit transport, identical binary | 3/3 passed, 0.17 / 0.17 / 0.16 s |
| Primitive CPU bounded extension | Passed, 0.03 s |
| Runtime exact-RNS graph | Passed, less than 0.01 s |
| Actual PRF GPU integration, average-case failure exponent 5 | Passed, 4.38 s |
| PRF and refresh integration builds | Passed without warnings |
| Existing BGG decoder check outside sandbox | Passed, 11.43 s |

The CPU/GPU scheme suites cover private products, aligned LUTs, linear
operations, native/strided readout, scalar/Family artifacts, independent online
secrets and noiseless refresh followed by another LUT. The PRF integration
reopens its artifact store and evaluates native digit readout; it reports an
average-case readout bound of 9 and deterministic bound of 23.

A full exponent-lut library run reported 231 passed, three failed and three
pre-existing ignored tests. The BGG failure was CUDA being unavailable inside
the sandbox; its outside-sandbox rerun passed. The other two failures are in
the explicitly deferred Lean exports (`no actual private selector caches
found`). This is not a green full-library result.

## Validation boundary and final audit

The user explicitly accepted the successful noisy one-output PRF integration
as sufficient. That fixture uses CBD eta=2, average-case failure exponent 5,
native digit readout, and artifact-store reopening. It is not a noiseless test.
The additional full-refresh process and its queued benchmark were terminated
at the user's request; temporary artifacts from the interrupted run were
removed. No further runtime verification or benchmark measurement is pending
within this accepted scope.

Before interruption, the full-refresh N=4 setup selected base 4, one fresh digit,
15 mask digits per each of three CRT channels, 2,208 distinct PRF outputs,
2,304 terminal digit emissions, and 49,815,936 estimated terminal artifact
bytes. Its modeled exposed error bound was 374 per channel. These remain setup
results, not a successful noisy full-refresh runtime or a measured benchmark.
No end-to-end speedup is claimed. The plan's fixed-chain digit-width/product
comparison and emitted per-level work counts describe the reuse overhead;
measured comparison remains unperformed.

| Requirement | Evidence / accepted boundary |
| --- | --- |
| Rename and five-file backend layout | `rns_bgg` source directory, public exports, both integration selectors and architecture documentation |
| Wide state invariant and exact low-ring gadget | CPU/GPU private-product, aligned-LUT, linear and exact-RNS tests |
| Prepared bounded transport and no new GPU kernel | Existing CenteredExtend/copy path, three transport runs, artifact/cache graph tests |
| Shared scalar/Family cryptography | Scalar calls from structural wrappers; private/LUT/readout/Program Family suites |
| Public preprocessing versus mask-secret preparation | Independent-online-secret and reopened-artifact tests |
| Faithful gate noise and actual executed PRF | Common Program trace, all-selector error gains, corrected level-width event counts and noisy PRF integration |
| Reusable refresh state and full coordinate routing | All-column noiseless GPU refresh followed directly by another LUT; noisy full refresh deliberately not completed |
| Parameter/storage and radix-reuse accounting | Planner tests, fixed-point refresh simulation, per-level dimensional report and explicit analytic reuse comparison |

Small-fixture arithmetic and modeled acceptance do not certify the earlier
N=65536, approximately 1300-bit, 21-step, security-100 target. The deferred Lean
failures are recorded above; the complete workspace is not claimed test-green.
