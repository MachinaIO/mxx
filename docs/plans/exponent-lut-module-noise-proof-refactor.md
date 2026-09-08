# Exponent-LUT module, noise-trace, and Lean refactor

## Status and scope

This is an implementation plan, prepared on 2026-09-08 against the current working tree. It does not authorize or perform implementation changes. The concurrent BFV implementation and its running validation are outside this planning task. Reconcile its final file inventory before applying this plan; do not overwrite its changes.

The refactor has four deliverables: clear ownership of generic versus scheme-specific code, fewer Rust modules with coherent responsibilities, fewer redundant concepts/data structures, and a common Program-driven noise/proof interface supporting BGG and BFV. It preserves the accepted BFV double-scale arithmetic in `docs/plans/exponent-lut-bfv-double-scale.md`.

Subsequent user instructions set the symmetric backend inventory to five files:
`encoding.rs`, `gpu.rs`, `mod.rs`, `noise.rs`, and `public_key.rs`. This replaces
the original three-file estimate below, without relaxing the requirement to
extract common orchestration instead of concatenating backend implementations.
Intermediate validation uses one successful run of affected CPU unit tests;
the final GPU round trip requires two successful runs. GPU validation is deferred
until the final implementation unless a GPU-specific behavior changes.

The latest API revision restores direct private multiplication as a primitive. It supersedes the earlier identity-LUT interpretation of `lut_id = None`; the double-scale arithmetic remains unchanged. The resulting contracts are:

- `ExponentLUTEncodingScheme` is the encoding trait name.
- Program primitives are `Add`, `MulPublic`, `MulPrivate`, and `Lut { rhs, lut_id }`. `private_mul` emits `MulPrivate` directly; `Lut` requires a `LutId`. There is no optional table ID or implicit identity LUT in the new representation.
- BFV stores two polynomial components and computes directly with 2-by-2 polynomial blocks. It uses ordinary RNS ModDown, without coefficient reconstruction into full-width integers or gadget matrices.
- Preprocessing owns payload-secret and public-data work. Online preparation generates mask-secret-dependent helpers and decoders. Online evaluation consumes those artifacts and still performs encoding ModDown.
- Add and public multiplication use compatible levels/scales. Native signed PRF digits, distinct mask/fresh digit counts, fresh-target weights, and aggregate-before-rounding semantics are preserved.
- Public construction contracts replace unnecessary runtime ciphertext validation. This refactor must not introduce coefficient scans, public preparation during cached evaluation, or expanded host loops over structural family members.

## 1. Findings in the current source

| Current location | Finding | Consequence |
| --- | --- | --- |
| `crates/exponent-lut/src/scheme.rs` | Contains the generic trait and `PreparedProgram`, but also `BggScheme` and its implementation. | Separate the definition of the interface from every concrete implementation, including tests. |
| `crates/exponent-lut/src/encoding.rs`, `public_key.rs`, `rhs.rs` | Use BGG wires, sampler layouts, GSW packages, and decomposition artifacts. | These files belong under `bgg`, despite generic-looking names. |
| `crates/exponent-lut/src/lib.rs` | Reexports concrete BGG types and defines an error containing BGG/import/RHS errors. | Clean up the facade and error types as well as moving files. |
| `crates/exponent-lut/src/noise.rs` | Mixes reusable exact variance/norm arithmetic with BGG decomposition gains and transfers. | Extract only the mathematical utilities; keep the BGG formulas under `bgg`. |
| `crates/exponent-lut/src/noise_refresh/{setup,refresh,readout,simulation,artifacts}.rs` | Still contains concrete BGG types and operations. PBC adapters also import them. | Moving `scheme/bgg_prf.rs` alone will not make refresh generic. |
| `crates/exponent-lut/src/bfv/noise.rs` | Already implements `ExponentProgramLowering` and records per-gate worst-case bounds. | Extend the existing trace with moments and source relationships; do not build another traversal. |
| `crates/exponent-lut/src/bfv/` | Currently contains 30 Rust files, including separate linear/family/artifact/GPU fragments. | Extract generic orchestration and transport; target three backend files, not one file per interface. |
| `crates/exponent-lut/src/lean/` | Currently contains 39 Rust files in addition to `src/lean.rs`. Many inspect graph topology and emit theorem bodies as Rust strings. | Replace the handwritten proof generator, while retaining mechanical IR export and the proof coverage it provides. |

The Rust directory in the last row is `src/lean/`. The separate `crates/exponent-lut/lean/` directory holds Lean source and Lake configuration.

## 2. Ownership rules and target tree

Files directly under `src` may contain encoding abstractions and algorithms that are both scheme-independent and not specific to noise refreshing. They must not import `mxx_bgg`, `crate::bgg`, or `crate::bfv`, apart from module declarations in `lib.rs`. They also must not depend on `noise_refresh` for otherwise general LUT declarations.

`src/noise_refresh` owns the refresh application and its PRF, PBC, label topology, digit scheduling, preparation/evaluation orchestration, and application reports. Its implementation depends on traits and associated types, never a concrete encoding backend. Backend adapter implementations live under `bgg` or `bfv`; they may depend on refresh contracts. This dependency is intentionally one-way.

```text
crates/exponent-lut/src/
  lib.rs                 # module declarations and genuinely generic exports/errors
  scheme.rs              # encoding/noise capability contracts and generic preparation
  program.rs             # declarations, builder, bindings, and shared traversal
  layout.rs              # native exponent-domain definitions
  readout.rs             # general value/terminal declarations and generic evaluation
  noise_utils.rs         # exact moments, norms, source bookkeeping, trace containers
  utils.rs               # shared polynomial construction and structural reductions
  lean.rs                # thin, implementation-independent IR/claim export adapter
  bgg/
    mod.rs               # concrete types and capability implementations
    public_key.rs        # public sampling, preparation, and public gate arithmetic
    encoding.rs          # encoding sampling and online gate arithmetic
    noise.rs             # BGG transfers, decomposition gains, refresh formulas
    gpu.rs               # GPU-only wrappers and tests
  bfv/
    mod.rs               # two-column types/parameters and capability implementations
    public_key.rs        # public sampling, preparation, and public gate arithmetic
    encoding.rs          # encoding sampling and online gate arithmetic
    noise.rs             # BFV primitive transfers, moments, chain/refresh planning
    gpu.rs               # GPU-only wrappers and tests
  noise_refresh/
    mod.rs               # refresh capability traits and application exports/errors
    prf.rs               # profile, canonical digit function, ordinary/batched orchestration
    refresh.rs           # public preparation, online preparation, evaluation, digit routing
    setup.rs             # generic production setup and artifact workflow
    simulation.rs        # application-wide scheduling, budgets, and reports
    lean.rs              # refresh protocol declaration and proof-export orchestration
    pbc/
      mod.rs             # public layout and route declarations
      schedule.rs        # schedule construction
      artifacts.rs       # scheme-independent PBC/label artifact transport
```

Neither backend needs separate `refresh.rs`, `artifacts.rs`, `program.rs`, `readout.rs`, or Rust `lean.rs` files. Their names describe interfaces or applications, not necessarily independent backend responsibilities. Keep the concrete cryptographic methods and small trait implementations together in `mod.rs`.

Achieve this by deleting duplicated orchestration and transport, not by concatenating all current files. The common Program interpreter owns traversal and cache bookkeeping; generic artifact transport uses existing `GraphValue`/schema and runtime artifact facilities; common readout owns target scheduling/finalization; noise refresh owns its protocol. Backends provide concrete operations, branch/target preparation, and field/binding declarations. Backend-specific proof endpoint bindings also stay beside the implementing methods in `mod.rs`.

The target is three files per backend. Respect the repository's approximately 2,000 implementation-line guideline after extraction and deletion; count non-test code accurately rather than stopping at a test-only import near the top. If a remaining cryptographic implementation still exceeds it, justify one substantial split from measured code size. Do not pre-create files for every trait or small adapter.

The proposed `bgg` directory belongs to the exponent-lut crate and is an adapter over `mxx-bgg`; it does not duplicate or move the lower-level BGG crate.

### Concrete migration and consolidation

| Existing code | Destination |
| --- | --- |
| Generic portion of `scheme.rs`; scheme-independent noise capability | `src/scheme.rs` |
| `BggScheme`, its implementations, and its concrete unit fixtures | `src/bgg/mod.rs`; move generic loops/cache handling to the root interpreter |
| `encoding.rs`, `public_key.rs`, `rhs.rs` | Concrete sampling, public/encoding operations, and RHS types in `src/bgg/mod.rs`; generic traversal/cache handling in the root interpreter |
| `scheme/bgg_prf.rs`, `scheme/bgg_refresh.rs` | Concrete operation adapters in `bgg/mod.rs`; PRF/refresh orchestration in `noise_refresh` |
| General `ValueReadout`/identity/table definitions mixed into `noise_refresh/readout.rs` | `src/readout.rs`; PRF-specific table construction stays in `noise_refresh/prf.rs` |
| `scheme/{terminal,readout}.rs` | `src/readout.rs`, with general capability contracts and `PreparedTerminal` |
| Refresh and PRF-specific contracts in `scheme/{refresh,prf}.rs` | `noise_refresh/mod.rs`; remove the `src/scheme/` directory |
| `algebra.rs`, small shared reduction helpers including `sum_terms` | `src/utils.rs`; keep distinct algorithms where their graph/error semantics differ |
| Exact rational and norm helpers from `noise.rs` | `src/noise_utils.rs`; move gadget/decomposition machinery to `bgg/noise.rs` |
| `noise_refresh/{direct,evaluation,digits}.rs` and generic portions of `refresh.rs` | `noise_refresh/refresh.rs` |
| `noise_refresh/prf_batch.rs` | `noise_refresh/prf.rs` |
| Generic setup/artifact names, manifests, and workflow from `noise_refresh/artifacts.rs` | `noise_refresh/setup.rs`; generic transport uses core/runtime facilities; concrete field/binding declarations stay in each scheme's `mod.rs` |
| Concrete BGG imports, payload arithmetic, preimages, decoder construction, and formulas currently in `noise_refresh/*` | `bgg/mod.rs` for cryptographic operations and bindings, `bgg/noise.rs` for formulas |
| BFV `blocks.rs`, `encoding.rs`, `linear.rs`, `family.rs` | `bfv/mod.rs`, including structural carriers and concrete operations; share family traversal where its semantics coincide |
| BFV `program.rs`, `family_adapter.rs`, `prf.rs` | Root Program/`noise_refresh` orchestration; small concrete adapters in `bfv/mod.rs` |
| BFV `terminal.rs`, operation portions of `terminal_family.rs`, `readout.rs` | General readout/family scheduling in `src/readout.rs`; BFV target and block arithmetic in `bfv/mod.rs` |
| All BFV `*_artifacts.rs`, `artifacts.rs`, artifact portions of `terminal_family.rs` | Common artifact traversal/import/export machinery; compact concrete metadata declarations in `bfv/mod.rs` |
| BFV parameters' noise recurrences, planner logic, `refresh_planning.rs`, and `noise.rs` | `bfv/noise.rs`; parameter types and validated bases remain in `bfv/mod.rs` |
| Small BFV `*_gpu.rs` files | `bfv/gpu.rs`; preserve the feature/test guards and GPU-only filename rule |
| PBC layout/evaluation/diagnostic helpers | `noise_refresh/pbc/mod.rs` where generic; concrete encrypted transports move to the backend |

`utils.rs` is not a destination for every small function: a helper used only by one operation stays inline in its owning file. No backend error type may leak into `utils.rs`, `readout.rs`, or `noise_utils.rs`; use generic/core errors and map them at the backend boundary.

Remove obsolete module paths and concrete root reexports, updating all workspace consumers, examples, tests, estimators, and documentation together. Do not maintain compatibility aliases or old artifact-format fallbacks. Keep common `lib.rs` error variants free of BGG/BFV and refresh-only errors.

## 3. Make noise refresh depend only on contracts

The existing generic evaluator is a starting point, not the completion criterion. Audit public signatures, associated type equalities, error variants, serialization, tests, and private helpers throughout `noise_refresh`, including PBC.

The division of responsibility is:

| Shared application code | Scheme adapter |
| --- | --- |
| Compile the native PRF schedule; declare selectors and public label families | Prepare/evaluate a selection using the backend's cryptographic operations |
| Choose label ranges, mask/fresh digit counts, and weights | Encrypt weighted digit/data targets and expose their prepared artifacts |
| Connect Add/MulPublic/MulPrivate/LUT operations and terminal reductions | Perform each public/encoding operation and supply its noise transfer |
| Group CRT channels and order sum, switch, decode, extract, recompose | Supply each arithmetic operation and associated carrier/metadata |
| Decide what belongs to preprocessing versus online preparation | Generate payload-secret material, or mask-secret helpers/decoders, in the corresponding phase |
| Bind setup identity, stage links, labels, output keys, and artifact roles | Serialize concrete matrices, rows, witnesses, and caches |
| Collect reports and solve application digit/budget conditions | Supply scheme-specific exposed-error and resource formulas |

Use the existing encoding and optional capability traits. Consolidate overlapping capabilities where they describe the same operation, but retain genuine requirements such as witness-based versus direct-decoder issuance. BFV must not implement dummy anchors; BGG must not be forced into BFV's direct-decoder construction. The application must keep operation ordering visible instead of delegating the entire refresh protocol to an opaque backend callback.

Generic setup containers carry associated `Parameters`, `PreparedArtifacts`, `OnlineHelpers`, and `Descriptor` types. There is no `match Bgg/Bfv`, concrete downcast, backend-name check, or BGG default/fallback in `noise_refresh`. Both production routes must instantiate this application layer, including setup and artifact import; a standalone generic arithmetic fixture is insufficient.

Distinct mask labels include the CRT channel. Fresh digits use the same logical source across channels. Preserve source and artifact identity through family slicing, cache export/import, and broadcast. This is also necessary for correct noise accounting.

Move concrete end-to-end fixtures into `bgg`/`bfv` test modules. Shared algorithm tests may use a small recording implementation to check operation ordering, without introducing concrete scheme imports into the shared modules.

### Direct private multiplication and explicit LUT tables

Restore `MulPrivate` as a primitive. Although this adds one enum variant, it removes the overloaded optional-table meaning, the absent-table/absent-RHS combination, and separate implicit-identity handling across interpreters, caches, noise traces, and proofs. This is a smaller semantic interface than using `Lut` for operations that do not evaluate a table.

```text
Gate::Add { lhs, rhs, output }
Gate::MulPublic { private, public, output }
Gate::MulPrivate { lhs, rhs: RhsId, output }
Gate::Lut { input, rhs: Option<RhsId>, lut_id: LutId, output }

builder.private_mul(lhs, rhs)       -> MulPrivate
builder.lut(input, table, rhs)      -> Lut with a required table ID
```

The RHS remains a preprocessing-fixed cryptographic operand, not an arbitrary online state-state ciphertext multiplication. Keep explicit-table binary LUT fusion: it still absorbs the RHS into each fresh table branch rather than first evaluating `MulPrivate` and then doing a second multiplication in every LUT branch.

Migrate existing `Lut { lut_id: None, rhs: Some(..) }` declarations to `MulPrivate`. For the old no-table/no-RHS case, reuse the input wire if it is a genuine no-op. If the caller needs a BFV level/scale transition, schedule it explicitly using a prepared unit RHS and the direct multiplication path. Do not silently remove a required level transition or disguise it as `MulPublic(1)`. An explicitly registered identity table remains a real LUT, with the ordinary explicit-table semantics.

**BGG direct path.** The current `src/encoding.rs` and `src/public_key.rs` already return from the absent-table case before entering the LUT helper/branch path. Reuse that direct implementation, rename its misleading identity-LUT helpers, and give it a dedicated primitive method/cache entry. For the current same-modulus convention:

```text
c = s A - mu r G + e
D = G^-1(C),      r C = nu r G + F
A_out = A D
c_out = c D = s A_out - mu nu r G + e D - mu F.
```

Preprocessing prepares/caches `D`; online evaluation performs the external product and stops. There are no LUT automorphisms, coefficient-table construction, mask switching, or branch reductions. The current BGG gate keeps its modulus and has no ModSwitch at this point. If a supported BGG level-transition contract calls for one, apply only that contract's switch and then stop; do not insert a new modulus change merely because the direct operation is renamed. Gadget layout/scale compatibility must be part of any such contract.

**BFV direct path.** This can be simpler than invoking the existing one-branch flat-LUT implementation. At `Q = p q`, with next state scale `p_next`, prepare one fixed two-by-two ciphertext in the low base:

```text
c = s A - p mu r + e                 (mod Q)
r C = p_next nu r + F                (mod q)
Ctilde = centered lift of C to Q
A_out = MD_(p,q)(A Ctilde)
c_out = MD_(p,q)(c Ctilde).
```

Let `rho = center_p`. Direct substitution gives:

```text
c_out = s A_out - p_next mu nu r + e_out          (mod q)
e_out = [e Ctilde - p mu F
         + s rho(A Ctilde) - rho(c Ctilde)] / p  (mod q).
```

The RHS equation's possible wrap by `q` disappears after multiplication by `p` modulo `Q`. Thus no automorphism helper, auxiliary mask public matrix, internal `B_k` rounding, identity table, or branch container is needed. Public preparation performs the public product/ModDown once; cached online evaluation performs the row product/ModDown once. Keep both operations on polynomial blocks with exact RNS conversion.

The direct BFV bound is correspondingly:

```text
B_out <= B_in K(Ctilde)/p + ||mu||_1 B_F + (S+1) gamma,
gamma = (p-1)/(2p).
```

Derive the direct operation's average-case transfer from its four signed error terms, retaining correlations between the inherited error, fixed ciphertext/fresh error, and rounding residues. Do not reuse the old LUT bound with a fictitious mask-helper error or internal public-rounding term. Prove this relation against the actual generated direct-path IR.

For one-hot PRF selection, combine weighted fixed RHS ciphertexts in the low base first and then call the direct `MulPrivate` path. Its fresh error contains every participating ciphertext error with its public multiplier; one-hot selects the payload, not the error sources. Structural batches remain structural. The optimization is a different rounding schedule from summing independently rounded outputs: prove the intended plaintext relation and trace its actual error rather than requiring bitwise ciphertext equality.

Migration includes Program validation/lowering, scalar/family bindings, operation caches and artifact descriptors, noise transfers, resource estimates, Lean claims, and callers that previously relied on `None`. Add a narrow graph-structure check that the direct BFV path has only its public and encoding ModDown operations and no LUT helper/branch work; check that BGG's direct path has no LUT work and no new switch in its unchanged-modulus mode.

## 4. Common noise utilities and scheme-owned transfers

### Trace structure

Extend `ExponentLUTEncodingScheme` with a noise evaluator capability parallel to its public and encoding evaluators, using the existing `ExponentProgramLowering` interface. `NoiseState` alone is not enough to supply transitions. Run all three interpreters over the same frozen Program and `ProgramGateInstance` identifiers.

`noise_utils.rs` owns exact rational arithmetic, generic mean/variance or second-moment containers, norm arithmetic, symbolic source relationships, and report/trace containers. It contains no gadget base, decomposition formula, BFV level-drop recurrence, or refresh digit schedule.

Each wire/gate trace records:

- Worst-case coefficient error and the physical error coordinate convention.
- Mean or a mean enclosure, centered variance/second-moment information, and the assumptions used by an average-case estimate.
- Plaintext coefficient bound, l1 bound, and l2-squared bound where needed, with explicit signed-monomial/one-hot/public-polynomial contracts.
- Level/scale metadata supplied by the scheme and references to inherited, fresh, helper, and rounding error sources.
- A term-by-term transition breakdown, so the planner, reports, and proof certificates consume the same result.

BGG's current `AverageVariance` describes doubled coordinates. Reuse its exact rational arithmetic, but convert explicitly at the BGG adapter boundary: variance of `2e` is four times variance of physical `e`. Do not silently reinterpret existing report values as BFV variances.

BFV already has primitive-driven worst-case tracing. Extend it rather than replacing it with a chain-only or refresh-only simulator. The optimized composed-selector path must expose a direct `MulPrivate` transition with the actual summed RHS error gain. A primitive noise estimate from an unoptimized per-slot evaluator must not be substituted for a different executed graph.

### Primitive transitions

| Operation | Worst case | Average case and dependency handling |
| --- | --- | --- |
| Add | `B_out = B_l + B_r`; propagate the actual plaintext sum/domain | `mean_out = mean_l + mean_r`; `Var(out) = Var(l) + Var(r) + 2 Cov(l,r)`. Reusing the same wire twice yields `4 Var(e)`, not `2 Var(e)`. |
| MulPublic by integer-lift polynomial `a` | `B_out <= ||a||_1 B_in`; propagate plaintext norms | Treat convolution as a linear map. `||a||_2^2 Var(e)` is valid for the specified independent, centered coefficient model; otherwise retain covariance or use a justified upper bound. Mean follows the same linear map. |
| Unary/binary LUT with an explicit table | Use the backend's actual cancellation and rounding equation; include all helper/fresh contributions | Derive moments from that same equation, with reused state/helper sources and rounding dependence preserved. Do not declare branches independent merely because they execute in parallel. |
| MulPrivate | Use the direct product equation and only its necessary rounding terms; BFV has no LUT mask-helper or internal branch-rounding terms. | Trace the direct RHS action, its fresh error, and any required rounding, including actual source dependencies. |
| Family reduction | Sum error bounds for every evaluated member; one-hot may tighten the payload bound only | Independent family indices may add variances only under the sampler/model contract. Broadcast, common helpers, and common ancestors preserve shared source identity. |

For BFV, let `p = q_i/q_(i+1)`, `rho = center_p`, `M_k = sigma_k(A) Ctilde_k`, `B_k = MD(M_k)`, `K = sum_k Ad_k Btilde_k`, and `z` be the raw online aggregate. The double-scale note, equations (6.4)–(6.6), gives the exact signed error decomposition:

```text
e_out = sum_k [sigma_k(e) Ctilde_k / p
             + ed_k Btilde_k / p
             - sigma_k(mu) F_k
             + sigma_k(s) rho(M_k) / p]
        + s rho(K) / p - rho(z) / p.
```

Use this equation as the source for both BFV analyses. With `S = ||s_bar||_1 + 1`, `gamma = (p-1)/(2p)`, and `K(C)` the maximum column sum of polynomial l1 norms, a general payload bound gives:

```text
B_out <= [B_in sum_k K(Ctilde_k) + B_d sum_k K(Btilde_k)] / p
         + ||mu||_1 sum_k B_F,k + (J S + S + 1) gamma.
```

Round the final bound outward using exact integer/rational arithmetic. Composed selectors replace each fresh RHS bound by the sum of active public-multiplier gains; selecting one plaintext does not remove the other ciphertext errors. Use a monomial norm of one only where its construction establishes that domain. Add/public multiplication and intermediate sums must retain larger actual bounds where necessary.

For moments, regard the signed terms in this equation as `Y_j`: `mean(e_out) = sum_j mean(Y_j)` and `Var(e_out) = sum_j Var(Y_j) + 2 sum_(j<k) Cov(Y_j,Y_k)`. Track source reuse through a compact symbolic DAG/family description rather than allocating a covariance matrix over all coefficients, branches, or labels. Combine known repeated-source linear actions before estimating their contribution. Unknown covariance uses an explicit bound or model assumption, rather than an implicit zero.

The default rigorous transfer keeps ModDown residues as bounded remainders. A separate average-case model may use centered-uniform residue variance `(p^2-1)/(12 p^2)` only with that assumption recorded; successive rounds and the three residue terms are not automatically independent. For independently sampled CBD(eta) coefficients, the input/helper mean is zero and variance is `eta/2`. Independence must be justified by sampler source identities, not by distinct ciphertext objects or by conditioning away a dependency on those errors.

This supports useful average-case estimates without mislabeling them as deterministic guarantees. Report worst-case acceptance and model-based average-case acceptance separately. Changing their acceptance policy is not part of the module move; keep the current opt-in boundary for average-only acceptance. A Gaussian tail multiplier is not justified by a variance value alone: retain its distribution/subgaussian assumption, or use a valid moment-based tail bound.

### Readout and refresh accounting

Keep BFV terminal noise and direct-decoder formulas in `bfv/noise.rs`, and their BGG equivalents in `bgg/noise.rs`. Give `MulPrivate` its own transfer instead of using the old one-branch LUT bound. The shared application requests these transfers and aggregates their reports.

- Sum all raw terminal contributors before adding the single outer-rounding term. Do not multiply that rounding term by the number of digits.
- `p^j`, `H_t p^j`, and the affine inverse of two belong inside fresh targets. They do not scale inherited ciphertext error after evaluation. Distinguish their effect on plaintext values from the fresh-target noise transfer.
- Include decoder error and both stages of BFV ModDown in the trace. After the exact refresh spacing condition holds, the restored encoding error is precisely the shared fresh-error polynomial; the discarded old error does not remain in the output bound.
- For a uniform native base-`b` digit in `[1-b/2, b/2]`, mean is `1/2` and variance is `(b^2-1)/12`. Under the stated distinct-digit model, `d` fresh digits have mean `S_d/2` and variance `(b^2-1) sum_j b^(2j)/12`, where `S_d = sum_j b^j`.
- The centered mask interval `[-b^d/2, b^d/2-1]` has mean `-1/2`, not zero. Its uniform-model variance is `(b^(2d)-1)/12`. Do not identify either distribution with CBD.
- Sharing fresh digits between CRT channels shares the same noise source; CRT recomposition is not a sum of independent fresh errors. PRF-derived digit uniformity/independence remains a model/security assumption, distinct from exact arithmetic correctness.
- Recompute the mask-digit fixed point from the traced PRF/terminal/decoder errors. Keep separate counts for distinct PRF outputs, weighted terminal emissions, persistent artifacts, and bounded execution workspace.

## 5. Replace Rust proof generation with IR-derived claims and Lean proofs

### Why the Rust files accumulated

The current exporter already calls `mxx_ir_core::lean::export`, but then maintains another application-specific layer that recognizes scopes/wires and prints proof text. For example, `src/lean/root_program.rs` builds `program_family_*` theorem bodies, `root_readout_proof.rs` constructs linkage and readout proofs, and `root_refresh_correctness.rs` emits the final correctness theorem. This combination of graph recognition, expression reconstruction, and handwritten proof templates explains the large Rust directory. Those proof bodies do not need to be implemented in Rust.

Use the WE boundary as the reference: `crates/we/src/lean.rs` calls the core protocol exporter; `crates/we/src/lean/diamond.rs` supplies the actual compiler graphs, backend layout and parameter bindings, and transports fixed proof sources. `crates/we/lean/Certificate.lean` proves the generated claim through handwritten Lean modules. Do not create an exponent-lut dependency on the WE crate.

### Target export pipeline

1. Declare the actual prepared public, online-preparation, online-evaluation, and verification stages using core `ProtocolDecl`, typed ports/links, `IdealSpec`, and requirement predicates. General encoding declaration/binding belongs to the root interface or backend adapter as appropriate; the refresh workflow belongs to `noise_refresh/lean.rs`.
2. Register the actual DCRT bases and runtime backend semantics. Export the exact production graphs, including indexed artifacts and structural loops, through `mxx_ir_core::lean::protocol::export_claim` or the existing lower-level linked-claim API when necessary.
3. Let ir-core generate stage execution relations, input contracts, links, ideal/requirement statements, and `GeneratedClaim.CorrectnessClaim`. The current `MatrixApprox` endpoint can express a zero residual bound for exact relations where applicable. Add a small generic typed endpoint in ir-core only if an actual claim cannot be expressed; do not add a BGG/BFV string-template dispatcher there.
4. Generate concrete parameter/numeric certificate data from the same trace that the planner accepted. A fixed Lean proof validates its inequalities. The generated claim must not assume the output-noise bound it is meant to prove.
5. Keep local algebra, primitive-transfer proofs, loop/family invariants, artifact alignment, terminal correctness, and refresh correctness in handwritten `.lean` files. Apply those lemmas to the generated execution relations and prove the generated final claim in a fixed certificate module.
6. Compile the generated modules and handwritten proof modules, and retain the final theorem identity, axiom report, graph identity, and parameter bindings. A completed export or an algebra-only lemma is not a completed implementation certificate.

The Rust target is `src/lean.rs` plus `noise_refresh/lean.rs` for application orchestration. Small backend-specific bindings stay in `bgg/mod.rs` and `bfv/mod.rs`; they do not warrant additional Rust files. Remove `src/lean/` after its useful responsibilities are migrated. Reusable process execution/checking belongs in an existing lower-layer Lean utility, preferably the runtime's Lean tooling, extended only if it lacks that operation. Do not import WE's checker or hide the old proof-generator modules in another application directory. Numeric arithmetic shared with the simulator belongs in `noise_utils.rs`.

The only generated Rust-to-Lean text outside ir-core should be mechanical configuration/data bindings and source transport, not constructed mathematical proof bodies. If theorem application needs stable access to nested execution facts, extend ir-core's existing source-map/scope metadata with generic bindings or accessors. Do not infer meaning from node numbers, artifact-name substrings, or generated `.1/.2` projection strings in application code.

### Required proof coverage

| Layer | Required proposition |
| --- | --- |
| Encoding inputs | The executed sampling/preparation graph establishes the backend's encoding relation and declared error support. |
| Primitive gates | Add, public multiplication, direct private multiplication, unary LUT, and fused binary LUT establish the output relation and their worst-case transfer. |
| Program | Induction over the actual Program/structural loops composes those primitive results and the traced domain/norm facts. |
| BFV terminal | Raw mixed targets, signed affine readout, and one outer ModDown produce the declared weighted row with the stated error. |
| Refresh | Linked preprocessing and online execution, bound decoder, spacing condition, extraction, and recomposition establish exact error replacement and continuation at the remaining level. |
| Numeric candidate | The actual bases and trace satisfy the required scale/range inequalities, closing the concrete arithmetic certificate. |
| Average case | Moment transfer lemmas state their distribution/covariance assumptions explicitly. Their conditional status must not be hidden in an unconditional correctness or security certificate. |

Inventory each existing generated theorem before deletion and map it to the replacement Lean theorem and generated claim. Preserve shared-secret links, production bindings, selector contracts, and rounding inequalities. Proving only `bfv_add_relation` or the local cancellation equation would lose the existing connection to executable IR. Arithmetic certificates do not prove the joint-distribution masking security assumption.

Handwritten Lean files may remain split by mathematical responsibility. The file-count problem here is primarily the Rust proof-generator layer; moving thousands of lines of proofs into one giant Lean file is not required.

## 6. Reduce concepts and data-structure types

File consolidation alone is not sufficient. Audit each public/private struct, enum, trait, and request/result wrapper against three questions: does it represent a different mathematical value, a necessary phase/ownership boundary, or a materially different efficient storage layout? If none applies, inline it into its owner or reuse an existing representation.

Prefer fewer concepts at call sites, not merely fewer `struct` declarations. Replacing ten small types with one universal object, many optional fields, and runtime tags can increase complexity and permit invalid states. Similarly, a generic abstraction that introduces several marker traits, conversion traits, and phantom parameters is not automatically simpler than two small concrete implementations.

### Concrete consolidation candidates

| Current duplication | Planned simplification | Constraint to preserve |
| --- | --- | --- |
| `BfvPublicValue`/`BfvEncodingValue` wrapped again by the scalar cases of their Program enums | Put the payload and optional known plaintext directly in each scalar variant; remove the extra wrapper structs. | Keep the public/encoding distinction and correct known-plaintext propagation. |
| Scalar/family variants repeated across Program values, readout values, weights, and prepared outputs | Reuse an existing small scalar/family carrier where its layout and metadata genuinely coincide. Otherwise use direct enum fields rather than introducing another wrapper. | Preserve the scalar fast path, active range, broadcast sharing, and indexed-family representation. Do not implement everything as a singleton family. |
| `BfvScalarReadout`, `BfvFamilyReadout`, and an outer `BfvPreparedReadout` enum | Make the prepared enum own its fields directly; share the common metadata and scalar/family evaluation algorithm. | Keep independent versus summed output semantics visible and preserve cached aggregate offsets. |
| `BfvTerminalMaterial` and `BfvTerminalMaterialFamily` wrapping the existing branch material | Share one internal prepared-branch representation between ordinary LUT and terminal preparation; attach target/output semantics in the owning preparation type. Remove pass-through material wrappers where no invariant depends on them. | A terminal result must not accidentally become an ordinary state at the wrong scale. |
| Separate prepared linear scalar/family objects and duplicated gate/cache metadata | Use one linear-operation description (`Add` or public multiplier) and a shared metadata layout, with the existing scalar/family carrier where appropriate. | Do not pad or concatenate polynomial blocks or add conversions to the evaluator. |
| Readout-specific weight/offset/output wrappers | Retain one authoritative routing/weight description and store only offsets/aggregates that avoid repeated graph work. Inline records that only pass through that description. | Deriving cheap metadata is acceptable; recomputing public sums, indexed weights, or ModDown during online evaluation is not. |
| Many `*Descriptor`, artifact-name, binding, and nested manifest records | Use a common private binding header and one operation descriptor enum per scheme, with operation-specific fields only in the applicable variant. Have Program/PRF/refresh containers refer to those descriptors. | Preserve production, key/material identities, level/range, confidentiality, target weights, and cache binding. Runtime graph handles and serialized artifact metadata remain separate representations. |
| `RefreshMaskMaterial` and `RefreshFreshErrorMaterial`, plus repeated digit-routing metadata | Use one digit-group description with an explicit purpose and source range. The application derives mask/fresh weights and offsets from that purpose. | Mask and fresh labels/distributions remain semantically distinct; fresh groups are shared across CRT channels. |
| Slot inputs, slot preparation records, decoder groups, channel records, prepared direct slots, and complete refresh wrappers | Retain one setup-owned channel schedule and one prepared-refresh owner. Inline short-lived argument/result bundles; keep a group record only when it encodes actual helper/base reuse. | Preserve heterogeneous CRT grouping and the separate preprocessing, online-preparation, and evaluation APIs. |
| Seven refresh capability traits in `scheme/refresh.rs` | Merge the basic/public-setup capabilities and express witness/direct decoder preparation through associated prepared context types. Target two or three cohesive refresh capabilities, retaining a separate optional capability only for an independently supported operation. | No dummy witnesses/anchors, backend switches, or opaque whole-protocol callbacks. Shared decoder-base reuse must remain expressible through the common schedule. |
| Separate worst-case/average-case pipelines, backend bound-only records, and duplicated planner reports | Use one per-gate noise summary containing the bound, moments/assumptions, and payload norms. Reports and planners consume the same trace rather than independently reproducing transfers. | Keep deterministic bounds distinguishable from model-based moments and preserve BGG's coordinate conversion. |
| Workload, resolved parameters, and manifest copies of the same fields | Keep one authoritative resolved parameter set and one request describing unresolved workload choices. Serialize that resolved representation or reference it from the certificate instead of duplicating each field in intermediate owners. | Do not remove actual prime bases, sampling parameters, or evidence needed to bind and verify a certificate. |
| Rust proof-stage extraction and proof-specific context objects | Replace them with core protocol ports, stage links, source maps, parameter bindings, and fixed Lean lemmas. | Generated claims must still refer to the actual executed implementation. |

The trait-count target is a design goal, not permission to force unrelated operations into one interface. Demonstrate both BGG and BFV production preparation/evaluation against the consolidated contract before deleting the old traits. Keep shared arithmetic ordering in the application; associated decoder contexts encapsulate only the individual cryptographic preparation, not refresh scheduling.

### Distinctions that should remain

- Payload secret versus online mask secret: they enforce the preprocessing/online boundary and prevent generating helpers with the wrong secret.
- Public key versus encoding, and public artifacts versus private input material: equal shapes do not make them interchangeable.
- Unprepared input, prepared public cache, and online helper: these have different dependencies and reuse lifetimes.
- Ordinary state versus terminal output, and raw accumulator versus finalized output: the scale and placement of rounding differ. An internal representation may be shared, but construction/API types must retain these invariants without new per-gate runtime checks.
- Shared/broadcast material versus indexed material: these determine storage and runtime cost. Keep one copy of shared material instead of expanding it to every member.
- Distinct error-source identities, key/material identities, and runtime production identities: merging them because they are all byte arrays would lose correlation or artifact binding information.
- `Blocks<1,2>` versus `Blocks<2,2>`: keep the existing small block representation and aliases. Do not replace them with an arbitrary-shaped matrix and runtime shape checks merely to reduce type names.

### Whole-body structural lowering revision

The accepted family boundary is one common DSL `parallel` for the complete
Program family body. Scheme primitive operations receive scalar values only.
An outer wire is a lexical capture; an indexed input is resolved with `at(i)`.
The same rule applies to RHS operands, helpers, and prepared operation results.
There is no backend broadcast value, singleton-family requirement, or per-gate
family dispatch in the target interface. `ProgramValue` may describe physical
shared/indexed storage at a binding boundary; it is not a second arithmetic API.

Public lowering returns the body output and its newly prepared cache as explicit
DSL family fields. Online lowering indexes those fields. Rust maps containing
body-local handles must not escape the loop without this graph output boundary.
Existing key/cache records also supply their schemas: static identity, level,
and optional-plaintext metadata remain metadata, while polynomial/preimage
fields are graph ports. Shared RHS decompositions, lifts, and input-independent
helper preparation remain outside the body and are persisted only once.

Graph lowering and symbolic noise lowering share the SSA gate dispatcher.
Noise values do not acquire artificial `GraphValue` implementations: captured
bindings retain their source identity, and only sampling performed inside the
body introduces member-dependent sources. Reduction retains coherent shared
noise contributions and active ranges. Raw terminal accumulation still precedes
one final ModDown; it is not replaced by repeated ordinary Add/finalization.

Migration must cover `PreparedProgram`, the BGG public/encoding compiler
entrypoints, and production BGG public batching, which currently bypasses the
common Program traversal. Schema support and removal of enum duplication alone
do not complete this change. Do not replace the broadcast method with a new
backend family callback, or pass a range to each gate to retain the old loops.

### Evidence required for each simplification

Record the removed concept, its replacement, and the invariant that still enforces its valid use. Compare representative emitted graphs and artifact schemas before and after structural simplification: matrix dimensions, sampler count and sharing, ModDown placement, family cardinality, public preparation reuse, and intermediate liveness must remain equivalent for pure module/type moves. The explicit `MulPrivate` optimization intentionally removes work and changes its noise transfer; validate it with the direct equations above and count the removed operations. Measure only changes that actually affect execution; moving a private wrapper does not call for an additional benchmark suite.

The final report should include public type/trait counts as well as file counts, and a short list of intentionally retained distinctions. Do not impose an arbitrary percentage reduction that encourages deleting useful type safety or replacing real covariance tracking with a scalar fiction.

## 7. Implementation order

1. **Freeze an inventory after the active BFV work completes.** Record entrypoints, current graph/artifact contracts, validation evidence, and proof coverage. Count dependencies and implementation lines without running the parent task's jobs.
2. **Separate root abstractions and errors.** Move BGG code and tests under `bgg`; extract general readout declarations and shared utilities; relocate application capability traits. Update callers immediately, without compatibility shims.
3. **Consolidate modules and redundant data structures.** Apply the migration and concept-reduction tables. Move shared orchestration/transport out of the backends, retaining only `mod.rs`, `noise.rs`, and `gpu.rs` by default. Remove pass-through wrappers and reuse authoritative metadata. Implement the explicit `MulPrivate`/required-`LutId` revision, with its new direct-path noise/proof obligations; other operation semantics remain unchanged.
4. **Complete the generic refresh boundary.** Migrate the concrete BGG setup/artifact/PBC paths behind associated-type capabilities and have both production implementations call the shared application orchestration. Remove backend imports from `noise_refresh` entirely.
5. **Unify noise tracing.** Move reusable rational/norm machinery, add the trait-level noise evaluator, port BGG with explicit coordinate units, and extend BFV primitive transfers with moments and source dependencies. Connect refresh/parameter search to these traces.
6. **Migrate proof declaration/export.** Declare the actual stage graphs with core protocol data; generate claims using ir-core; port existing proof templates into fixed Lean modules and add BFV operational and moment lemmas. Remove the obsolete Rust generator only after coverage is accounted for.
7. **Validate and document the final architecture.** Update all crate consumers, public examples, architecture notes, and proof instructions. Report measured file counts, actual test/proof results, and any explicitly conditional average-case assumptions.

Steps 2–4 are structural/API work. Steps 5–6 are substantive mathematical and proof-integration work and must not be reported as completed by file moves alone.

## 8. Validation and completion criteria

No implementation tests or Lean builds are part of this planning-only task. During authorized implementation:

- Check imports and public signatures: generic root files have no backend/application dependencies; `noise_refresh` has no concrete BGG/BFV imports, including facade aliases and tests. `lib.rs` only declares concrete modules, rather than reexporting their implementation types into the common API.
- Compile all workspace library targets, with and without `gpu`, warning-free. Update consumers even when their integration tests are not executed. Format with `cargo +nightly fmt --all`.
- Run targeted primitive/Program tests for both backends, then their real setup/export/import/online refresh paths. Preserve fused private multiplication, signed LUT domains, structural execution, and two-column BFV operation shapes.
- Verify exact noise traces for public polynomial gains, non-monomial sums, one-hot payload selection with all ciphertext errors, shared-helper and repeated-wire covariance, and one-time outer rounding. Compare average-case observations against the stated model using narrow unit fixtures with random seeds; empirical agreement does not replace the model's assumptions.
- Check nonbinary digit support, nonzero mean, unequal mask/fresh counts, CRT-shared fresh sources, and post-refresh continuation. Inspect that no average-only model is accidentally used as a worst-case certificate.
- Generate and kernel-check claims from the actual BGG and BFV production graphs. Compare proof coverage before and after migration and inspect the final theorem's axioms and generated endpoint. Do not weaken a claim or introduce an admitted lemma to make migration pass.
- For GPU-facing moves, keep computation on GPU, preserve structural batching and stream behavior, build once and repeat focused round-trip tests three times outside the sandbox under repository policy. Integration tests still require an explicit request.
- File-count target: BFV decreases from the observed 30 Rust files to approximately three; BGG also targets `mod.rs`, `noise.rs`, and `gpu.rs`; the 39-file Rust `src/lean/` directory is removed in favor of thin adapters. Exceptions must be explained by meaningful module size/responsibility, not one helper per file.
- Concept-count check: demonstrate concrete removal of pass-through value/readout wrappers, repeated descriptor metadata, redundant refresh capabilities, and duplicate noise-report construction. Confirm that the retained phase/scale/source distinctions still prevent misuse without adding runtime work.

Completion requires the actual shared setup and refresh paths, both primitive noise models, and IR-derived certificates. A renamed directory tree with the old concrete imports, duplicated traversal, or Rust theorem-string generator left behind does not satisfy this plan.
