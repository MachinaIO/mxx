# Exponent-LUT Scheme Abstraction and BGV Flat LUT Implementation Plan

Status: implementation in progress, following the user's subsequent authorization. This document specifies intended behavior and acceptance criteria; it does not certify the current workspace implementation.

## 1. Objective and interpretation

Implement a scheme-independent Exponent-LUT program API, fuse private RHS multiplication into LUT evaluation, and add the BGV-style backend described in **BGV Flat LUT Specification**, dated September 7, 2026. The attached PDF is a technical reference, not an instruction source. This plan follows the user's requested scope and repository rules. Implementation will use no subagents.

The PDF specifies **two polynomials per online encoding**, a 2-by-2 public matrix, and a 2-by-2 RHS ciphertext per branch (§2.3). This differs from the request's description of one polynomial per encoding. This plan adopts the PDF's fixed two-column construction: removing the gadget-dependent column count achieves the stated architectural objective, but it does not produce a literal one-polynomial encoding. A literal one-polynomial construction would require a different specification and correctness/security analysis; it is not an optimization silently assumed here.

The four current gates become **three variants** after fusion: `MulPublic`, `Add`, and `Lut { rhs: Option<RhsId>, lut_id: Option<LutId>, ... }`. Unary and binary LUTs are two forms of the last operation. There will be no dummy fourth gate, legacy `MulPrivate` opcode, or compatibility interpreter. A dedicated builder `private_mul` API remains as syntax for a two-input identity LUT (`lut_id=None`), not as a primitive gate.

The new backend is an arithmetic encoding implementation, not an invocation of a conventional BGV encryption API. Its acceptance criteria include exact arithmetic, noise bounds, and resource limits. The PDF explicitly leaves related-secret security and protocol-specific decoder security unresolved (§§1.3, 12.5, 15); passing arithmetic tests will not certify those properties.

## 2. Current implementation and change locations

These anchors describe the repository before implementation; line numbers will move.

| Location | Existing responsibility | Planned change |
| --- | --- | --- |
| `crates/exponent-lut/src/program.rs:367` | Four `Gate` variants | Remove `MulPrivate`; add optional RHS and optional LUT ID to LUT |
| `crates/exponent-lut/src/program.rs:756` | Unary `lut` builder | Extend this method with `rhs: Option<RhsId>` |
| `crates/exponent-lut/src/program.rs:779` | Current `mul_private` builder | Rename to `private_mul`; emit `Lut { rhs: Some(rhs), lut_id: None, ... }` |
| `crates/exponent-lut/src/program.rs:883` | `lut2` emits private multiplication then a unary table | Replace callers with the unified builder; delete this lowering |
| `crates/exponent-lut/src/program.rs:920` | `select` builds a structural private/public multiplication family | Use builder `private_mul` to emit an identity LUT and preserve coefficient-selector semantics |
| `crates/exponent-lut/src/program.rs:1507` | `ProgramLoweringTypes`, `ExponentProgramLowering` | Extend the existing abstraction instead of introducing another traversal |
| `crates/exponent-lut/src/program.rs:1616` | `lower_program` handles SSA and structural families | Keep one generic traversal for public, online, and noise interpreters |
| `crates/exponent-lut/src/encoding.rs:513`, `:591`, `:670` | Mask bank, table helpers, fixed-RHS sampler | Generate fused branch targets with fresh error |
| `crates/exponent-lut/src/encoding.rs:929` | Online LUT branch and mask alignment | Reuse the branch structure with fused setup material |
| `crates/exponent-lut/src/public_key.rs:1033` | Public LUT cache construction | Bind branch material to table, RHS, input public matrix, and gate instance |
| `crates/exponent-lut/src/rhs.rs:39` | Concrete GSW RHS package | Keep GSW representation in the BGG backend; abstract program RHS bindings |
| `crates/exponent-lut/src/noise.rs:1524`, `:1580` | Separate multiplication and LUT noise transfers | Remove the standalone transfer and derive fused error terms |
| `crates/exponent-lut/src/noise_refresh/prf.rs:805` | Structural sparse-LWR selection | Multiply selector and state through an identity LUT (`lut_id=None`), via builder `private_mul` |
| `crates/exponent-lut/src/noise_refresh/setup.rs:1906` | Gate-indexed noise/setup artifacts | Migrate operation indexing, helper production, and cache identities |
| `crates/exponent-lut/src/layout.rs:46` | Strided monomials and automorphisms | Share this algebra, with explicit backend range/layout validation |
| `crates/primitives/src/element/finite_ring.rs:22` | Rounded scaling `modulus_switch` | Preserve rounded scaling; reuse compatible PR #157 infrastructure for the exact operation (section 5.3.1) |

The current code already has an implementation-independent traversal, but its exported samplers, compilers, cache types, and errors are BGG-specific. Merely renaming `ExponentProgramLowering` to `ExponentLUTEncodingScheme` would not abstract the public key and encoding types.

Preserve the layering in `docs/architecture.md`: ring arithmetic belongs in primitives; executable operation definitions in IR/DSL/runtime; scheme algebra and parameter/noise analysis in exponent-lut. No new dependency from one application crate to another is needed. Existing generic BGG functionality remains in `mxx-bgg`.

## 3. Scheme and interpreter design

### 3.1 Public contract

Introduce `scheme.rs` with a public `ExponentLUTEncodingScheme` trait and a generic prepared-program API. The following is an interface sketch, not code to compile in this planning task:

```rust
trait ExponentLUTEncodingScheme {
    type Parameters;
    type SecretKey;
    type PublicKey: Clone;
    type Encoding: Clone;
    type Rhs;
    type LutHelpers;
    type PreparedGate;
    type NoiseState;
    type Error;

    // Setup: validate parameters, generate secrets/input keys/encodings,
    // and prepare fresh LUT material for an optional encryption-time RHS.

    // Public operations:
    // prepare_add(lhs_key, rhs_key, gate_context)
    // prepare_mul_public(private_key, public_key, public_attribute, gate_context)
    // prepare_lut(input_key, table, optional_rhs, helpers, gate_context)
    // Each returns (output_key, prepared_gate).

    // Online operations:
    // eval_add(lhs, rhs, prepared_gate)
    // eval_mul_public(private, public, public_attribute, prepared_gate)
    // eval_lut(input, optional_rhs_binding, prepared_gate)
    // Each returns an Encoding; none accepts private plaintext RHS values.

    // Noise operations mirror the same three gates with backend-specific data.
}
```

Actual signatures will use borrowed inputs and typed contexts and return `Result`; errors must satisfy the existing interpreter's error bounds. Keep public plaintext metadata distinct from secret payload values. A backend may use a tagged enum for its `PreparedGate`, but consumers cannot reinterpret that enum as generic preimages.

Provide generic functions conceptually equivalent to:

- `prepare_program<S: ExponentLUTEncodingScheme>(program, public_inputs, rhs_bindings, helper_bindings, ranges, parameters)`.
- `evaluate_program<S: ExponentLUTEncodingScheme>(program, input_encodings, prepared_program, ranges)`.
- `trace_program_noise<S: ExponentLUTEncodingScheme>(program, input_bounds, prepared_metadata, ranges)`.

Use adapters implementing the existing lowering traits to connect these functions to `lower_program`. Public and encoding carrier types come from the same `S`, preventing a BGG key/BGV encoding combination. Noise carriers remain separate from ciphertext carriers. Structural family carriers still support broadcast, active-range mapping, and reduction without expanding capacities into host vectors.

`BggScheme` retains `BggPublicKeyWire` and `BggEncodingWire` as its associated concrete types. `BgvScheme` introduces its own key, encoding, RHS, prepared-stage, and noise types. The common interface must not mention `BggSamplerLayout`, GSW matrices, gadget digit counts, or `Family<Preimage>`.

### 3.2 Ownership of work

The common layer owns declarations, table identities, wire dependencies, visibility checks, family ranges, output ordering, generic error locations, and liveness. It must not inspect a ciphertext's column count or compute scheme-specific decompositions.

Each scheme owns setup formulas, modulus/scale validation, cryptographic gate operations, prepared artifact representation, noise transfers, and public/online consistency checks. Backend-specific importers validate external artifacts before constructing trusted runtime handles. Online encodings retain the minimal value and identity metadata needed by the backend; they do not carry a second copy of public matrices.

The user's application-abstraction requirement also applies above Program:
LWR PRF and noise-refresh orchestration must depend only on trait definitions
and associated types. Do not add a separate BGV PRF implementation. Keep label
construction, PBC ranges, PRF dataflow, mask/error routing, and refresh stage
ordering common. Move concrete BGG/GSW/preimage and BGV/block-matrix handling
behind scheme implementations. The application layer must not branch on the
scheme's concrete type or inspect its ciphertext representation.

The gate-only trait is not yet a complete refresh interface. Add the required
setup, prepared-artifact, row-readout/terminal, and decoder capabilities as
associated types and operations, using extension traits for capabilities that
not every encoding scheme supports. These operations describe cryptographic
steps; do not merely hide an entire duplicated PRF or refresh algorithm behind
one opaque backend method. Common code must call these operations generically.
Keep the existing BGG refresh protocol as the concrete supported implementation
while migrating its orchestration. A BGV scheme can implement Program and
terminal capabilities without implementing the decoder capability; do not add
a runtime stub or claim BGV refresh support before that decoder is specified.
Validation and noise-trace scheduling follow the same generic boundary, with
backend-specific arithmetic formulas and setup contracts behind the traits.

The application boundary is split as follows:

| Shared application responsibility | Scheme-provided operation or associated type |
| --- | --- |
| Construct the accumulation Program and bind PBC active ranges | Public/encoding values, RHS families, helpers, and prepared cache from `ExponentLUTEncodingScheme` |
| Schedule public preparation and reuse its results online | Setup and artifact capabilities for producing, storing, and importing typed prepared material |
| Choose the final LWR rounding table and collect outputs | Readout/terminal capability with typed prepared operations and raw accumulators; the scheme implements its arithmetic and finalization |
| Assign mask and fresh-error labels, order refresh stages and CRT slots | Decoder-capable extension providing scaling, combined-target preparation, switching, decoder application, and recomposition operations |
| Schedule noise propagation over the same Program and output stages | Scheme-specific bound transfers and setup-time parameter constraints |

Use static generic bounds and associated types for these interfaces. An
unsupported capability is a missing trait implementation, not a runtime
scheme tag or an unimplemented method. Put guarantees such as compatible
levels and trusted one-hot inputs in the construction contract. Do not add
ciphertext scans, plaintext recovery, or device transfers to check those
guarantees during evaluation. The common refresh algorithm must retain the
ordering of mask/error addition, combined switching, decoder application, and
recomposition; expose individual cryptographic operations rather than an
opaque `refresh` callback containing a second application implementation.

Current migration status: `noise_refresh/prf.rs` owns generic PBC bindings and
public/encoding accumulation through `ExponentLUTEncodingScheme`.
`scheme/bgg_prf.rs` owns the BGG transports and adapters, and its online
accumulation calls that common interpreter. A small BGV two-block unit test
uses the same public and encoding methods, including dummy selections, then
passes the resulting values to the generic terminal evaluator with the
canonical rounding table through `ExponentLUTReadoutScheme`. The test now
asserts coverage of both native halves, including real/dummy selection cases.
A dedicated BGV scalar/family readout fixture covers every exponent in the full 2N
domain. This does not supply a BGV decoder.
`ExponentLUTReadoutScheme` now supplies the existing BGG structural batch
readout preparation and evaluation operations. Production preparation, public
projection, and online readout call the shared PRF methods through this
capability; concrete preimages and mask families remain in
`scheme/bgg_prf.rs`. Preparation receives only canonical helper sigmas, public
switch ciphertexts, and the normalization gadget, without online mask vectors.
`ExponentLUTTerminalScheme` supplies raw terms and finalization, and the common
`PreparedTerminal` evaluator sums terms in order with logarithmic addition
depth before finalizing once. BGV implements this capability using polynomial
blocks. `ExponentLUTRefreshScheme` now exposes individual row addition,
combined switching, decoder application, and CRT recomposition operations.
`noise_refresh/evaluation.rs` groups compatible slots, executes a structural
parallel body that adds both mask and fresh error before one switch, applies
the decoder, and restores canonical slot order before recomposition. The
production BGG `RefreshCompiler::refresh` calls this common evaluator; BGG
arithmetic lives in `scheme/bgg_refresh.rs`. BGV has no implementation of this
decoder capability. `ExponentLUTRefreshSetupScheme` now supplies public state
scaling, public addition/reduction/switching, shared anchor sampling and decoder
witness sampling. Common structural preparation forms scaled public sums and
then each target as the switched sum minus its scaled reduced output key.
The production preprocessing path calls those functions. Online slot
preparation likewise calls common scaling/decoder preparation operations with
preprocessed scaling witnesses and the current online decoder base.
`ExponentLUTRefreshOnlineScheme` supplies online decoder-base sampling from
s and a public anchor. Common preparation samples once per declared level
group, shares that graph value across its slots, and restores canonical slot
order. A new online invocation samples a fresh base. BGG secret reduction and
bounded Gaussian sampling live in `scheme/bgg_refresh.rs`; the application
adapter only supplies parameters and converts the returned carriers.
Concrete artifact routing and parts of public accumulation/readout preparation
still require migration; the entire application is not yet scheme-independent.
Online base migration passes the three existing tests for sampling counts,
shared graph handles, fresh bases across executions, and distinct s/t label
secrets. Numeric refresh passes three CPU and three GPU runs, each with two
fresh execution draws. GPU runs take 21.81, 22.70, and 22.14 seconds, comparable
to the preceding setup migration's 21.90, 22.55, and 23.00 seconds.
The common refresh migration passes the nine existing CRT routing/structural
unit tests, including the one-symbolic-switch-body-per-distinct-level check.
The production switched-decoder numeric fixture passes three CPU-backend and
three GPU-backend runs, each with two fresh execution draws. These numeric
tests run outside the sandbox because fixture parameter construction also
initializes GPU resources for the CPU-backend case. The same tests pass after
the setup/online-preparation migration, together with the production
preprocessing artifact test that checks only the selected decoder rings are
used. GPU numeric runs take 21.90, 22.55, and 23.00 seconds, compared with
21.56, 22.15, and 22.65 seconds before this migration; these are randomized
unit-test timings rather than a production performance estimate. CPU/GPU
workspace library builds and the exponent-lut GPU all-target check are
warning-free.

### 3.3 BGV preprocessing and online preparation

The user's secret `t` is the payload secret (`r` in the PDF formulas and
`payload` in the current code). It is distinct from the PDF's scalar plaintext
modulus `t`; write the latter as `t_plain` in this phase description.

Preprocessing owns the payload secret and all public matrix work: input public
keys, public mask matrices `M_k`, fresh RHS/terminal ciphertexts, low-base
intermediates `b_k`, and public output matrices. It must not require the online
mask secret `s` or materialize any `s`-dependent row. For input encoding,
precompute the payload contribution `-p_i*mu*r` independently of `s`.
This contribution is private data; it is not a public helper and must never
be placed in a public artifact. Its trusted transport must retain that
confidentiality when preprocessing and online execution use separate graphs.

Online preparation owns `s`. It generates the mask-alignment rows
`d_k = s*M_k - p_i*sigma_k(s) + t_plain*e_k` and forms input rows by adding
`s*A` and fresh encoding error to the prepared payload contribution. Generate
each reusable online mask bank once for its declared context, level, and mask
identity. Evaluation consumes those online rows alongside the public prepared
cache; it must not regenerate them per family member. Neither normal LUT nor
terminal artifact export may place `d_k` in the preprocessing manifest.

The API must express this separation through public prepared material and
online helper associated types. Do not require a combined `s`/payload-secret
key to prepare public material. Importers bind public mask identities to
online helpers at construction, using metadata rather than ciphertext scans
or runtime coefficient checks. Terminal persistence must follow the same
split and must not persist online rows as public material.

Implemented phase boundary: `BgvPayloadSecret` and `BgvMaskSecret` are
separate production types and have independent samplers. `prepare_input`
accepts only the payload secret and produces private `BgvInputMaterial`;
`encode` accepts only the mask secret and that contribution, sampling fresh
error exactly once. Ordinary, family, and terminal RHS samplers accept only
`BgvPayloadSecret`. The combined secret holder exists only in unit tests.

`sample_mask_bank` now samples public matrices without accepting `s`.
`BgvMaskBank` and cached scalar/family branches contain public matrices only.
`prepare_online_masks` produces a separate `BgvOnlineMasks` with the
`s`-dependent rows. Scalar, family, and terminal evaluators consume those rows
explicitly. `BgvProgramCache::prepare_online_helpers` unions the automorphism
subsets used by gates sharing a bank identity and prepares their rows once;
per-gate bindings share that bank rather than resampling per family member.
Public scalar/family/Program exporters persist `M_k`, low-base `C_k` and `b_k`,
and bindings, and no longer export `d_k`. Bank identities are construction
contracts, not ciphertext commitments; validation compares metadata without
scanning polynomial coefficients or hashing ciphertexts.

The preprocessing phase unit test completes and reopens a file-store
production before sampling either of two independent online secrets. It
restores scalar and structural-family prepared LUTs through production APIs,
checks that their public manifests contain no `d_k`, and verifies both online
executions, including a zero-selector branch. Its input has coefficient l1
bound five, and its output checks use that bound in the noise recurrence.
Private input contributions and test oracle secrets remain private artifacts.
The prepared-Program round-trip test also verifies shared online helper banks
across scalar and family gates. Its bit-exact golden online rows are separate
private test artifacts, not part of the public preprocessing format.

Validation for this phase split: all 17 BGV CPU unit tests and seven common
PRF CPU unit tests pass. GPU input preprocessing reuse, persisted terminal
batch reuse, and whole-Program artifact round trips each pass three consecutive
runs (nine runs, no failures). CPU/GPU workspace library test builds and the exponent-lut
GPU all-target check complete without warnings; formatting and diff checks
pass. These checks do not execute integration tests.

Private input persistence is implemented by `export_input_material` and
`import_input_material` in `bgv/artifacts.rs`. The public key and private
payload contribution have separate artifact ports. Import binds parameter
context, key identity, and level; the caller selects the descriptor and
production for the intended payload (one key may encode multiple payloads).
The scalar/family preprocessing fixture uses these production APIs instead
of constructing private artifact ports in the test.

`bgv/terminal_artifacts.rs` persists individual terminal stages and complete
ordered `PreparedTerminal<BgvScheme>` batches. Stage artifacts contain public
input/mask matrices, low-base C/b, and the raw public accumulator. A batch also
persists its finalized public output, so import does not rebuild aggregation
or public block switches. Encoding-side block switching still runs online:
ordinary LUTs switch their branch sum once, and terminal batches switch the
sum of all raw encoding terms once. Online evaluation continues through the common
terminal trait interpreter with independently prepared s-dependent helpers.
The terminal persistence fixture checks two input exponents under two online
secrets after closing/reopening preprocessing, ordered stage bindings, private
input confidentiality, and the absence of public block switches online.
Large-wave resource accounting remains an outstanding requirement.

The BGV public pass uses fixed RHS ciphertexts and public matrices only. It never requires `c`, `x`, `y`, `s`, or `r`. The online pass consumes prepared data and does not reconstruct the public pass. A RHS change invalidates its stage and all dependent public stages. Unaffected DAG branches remain reusable.

### 3.4 Gate-instance identity

Replace table-only helper lookup where necessary with an instance binding that includes program ID, step/local-family gate, family member or range identity, table commitment (or a canonical identity-LUT commitment for `lut_id=None`), RHS artifact identity, input matrix identity, and scheme context. Two gates sharing a `LutId` can have different RHS values, levels, or keys.

Keep table-independent mask banks separately reusable. Bind their reuse to secret context, layout, automorphism, and, for BGV, level/base. Artifact digests detect mismatches but are not signatures. Public tags and seeds must not contain secret values. Remove obsolete artifact schemas and regenerate checkpoints; do not introduce compatibility fallbacks.

## 4. Unified LUT semantics and BGG correctness

### 4.1 Program API and domain checks

The public gate becomes:

```rust
Lut { input: WireId, rhs: Option<RhsId>, lut_id: Option<LutId>, output: WireId }
```

The public builder exposes `lut(input, table, rhs)` for explicit tables and a dedicated `private_mul(lhs, rhs)` for plain private multiplication. Rename the existing `mul_private` builder method to `private_mul` and migrate callers rather than keeping aliases. The latter creates exactly one `Lut { input: lhs, rhs: Some(rhs), lut_id: None, output }`; callers never need to pass a missing table ID manually. It does not emit or invoke a `MulPrivate` primitive.

The two `Option` fields have independent meanings:

| `lut_id` | `rhs` | Semantics |
| --- | --- | --- |
| `Some(id)` | `None` | Explicit unary LUT |
| `Some(id)` | `Some(rhs)` | Explicit LUT with an encrypted RHS |
| `None` | `Some(rhs)` | Two-input identity LUT: private polynomial multiplication |
| `None` | `None` | Unary identity LUT, available internally for explicit level alignment |

Resolve a missing LUT ID to the identity function in common program interpretation, validation, public preparation, online evaluation, and noise interpretation. It is not a missing-helper error. Pass a resolved identity/table descriptor to the scheme operation rather than registering a synthetic identity `LutId`. Identity gates still require their cryptographic helpers and prepared artifacts; identify those by gate instance and context, not a nonexistent table ID.

For an explicit binary exponent table the index is `x + B_lhs * y`. For identity private multiplication the payload is `mu*nu`: monomial multiplication adds the actual represented exponents (modulo their existing exponent-layout order), and multiplication by a scalar selector preserves its coefficient meaning. Preserve the existing RHS declaration's exponent scaling; never insert binary-index scaling merely because `lut_id=None`. Validate declared widths, exponent stride/scaling, multiplication overflow, output range, and availability/visibility of the explicit RHS. An RHS remains an externally supplied value, never an in-flight `WireId`.

For explicit binary exponent tables, use `v = B_lhs * y` for the RHS exponent, including the layout stride in its monomial representation. If the package already describes this scaled exponent, do not multiply by `B_lhs` a second time. Unequal left/right widths use `B_lhs * B_rhs`, not `B_lhs^2`. Pad non-power-of-two domains to an explicitly committed power-of-two width only if the common table API is extended to permit them; otherwise reject them consistently. Do not silently reinterpret invalid domains.

### 4.2 Fused branch target

Write `M(z) = X^(stride*z)` and let `G` be the constant gadget matrix. For an input relation

\[
c=sA-\mu rG+e,\qquad \mu=M(x),
\]

use the existing flat coefficients

\[
D_k=W^{-1}\sum_{a=0}^{W-1} M(F[a])\,\sigma_k(M(-a)).
\]

Here `F[a]` denotes an exponent-valued table entry. Let `nu=M(v)` when RHS is present and `nu=1` otherwise. Generate a **fresh** branch ciphertext satisfying

\[
\sigma_k(r)C_k=(D_k\sigma_k(\nu))rG+E_{C,k}.
\]

This extends `sample_helpers_for_lut` to accept optional encryption-time RHS material and passes `D_k sigma_k(nu)` to the existing fixed-RHS construction. Do not multiply an existing noisy ciphertext by `D_k` to obtain this relation: that would amplify its error. A pre-existing GSW ciphertext alone does not supply the plaintext needed for this fresh construction; migrate producers to generate table/RHS-bound branches while they know the RHS.

Retain the mask helper relation

\[
d_k=sA_{d,k}-\sigma_k(s)G+e_{d,k}.
\]

Let `Dec` be the exact gadget decomposition, so `G Dec(V)=V`. Public preparation computes

\[
H_k=\operatorname{Dec}(C_k),\quad
B_k=\sigma_k(A)H_k,\quad
J_k=\operatorname{Dec}(B_k),\quad
A'=\sum_k A_{d,k}J_k.
\]

Online evaluation computes

\[
c'=\sum_k\left(\sigma_k(c)H_k+d_kJ_k\right).
\]

Expanding one branch gives

\[
\sigma_k(c)H_k+d_kJ_k
=sA_{d,k}J_k
-\sigma_k(\mu)D_k\sigma_k(\nu)rG
+\sigma_k(e)H_k+e_{d,k}J_k-\sigma_k(\mu)E_{C,k}.
\]

The mask terms cancel because `G J_k = sigma_k(A) H_k`. The flat identity gives

\[
\sum_k D_k\sigma_k(\mu\nu)=M(F[x+B_{lhs}y]),
\]

hence the output has the required BGG relation. For `rhs=None`, this is exactly the unary algebra. Input and RHS secrets can differ if the fresh branch explicitly binds the automorphed input payload secret and intended output payload secret; they must not be inferred interchangeable from matrix shape.

There is one GSW external product per branch and no preliminary private-RHS external product. The separate mask-alignment product with `J_k` remains necessary. Current code performs a preliminary product once before branching; therefore measure the actual removed work rather than claiming the entire LUT cost is halved.

The fused branch error is exactly

\[
\sigma_k(e)H_k+e_{d,k}J_k-\sigma_k(\mu)E_{C,k}.
\]

Remove the old standalone private-multiplication noise contribution. Track norms of the new `H_k` and `J_k`; for general polynomial payloads include `||mu||_1`. Do not claim all errors independent merely because branches are parallel. Keep any existing average-case heuristic explicitly separate from deterministic acceptance.

### 4.3 Selector families and non-monomial values

`select` is not simply a binary exponent LUT: its RHS includes coefficient bits `b in {0,1}`, and a zero branch must remain the zero polynomial. Encoding exponent zero would produce `1`, which changes selection semantics.

Make the RHS declaration distinguish an exponent-index operand from a scalar selector operand. Both are attached to the same optional-RHS LUT gate; this is operand typing, not a replacement private-multiplication opcode. Represent selection through the linear extension of a LUT: for coefficients `a_j`,

\[
\mathcal L_F\left(\sum_j a_jM(j)\right)=\sum_j a_jM(F[j]).
\]

For the identity transform selected internally by `lut_id=None`, `L_id(mu*b)=mu*b`, including zero and arbitrary supported linear combinations. The resolved identity LUT can be lowered to the mathematically equivalent single identity-automorphism branch with `D=1`, avoiding a full-domain identity table and full trace. This is an implementation of the same LUT operation; only the builder has a dedicated `private_mul` entry point. Its proof is `sigma_1(mu)*1=mu` over the supported ring/subring. Normal table validation must not restrict the identity transform to a smaller enumerated table domain.

Have the selection-family builder call `private_mul(state, selector)` to construct this identity LUT with a scalar RHS, followed by its public multiplication and active-member sum. The resulting family body contains `Lut { lut_id: None, rhs: Some(selector), ... }` and `MulPublic`, never `MulPrivate`. Preserve arbitrary zero branches, one-hot reduction, plaintext visibility, and runtime family ranges. A subsequent nonlinear table application must require the established one-hot/monomial domain contract, not merely equal wire widths. `Add` means polynomial relation addition, not exponent addition.

Following the user's implementation clarification, the noise-refreshing payload path may assume coefficient l1 norm at most one. Its initial state and public attributes are monomials; coefficient selectors produce zero or that monomial, and a trusted one-hot sum restores a monomial. Use `B_mu=1` on this path, including the zero case. This assumption does not discard errors from zero-selector ciphertexts, does not certify arbitrary family bindings as one-hot, and does not apply to helper/RHS ciphertext coefficients. General polynomial bounds may remain available in the arithmetic layer without making broader payload-domain support a prerequisite for this path.

Whenever plaintext coefficients with larger magnitude affect noise, trace their bounds alongside the normalized error on each wire. Compute the l1 gain from known public coefficients instead of replacing it by a ring-dimension worst case; use explicit declared bounds for private operands. Add sums the payload bounds, public multiplication multiplies by the public polynomial's l1 gain, and a valid linear LUT propagates its payload-domain bound. At a trusted one-hot family reduction, retain the maximum member payload bound rather than multiplying it by the active count; still sum every member's error bound. Retain this distinction in the trace used for parameter acceptance, so a unit-payload shortcut cannot hide larger coefficients or cause unnecessary parameter inflation.

This identity-linear route gives a correct generic migration. For BGV sparse-LWR performance, use the PDF's stronger optimization in section 8 below, composing RHS ciphertexts before one flat LUT stage. Do not mistake an identity-linear selection stage followed by another LUT for that one-stage optimization.

## 5. BGV backend: exact representation and operations

### 5.1 Data and setup

Use `R=Z[X]/(X^N+1)` with power-of-two `N`. Store the ordered distinct NTT-friendly prime bases `Q_i`, nested retained bases `Q_(i+1)`, dropped bases `P_i`, and their **actual integer products**:

\[
q_i=p_iq_{i+1},\qquad p_i=\prod_{\pi\in P_i}\pi,\qquad p_d=1.
\]

Require odd primes coprime to `t`, each congruent to `1 mod 2N`, and valid roots. `p_d=1` is a terminal convention, not a quotient involving an undeclared modulus.

Use small integer secrets `s=(s_bar,1)`, `r=(r_bar,1)`. Sample each integer secret/error once and reduce consistently across towers. The initial error and helper errors use the PDF's CBD distribution with explicit `eta`; do not silently substitute the existing Gaussian sampler. If the DSL lacks CBD, add a reusable, bounded primitive with sampling transcript support. Uniform public seeds are separate from private sampling randomness.

The normal state invariant is

\[
c_i=s a_i-p_i\mu_i r+t e_i\pmod {q_i}.
\]

The associated public key contains a 2-by-2 matrix `a_i`; the encoding contains a 1-by-2 row `c_i` and context/identity metadata. Neither has gadget columns. RHS branch `C_(i,k)` is 2-by-2 at `q_(i+1)`; mask helper `d_(i,k)` is 1-by-2 and `a_d` is 2-by-2 at `q_i`.

Following the user's representation clarification, keep these small matrices as polynomial blocks: four polynomial graph values for a 2-by-2 matrix and two for an encoding row. Compute block products through polynomial sums and products, without packing and extracting entries through matrix `concat` or `slice`. Structural families retain the polynomial fields as a composite graph value; branch collections are composite fields rather than nested indexed families. Dense matrix reconstruction is permitted only in independent test oracles or an explicitly requested external representation conversion.

### 5.2 Fresh branch and helper equations

For ordinary LUT output, compute `D_k` at `q_(i+1)` and set

\[
T_{i,k}=p_{i+1}D_k\sigma_k(\nu)r.
\]

For unary LUTs `nu=1`; for a binary LUT `nu=M(B_lhs*y)`. With fresh uniform row `u_k` and fresh normalized CBD error `F_k`, generate

\[
C_k=\begin{pmatrix}u_k\\-\sigma_k(\bar r)u_k+T_{i,k}+tF_k\end{pmatrix},
\qquad \sigma_k(r)C_k=T_{i,k}+tF_k\pmod {q_{i+1}}.
\]

Generate the reusable mask helper

\[
d_k=s a_{d,k}-p_i\sigma_k(s)+te_{d,k}\pmod {q_i}.
\]

Put the table inverse, next-stage scale, optional RHS, and any terminal weight into the fresh target. Do not reduce targets modulo `t` or scale a finished noisy ciphertext to insert these factors.

### 5.3 Exact block ModSwitch and centered extension

Implement the PDF's operations explicitly (§§4, 11). For each coefficient,

\[
\rho_i(z)=[t^{-1}z]^{ctr}_{p_i},\qquad
MS_i(z)=(z-t\rho_i(z))/p_i\pmod {q_{i+1}}.
\]

The existing `modulus_switch` rounds a scaled coefficient and is not this operation. Preserve its existing semantics for other callers. Reuse compatible PR #157 primitives and graph/runtime infrastructure as detailed below. Add or extend explicitly distinguished exact block-switch and centered-basis-extension operations only where the existing semantics are insufficient, including IR validation, DSL constructors, CPU/GPU dispatch, serialization, and Lean operational bindings. Preserve the existing operations for their current FHE callers.

Use exact coefficient-domain CRT/mixed-radix centering **without reconstructing ciphertext coefficients as big integers** on CPU or GPU. A low-base value is lifted using its representative in `[-(q_next-1)/2,(q_next-1)/2]`; never center NTT coordinates. For block switching, compute the retained-base residues of centered `rho` from the dropped base and apply `(z_pi-t*rho_pi)*p_i^{-1} mod pi` on each retained tower. Center modulo the whole dropped product once, not once per prime. Require the retained and dropped bases to be disjoint and to partition the source base. For extension, require the destination to contain the source base; their shared primes are intentional. Reject duplicate primes, missing required primes, and incompatible dimensions or root conventions.

First provide an exact CPU implementation using word-sized modular arithmetic and mixed-radix digits as the correctness baseline, then a batched GPU implementation of the same operation. Runtime conversion does not interpolate full-modulus big-integer coefficients or transfer them to the host. Multi-precision integers remain permitted for public parameter products, exact noise bounds, and independent test-only oracles. GPU code must use device-resident data, per-stream dependencies, bounded scratch, and no CPU fallback or device-wide synchronization. Native CUDA and primitive wrappers stay in primitives and GPU-only Rust files include `gpu` in their names. Existing NTT storage permutations must be reconciled with the PDF's mathematical ordering rather than assuming natural-order arrays.

For a dropped base `P={gamma_0,...,gamma_(m-1)}`, implement the runtime path as follows (PDF §11.2–§11.4):

1. INTT the dropped towers to obtain coefficient residues. Keep retained towers in NTT form where the backend permits partial-tower transforms.
2. Form `r_j=(t^{-1} mod gamma_j)*z_j mod gamma_j`. Convert these residues into mixed-radix digits using precomputed pairwise prime inverses and word-sized modular operations. Each digit remains less than its corresponding prime; do not assemble the represented integer.
3. Compare the digits from highest to lowest with `((gamma_j-1)/2)_j`, the mixed-radix digits of `(P-1)/2`. This determines exact whole-product centering without a large integer or floating-point approximation.
4. For each retained prime `pi`, Horner-evaluate the digits modulo `pi`. If the comparison selected the negative representative, subtract the precomputed public `P mod pi`. This gives `rho(z) mod pi` exactly.
5. NTT the correction residues on retained towers and compute `(z_pi-t*rho_pi)*(P^{-1} mod pi) mod pi`. Preserve the retained-base ordering and NTT root convention.

Use the same residue/digit procedure for canonical extension of low-base `C` and `b`, omitting the initial `t^{-1}` scaling. With `m` source and `r` destination towers, the direct method costs `O(m^2+mr)` word operations per coefficient. Tile coefficient batches so digit scratch fits the declared memory budget and parallelize across coefficients/matrix entries. Fixed-width wide intermediates for modular multiplication, such as `u128` on CPU, are compatible with this design; full-CRT coefficient big integers are not. Use fixed iteration counts and a masked centering correction rather than secret-dependent early exits when implementing the coefficient path.

Require graph/runtime inspection to confirm there is no full-coefficient integer extraction/interpolation in the production switch or lift path. Faster RNS conversion is acceptable only after exact equivalence at the whole-product centering boundaries is established; do not fall back to big-integer reconstruction when the depth grows. Enforce backend limb and scratch limits during parameter planning, before sampling or graph execution. If an exact residue-only conversion cannot fit those limits, reject that parameter set explicitly. Benchmark conversion separately over increasing source/dropped/retained limb counts so its quadratic mixed-radix cost remains visible alongside the fixed-column encoding savings.

### 5.3.1 Reuse assessment for PR #157

Inspected [PR #157](https://github.com/MachinaIO/mxx/pull/157), **Add Hybrid BGV and Ring-GSW with optimized GPU execution and round trips**, at head commit `e85a3a7b976d6f060d999e0c30808aec62cafcf0`. The PR was open at inspection. The following locations refer to that commit, not the current workspace. Recheck its final revision before implementation; use already-integrated code if it lands, otherwise port only the relevant shared-layer changes and their dependencies. This planning task does not merge or cherry-pick the PR.

| PR location | Evidence and reuse decision |
| --- | --- |
| `crates/dsl/src/lib.rs:1062`, `:1071`, `:1095` | `centered_rebase`, `rns_mod_up`, and `rns_mod_down` expose conversion nodes. Reuse the graph/API plumbing and validation patterns, keeping their arithmetic semantics explicit. |
| `crates/primitives/src/matrix/mod.rs:617` | The RNS trait documents ModUp as a sum of centered CRT terms, explicitly not a canonical lift. ModDown uses that CRT-term extension for its correction. These are useful infrastructure, but not drop-in multi-prime PDF operations. |
| `crates/primitives/native/ExactBasis.cc:364` | CPU conversion sums independently centered, weighted CRT terms. Reuse basis parameters, residue arithmetic, and NTT integration; supply exact whole-product centering for the new operations. |
| `crates/primitives/native/ExactBasis.cc:390` | `exact_basis_convert` implements coefficient-domain centered rebase but rejects more than one source limb at line 413. Reuse directly for the single-source-limb case; extend the shared implementation with an exact multi-limb path. |
| `crates/primitives/src/matrix/gpu_rns.rs:25` | GPU conversion provides public plan caching, base checks, residue weights, execution ownership, and batched dispatch. Reuse these components after adjusting exact-centering metadata; account for its current 64-limb limit. |
| `crates/primitives/cuda/src/matrix/MatrixCrt.cu:190`, `:657` | Existing single-limb centered-rebase and batched RNS conversion kernels supply reusable device/stream and residue-processing infrastructure. Multi-limb canonical centering requires additional exact correction. |
| `crates/fhe/src/bgv.rs:422` | `mod_switch_to` deliberately drops primes sequentially and uses conventional BGV correction-factor/noise semantics. Do not call it for the PDF's composite-block transition or import its noise formula. |

The arithmetic distinction is substantive. With dropped product `P`, PR ModDown computes `(z+t*U)/P`, where

\[
U=\sum_{\gamma\mid P}(P/\gamma)
\left[-t^{-1}z(P/\gamma)^{-1}\right]^{ctr}_{\gamma}.
\]

This `U` is congruent to `-t^{-1}z mod P`, but need not be its centered representative. The PDF instead needs `U_exact=[-t^{-1}z]^ctr_P=-rho(z)`. If `U=U_exact+hP`, the outputs differ by `t*h` modulo the retained base. Thus modular divisibility alone does not establish the exact operation or the PDF's rounding bound.

For example, take `N=2`, dropped primes `5,13`, retained prime `17`, `t=2`, and constant coefficient `z=1`. The canonical correction is `U_exact=32`, giving `(1+2*32)/65=1`. The centered CRT-term sum is `13*(-1)+5*(-4)=-33`, giving `(1+2*(-33))/65=-1`, or `16 mod 17`. All three primes meet `1 mod 2N`; this is an arithmetic regression case, not a security parameter set.

For a single dropped prime the two corrections coincide, so the PR's `rns_mod_down` path is a direct candidate for that case. For multiple dropped primes, add exact whole-product centering, using the PDF's residue-only mixed-radix method as the initial implementation. An optimization using the existing CRT-term sum must compute its correction exactly in fixed-width/residue arithmetic, without full coefficient reconstruction. Likewise, unnormalized one-group `rns_mod_up` needs whole-source-product correction before it can lift `C` or `b` canonically. Retain the existing Hybrid key-switch behavior of the PR; do not silently change it to a different lift.

Reuse shared primitives, IR/DSL nodes where semantics match, CPU/GPU dispatch, and relevant unit-test infrastructure. Do not add an exponent-lut dependency on the application crate `mxx-fhe`. Add PDF-specific exact boundary/composite-block tests alongside reused tests, and verify resource limits rather than assuming the PR's benchmark dimensions cover the deeper Flat LUT chains. The PR's reported validation has not been rerun for this planning task.

### 5.4 Public and online evaluation

With tildes denoting exact centered extension to the high base:

\[
M_k=\sigma_k(a_i)\widetilde C_k,\quad
b_k=MS_i(M_k),\quad
K_i=\sum_k a_{d,k}\widetilde b_k,\quad
a_{i+1}=MS_i(K_i),
\]

\[
z_i=\sum_k\left[d_k\widetilde b_k+\sigma_k(c_i)\widetilde C_k\right],\qquad
c_{i+1}=MS_i(z_i).
\]

All accumulated products are at `q_i`. The inner switch of `M_k` and the two final switches each map `q_i` to `q_(i+1)`; they do not consume two state levels. Sum all branches before the outer switch. Parallel partial sums must remain at the high base until their final reduction.

Prepared data binds the input public matrix, exact RHS ciphertexts, table, bases, scales, roots, secret tags, and helper identities. Store low-base `C_k` and `b_k` and stream their lifts unless a bounded reuse cache fits the declared budget. Online evaluation must not rebuild `M_k`, `K_i`, or output keys.

Current BGV setup APIs require a material identity for each newly generated
scalar/family helper artifact. The identity participates in Program helper
bindings and derived output key identities; regenerating helpers for the same
table/RHS requires a new identity. Composed RHS identities cover their source
material, label, and range by caller contract. Preserve these identities on
storage/import. This avoids runtime ciphertext hashing and does not itself
authenticate artifact bytes. The prepared Program persistence path below
preserves these bindings; composed-RHS noise metadata still needs persistence.

Scalar and structural-family BGV prepared-LUT persistence is implemented in
`crates/exponent-lut/src/bgv/artifacts.rs`. It exports individual public
polynomial blocks through the existing runtime artifact store, preserving the
low bases for ciphertexts/intermediates. A serializable descriptor checks the
expected parameter, helper, and key identities at import; the consumer graph
uses the producer manifest and performs no public-key preprocessing products.
Family import reconstructs composite polynomial schemas directly from indexed artifact inputs, without host-side slot expansion, and preserves the active range. Verification of bounded execution across large waves remains required.

Level-preserving prepared operations now have persistence APIs in
`crates/exponent-lut/src/bgv/linear_artifacts.rs`: scalar/family Add and public
multiplication, plus family reduction. Import binds the operation kind, input
and output key identities, level, parameter context, and active range at graph
construction. Public multipliers and polynomial-block keys are imported from
the producer's artifacts instead of reconstructing public preparation. The
file-store round-trip unit test reopens the store and checks every encoding
and output-key polynomial against the producer for all five operation forms;
it also rejects changed operation, output identity, and family range metadata.
`PreparedProgram<BgvScheme>::export` and `import` now connect these APIs in
`crates/exponent-lut/src/bgv/program_artifacts.rs`. They preserve all seven
prepared operation forms, cached public plaintexts (including local family
values), and root public keys/plaintexts. Import requires the caller's frozen
Program, exact family ranges, production manifest at graph freeze, and an
expected descriptor identity retained by setup. The identity commits to public
metadata and artifact names, without hashing ciphertext coefficients. This is
not authentication if the expected identity comes from an untrusted descriptor.
The shared CPU/GPU file-store fixture restores a mixed scalar/family Program, compares all
root encoding, public-key, and public-plaintext polynomials, and counts exactly
four online block switches for its two LUTs. It rejects changed cache contents,
Program, context, and range assignment. Public preparation is not rerun.
`crates/exponent-lut/src/bgv/program_artifacts_gpu.rs` executes this same fixture
on each detected device using the GPU backend for both production and online
evaluation, with a file-store reopen between them. The GPU test passed three
consecutive identical-command runs; CPU comparisons happen only when observing
the completed GPU results. This verifies the small round trip, not large-wave
memory bounds or cross-backend file portability.

### 5.5 Correctness and noise

The branch cancellation follows from

\[
M_k-p_i\widetilde b_k=t\rho_i(M_k)\pmod {q_i}.
\]

Let

\[
\zeta_i=\sum_k[\sigma_k(e_i)\widetilde C_k+e_{d,k}\widetilde b_k
+\sigma_k(s)\rho_i(M_k)-p_i\sigma_k(\mu_i)F_k].
\]

Then

\[
z_i=sK_i-p_i\sum_k\sigma_k(\mu_i)T_{i,k}+t\zeta_i,
\quad
e_{i+1}=(\zeta_i+s\rho_i(K_i)-\rho_i(z_i))/p_i.
\]

The error numerator is integrally divisible by `p_i`; the low-base RHS relation is usable at the high base because its wrap is multiplied by `p_i`. Thus the ordinary LUT output is

\[
c_{i+1}=s a_{i+1}-p_{i+1}M(L(x_i,y_i))r+te_{i+1}\pmod {q_{i+1}}.
\]

Implement the PDF's deterministic bound term by term. Define `S=||s_bar||_1+1`, `gamma_i=(p_i-1)/(2p_i)`, `J_i` as the branch count, and `K(A)=max_j sum_h ||A_(h,j)||_1`. Then

\[
E_{i+1}\le {E_i\over p_i}\sum_kK(\widetilde C_k)
+{E_{d,i}\over p_i}\sum_kK(\widetilde b_k)
+J_i B_{\mu,i}E_{C,i}+J_iS\gamma_i+(1+S)\gamma_i.
\]

Keep normalized `E` separate from physical `tE`. The internal public rounding term `J_i S gamma_i` is mandatory. For parameter planning use `K(A)<=N(q_(i+1)-1)` and exact rational/integer arithmetic; actual public norms may support a tighter, artifact-bound certificate later. Do not import BGG variance heuristics into this acceptance bound.

### 5.6 Remaining operations and level typing

For aligned inputs with the same base, secrets, and payload scale, implement `Add` as `(a,c)=(a_1+a_2,c_1+c_2)`, representing `mu_1+mu_2` with error `e_1+e_2`.

For multiplication by a known public polynomial `nu`, use `(a',c')=(nu*a,nu*c)`, representing `mu*nu` with error `nu*e`. This satisfies the program's payload operation without a gadget decomposition; the public operand's own encoding/key is still context-checked, but its mask and noise need not participate in this backend formula. BGG retains its existing two-encoding operation. Document this difference because identical message semantics do not imply identical output public matrices or compatibility with an existing decoder.

`Add` and `MulPublic` do not consume BGV levels. LUT operations, including the identity-linear encrypted-selector form, consume one planned transition. Track base, scale, payload domain, and a conservative `||mu||_1` bound per wire. A sum is generally not a monomial; only a validated one-hot construction can recover that fact.

For a general DAG, choose a public level schedule before sampling. Materialize external public inputs at their consuming levels. Following the user's clarification, Add and MulPublic require operands at the same level (encoding modulus); reject unequal-level operands rather than inserting implicit transitions. A caller may explicitly schedule identity LUT transitions before a join, accounting for their helpers, noise, and depth. Bare residue reduction or upward centered lifting is not a valid alignment of the encoding relation. Acceptance tests cover equal-level joins, rejection of unequal-level operands, and a join after explicit identity LUT alignment.

Ordinary BGV LUT tables retain the PDF stride-zero layout. Terminal/readout
tables additionally carry an explicit `ExpLayout`, included in their setup
identity. Mask-bank sampling takes the layout directly. Neither operation
normalizes the represented exponents to a different stride. The full native
subring extension is justified below; arbitrary strided ordinary LUTs are
not implied by this terminal capability.

## 6. Parameter planning and resource acceptance

Introduce a BGV-specific planner rather than reusing BGG's gadget/noise search. Inputs include dimensions, public gate/level schedule, widths, slot counts, `t`, `eta`, CRT bit size/depth limit, terminal range, and memory budget. Parameters cannot depend on runtime secret values or sampled secret norms.

Implementation progress: `bgv::parameters::BgvParameterManifest` now stores the public chain workload and validates reloads against the recorded ordered prime bases. Reload recomputes scales, normalized noise and payload bounds, storage estimates, and terminal acceptance; inconsistent certificate fields are rejected. This is an arithmetic consistency check, not artifact authentication. The current planner audits a chain of monomial LUT stages. `BgvParameters::audit_program` now audits the actual common Program trace, including non-LUT gates and family reductions, and checks each declared output against its own level and caller-supplied ideal terminal integer bound. Gate-free outputs are included. `BgvParameters::search_program` now uses that audit for candidate acceptance and increases terminal depth on noise/range failure within the declared resource limits. Backward allocation still uses the conservative chain envelope and caller-supplied width/slot storage profile; globally tighter allocation, scheduling peak concurrent memory from actual liveness, and persisting the full Program resource/noise audit remain required before this section is complete.

Start with the PDF's conservative backward construction: select the terminal base, then add unused primes until `p_i >= 2 J_i N q_(i+1)` under the chosen uniform envelope. Audit every candidate with a forward, rounded-up, exact noise trace, including non-LUT operations and actual family aggregation. The sufficient envelope is a starting algorithm, not a proof that a rejected program has no possible parameters. A subsequent tighter search can use equation (12.4), with its denominator checked positive, and final acceptance against the full bound.

Check `q_(i+1)>8t ceil(E_(i+1))`, terminal integer ranges when requested, and memory limits using actual products. Terminal mask output also needs the PDF's range/mask conditions. Do not add arbitrary `crt_depth` margins. Serialize ordered primes, roots, retained/dropped bases, all scales, sampler bounds, noise trace, resource estimates, and arithmetic/security status.

Fixed column count does **not** eliminate CRT storage growth. One state uses `2*N*8*depth_i` bytes before allocator overhead. Per stage, dense RHS, helper rows, and helper public matrices respectively use `4*J*m*N*8*depth_next`, `2*J*N*8*depth_i`, and `4*J*N*8*depth_i` bytes. Include public states, prepared matrices, lift scratch, device replicas, and peak concurrent waves.

The PDF's conservative profile has depths `1,3,7,15,31` when extending backward through stages (§13). Therefore no blanket speedup over BGG is promised. Benchmark equal public computations, arithmetic acceptance criteria, and output requirements; identify security assumptions separately. Include setup, preprocessing, online evaluation, storage/load, basis conversion, NTT, and peak memory. Report branch product counts separately from end-to-end latency.

## 7. Artifacts, validation, and proofs

Add BGV-specific modules for parameters, types/setup, operations/preparation, artifacts, and noise under `crates/exponent-lut/src/bgv/`; expose `BgvScheme` through `scheme.rs` and `lib.rs`. Group the existing concrete implementation as the BGG backend without moving generic BGG primitives into exponent-lut. Reorganize long files as needed without unrelated rewrites.

Keep prepared artifacts scheme-specific. BGG stores its necessary gadget preimages; BGV preprocessing stores low-base matrices and high-base public mask matrices; online preparation owns the separate secret-dependent helper rows. Update all dependent setup/import/export, cache-role, operation-count, and benchmark accounting paths. A program format change invalidates old serialized programs/checkpoints instead of loading them through a legacy branch.

Validation cost follows the user's execution contract: establish parameter, shape, level, layout, and artifact-binding preconditions at setup/import or graph construction, and let the executable graph rely on them. Do not add repeated coefficient scans, ciphertext relation checks, plaintext norm measurements, or host/device transfers to evaluation when declared input contracts suffice. One-hot and private payload bounds are caller contracts used by the public noise trace, not properties to rediscover from ciphertexts at runtime. Manifest certificate validation is an explicit setup/import operation and must not be invoked per gate, slot, or evaluation. Retain checks for genuinely dynamic inputs only where the contract cannot establish the required condition earlier.

Document each such precondition on the producing constructor or input API and preserve it through typed prepared handles. This includes equal levels for Add/public multiplication and the declared one-hot and payload-bound assumptions. Propagate larger payload bounds algebraically in the setup-time noise trace; do not measure plaintext coefficients during execution to enforce them. Expensive relation and norm checks belong in targeted tests, not in executable graph nodes or debug assertions on the evaluation path. A validation check proposed for that path must first explain why a setup or input contract cannot eliminate it.

Update the existing native Program/Lean linkage for the unified gate and selector semantics. Add BGV lemmas for exact centered extension/block switching, helper cancellation, error integrality, the flat identity, the noise recurrence, and level/scale composition. Operational relations for new primitive nodes belong to the owning shared layers; encoding/noise theorems belong to exponent-lut. Do not report an exported or admitted statement as a checked proof, and do not treat existing BGG decoder proofs as BGV proofs.

Proof progress: `ExponentBgv.lean` proves equal-level Add/public multiplication, helper cancellation with an explicit internal switch residual, branch aggregation, fresh-RHS wrap cancellation, the scaled outer-switch relation, and coefficientwise error-numerator divisibility with an explicit integer wrap witness. These algebraic lemmas are checked without admissions. They do not yet connect an exported BGV execution to all premises or prove the full deterministic norm recurrence. The BGG fused-message identity and its Program-state specialization are checked in `ExponentLut.lean` and `ExponentProgram.lean`; the identity branch derivation/exporter uses `identityLutStep`.

## 8. Sparse-LWR and terminal output integration

Migrating existing BGG PRF, setup, simulation, and readout callers away from `MulPrivate` is required for this change to be complete. Keep their payload semantics and existing decoder contracts explicit.

Implementation progress: `BgvScheme::compose_lut_material` now constructs the active-slot low-base ciphertext sum from public polynomial labels, preserving polynomial blocks. Its result enters the ordinary scalar LUT preparation, so centering/lifting and the public intermediate are computed after composition. CPU/GPU fixtures cover a nonzero range start, excluded nonzero selectors, both selected public labels, and modulo-wrap outputs. The common noise interpreter accepts the sum of public multiplier l1 bounds as the fresh-error gain; `BgvLutNoiseMaterial::from_one_hot_labels` pairs this with the maximum label payload bound under the caller's one-hot contract. These are setup-time metadata calculations. Application-level sparse-LWR wiring, persistence of the composed material and its noise metadata, and terminal targets remain required.

For the BGV sparse-LWR route (§9), prepare per-slot ciphertexts with targets `p_next D_k b_h r`. After receiving a public label, form

\[
\widehat C_k=\sum_h\sigma_k(M(v_h))C_{k,h}\pmod {q_{i+1}}.
\]

Reduce and center this composed ciphertext **before** computing `b_k`. Never sum individually rounded `b_(k,h)`. This composition can be implemented as scheme-specific preparation of an RHS family while the online Program still invokes one unified LUT. It requires an explicit one-hot selection contract and a public transform domain covering `u+v_h`; for the PDF's unstrided layout use `W>=2 Q_LWR-1`. The final block can combine modular reduction with LWR rounding.

The fresh RHS bound is `E_C <= m*eta`: unselected slot ciphertexts still contribute error. Stream slots and branches with bounded waves; preserve existing parallelism. Validate layout assignment failure, dummy slots, public-label binding, and cache invalidation for label changes.

Provide a typed terminal-target extension of LUT preparation for the PDF's coefficient/row outputs (§10); ordinary `LutTable` entries remain exponent values. In terminal mode, represent a public row-valued table separately so that an exponent `0`, coefficient `0`, and a zero row are never conflated. For a weighted bit at row position `a` and coefficient `j`, use fresh target `T_k=-omega X^j D_k u_a`. The result is `c=s a+omega*b*X^j u_a+t e`; the sign and omission of `p_next*r` are intentional. Large weights belong in fresh targets, not post-evaluation multiplication.

Batch terminal outputs by summing raw `K` and `z` at their common high base and switching each sum once. Expose terminal/readout capability through a scheme extension trait, with the typed BGV raw accumulator as its concrete implementation, not a new Program gate or a BGV-specific application call. Count all `g*J` branches and all output coefficients in noise/resource accounting.

Concrete BGV terminal arithmetic is implemented in
`crates/exponent-lut/src/bgv/terminal.rs`: public row-polynomial tables, fresh
negative targets without `p_next*r`, optional encrypted RHS, and separately
typed public/online raw accumulators. Both accumulators sum before the final
switch and bind the same ordered stages. The deterministic terminal noise
calculation counts all branches and adds outer rounding only once; weights
affect the integer-range check, not inherited or fresh error gains. CPU and GPU
fixtures cover 12 input combinations, zero rows, negative coefficients, both
row positions, `2^60` weights, and encrypted RHS shifts. GPU terminal and
prepared-Program regression tests each passed three consecutive runs. The
weighted terminal fixture now calls the common `PreparedTerminal` evaluator
through `ExponentLUTTerminalScheme`, retaining the exact switch-count check.
The common PRF fixture also connects its accumulated output to a terminal
rounding table. Terminal stage and batch persistence are implemented in
`bgv/terminal_artifacts.rs`, including the finalized public output and separate
online masks. Large-wave resource accounting still remains to be completed.

`bgv/terminal_family.rs` provides structural preparation and evaluation of
independent terminal outputs sharing a fixed row table. Public matrix-dependent
work uses one indexed graph body; online evaluation shares mask rows and
ciphertext lifts outside that body. Each member aggregates branches before
its own final row switch. These members are separate outputs, not terms of
one weighted terminal sum. The affine readout trait adapter connects this family carrier, including
structural artifact persistence.
The comparison fixture uses distinct public matrices and payloads over a
nonzero active range and checks exact equality with scalar terminal evaluation.

`bgv/readout.rs` implements scalar and structural-family affine value readout through
`ExponentLUTReadoutScheme`. Preprocessing embeds `h(k)=2*f(k)-1` in fresh
terminal targets. For a signed monomial, the negacyclic relation gives
`h(k+N)=-h(k)`; the online affine correction `(h+1)/2` therefore returns
`f(k)` or `1-f(k)`, matching the existing native PRF convention. If the
terminal relation is `c=s*A+h+t_plain*e` with even `t_plain`, transforming the
public key to `A/2` in preprocessing and the online row to `(c+[1,0])/2`
gives error `(t_plain/2)*e`. All divisions use the modular inverse of two at
the terminal modulus. The normalized bound on `e` is unchanged; the readout
public output explicitly reports its reduced error multiplier. Parameter
acceptance must still include the intermediate signed-table bound
`max |2*f(k)-1|`, not only the final value range.

The optional public row weight `w` routes and scales the output at
preprocessing: use terminal rows `h(k)=w*(2*f(k)-1)` and online correction
`(c+w)/2`. The result is `s*(A/2)+w*f+(t_plain/2)*e` in the first half,
and the corresponding complementary value in the second half. Thus a weight
such as `[0, q_t*2^h*X^j]` places a mask digit in the second polynomial's
j-th coefficient without multiplying inherited error by the digit weight.
The weight is part of the terminal table identity and the persisted readout
descriptor. Parameter acceptance uses the weighted intermediate table bound.
One preparation supplies one weight, shared across its scalar/family inputs;
structural preparation of different per-digit weights remains part of the
refresh routing connection.

For stride `d=2^m`, let `Y=X^d` and `W=N/d`, so `Y^W=-1`.
The existing layout automorphisms reduce to all odd residues modulo `2W`
on this subring. Consequently
`sum_sigma sigma(Y^a) = Y^a * sum_(j=0)^(W-1) (Y^(2a))^j`
is zero unless `a=0 mod W`, and equals `W*(-1)^(a/W)` otherwise.
Replacing each terminal table monomial by `Y^(-j)` thus gives the same
selected-row identity on `0..W` and its negation on `W..2W`. The affine
correction recovers the existing complementary PRF convention in both halves.
Only `W` branches are needed. Each automorphism still permutes signed
coefficients in the ambient degree-N ring, so the existing helper and noise
bounds apply with the actual branch count `W`; the ambient convolution
bounds continue to use `N`. This extension concerns the input payload
subring, not an assumption that helper or encoding errors lie in that subring.

This adapter requires one signed monomial per input, the full native subring
width `W=N/2^m` for stride `2^m`, and even `t_plain`. Unsupported layouts
and odd multipliers are rejected at preprocessing. Zero/general polynomial payloads
are not covered by the bounded affine-value contract. Both scalar and family
readout support the strided native subring. The terminal primitive keeps its
original semantics and error multiplier.

Scalar and family affine readout export their prepared terminal branches, finalized terminal
public matrix, and affine public matrix. Its descriptor binds the readout
identity and terminal binding to the preprocessing production. Import restores
these ports without public aggregation or block switching; only the constant
affine inverse and offset are reconstructed. Online secret-dependent masks
remain outside these public artifacts. The full native-domain round-trip
fixture closes and reopens the preprocessing store, restores both input
material and readout artifacts, rejects a mismatched stage identity, and
checks the online graph has exactly two row-block switches per input with
no public switches. The same fixture restores a structural family over a
nonzero active range and checks exact public/encoding equality with every
scalar member; the family adds one two-block online switch body.

The user's refresh layout clarification overrides the high-digit restoration
interface illustrated in PDF §10.3 for the intended BGV integration:

- BGG places the encoding residue to recover modulo `q_t` at scale `q/q_t`
  and uses the PRF mask to hide noise in the low digits.
- BGV places that residue in the low digits and scales the PRF mask by `q_t`
  to hide noise in the high digits. Here `q_t` is the reconstruction modulus,
  distinct from the encoding relation's `t_plain` error multiplier.

Low-digit recovery also needs an error-divisibility relation. For a decoded
integer representative `x + a*e + q_t*M`, reducing modulo `q_t` removes the
mask but retains `a*e mod q_t`. Mask scaling alone therefore does not convert
an arbitrary inherited error into a fresh one. The affine PRF readout has
`a=t_plain/2`; choosing its plaintext modulus as `2*q_t` makes that particular
readout error divisible by `q_t`. Whether slot-specific PRF parameters use
this relation, and how the current state's inherited error is transformed,
must be fixed in the BGV decoder construction. This condition does not rule
out a decoder that cancels or otherwise transforms the inherited error.

The BGV decoder target, public preprocessing, online mask preparation, and
noise accounting must all use this placement. An integer representative of
the intended combined output has the form `x + q_t*(z + M)`, with `x` the
desired residue, `z` the high-digit term to hide, and `M` the PRF mask.
This describes the layout, not a proved decoder construction. Recovering
`x` by reduction modulo `q_t` additionally requires specifying the ambient
modulus and representative/wrap convention; reduction through an unrelated
modulus does not automatically preserve it. Do not reuse BGG's high-digit
rounding or its mask bounds for this operation. Common orchestration must
delegate the layout-dependent operations to scheme capabilities.

The user clarified that refresh targets the selected scheme's own encoding:
BGG refreshes BGG and BGV refreshes BGV. The refresh state therefore uses
`ExponentLUTEncodingScheme::Encoding`. PRF output counts must follow that
state's actual polynomial count: BGG uses its gadget-dependent column count,
whereas the PDF BGV row has two polynomials. Both have N coefficients per
polynomial. Mask counts include the CRT-slot and mask-digit axes; fresh-error
labels are shared across CRT slots and must not acquire that multiplicity.
`encoding_dimensions` exposes these dimensions through the base scheme trait,
and `RefreshTranscriptTopology::new` derives the transcript dimensions without
inspecting ciphertext coefficients. `RefreshPrfLabelIndex::new` consumes this
same topology, so its mixed-radix label enumeration can use either scheme's
carrier without separate dimension arguments. Production BGG setup now shares
one topology constructor between label generation and noise accounting; its
sampler and decoder remain BGG-specific. With `C` state polynomials, mask-label
counts `d_t` per coefficient in slot `t`, and `d_f` fresh labels per
coefficient, the total is `C*N*(sum_t d_t + d_f)`. The construction of `d_f`
must include every independent PRF draw required by the fresh-error
distribution. These counts are shared inputs to routing, noise accounting,
and resource estimates; they are not a certificate that those BGV refresh
stages are already connected.

This implements the PDF's evaluation and terminal-output interfaces. The BGV
refresh/decoder connection remains unimplemented: the PDF does not supply a
decoder construction for the user's low-digit layout. Reject attempts to use
a BGG decoder with BGV artifacts. Do not claim modulus restoration by centered
lifting, and do not label a leveled BGV PRF run as end-to-end refreshed
evaluation. Before enabling that API, specify decoder/preimage generation,
combined masked targets, range/noise bounds, and protocol security for the
layout above. The affine readout adapter only produces PRF values; it does not
yet perform the `q_t`-scaled refresh masking.

## 9. Implementation sequence and acceptance gates

1. **Extract the scheme boundary.** Add `ExponentLUTEncodingScheme`, concrete BGG association, and generic preparation/evaluation adapters around `lower_program`. Initially preserve the four operations internally in this intermediate refactor. Acceptance: narrow BGG unit tests retain relation, visibility, family, and cache behavior; common code has no concrete GSW/preimage dependency.
2. **Fuse the final gate API and BGG helpers.** Change builder/validation/traversal, generate fresh RHS-bound branch targets, remove the standalone private-multiplication primitive, retain builder `private_mul` as a `lut_id=None` two-input LUT constructor, and migrate selector families. Update BGG public preparation, online evaluation, noise formulas, artifacts, PRF callers, and Lean statements together. Acceptance: the final source/API has only three gates and binary LUTs emit one gate, with no hidden preliminary GSW product.
3. **Add BGV parameters/types and exact CPU primitives.** Reuse compatible PR #157 shared-layer changes first; implement and validate prime/base context, CBD sampling, exact multi-limb centered extension, and whole-block switching with executable graph support. Acceptance: exact boundary/composite-drop tests and sampling consistency tests pass, and the production conversion path uses only residues/mixed-radix digits rather than full-coefficient big integers.
4. **Implement BGV setup and gate operations.** Add input/helper/RHS sampling, fixed-RHS public preparation, online evaluation, Add/MulPublic, and explicit level scheduling. Acceptance: all small-domain unary/binary inputs and two-stage scale composition satisfy the exact error relation.
5. **Implement the deterministic planner and artifact path.** Track all error terms, terminal ranges, and memory. Acceptance: both accepted manifests and deliberately insufficient noise/depth/memory budgets are exercised; reload reproduces the same public/online relationship and mismatched contexts fail.
6. **Add GPU execution of the same primitives and scheme graphs.** Preserve batching, device residency, and bounded memory. Acceptance: CPU/GPU exact arithmetic parity and repeated narrow GPU round trips; no host fallback.
7. **Complete generic applications and terminal capabilities.** Implement LWR PRF and noise-refresh orchestration using scheme traits and associated types only; migrate existing BGG callers and provide BGV implementations of supported capabilities. Implement streamed one-hot RHS composition and weighted/raw terminal output behind these interfaces, keeping the unavailable BGV decoder capability explicit. Acceptance: common application code has no concrete encoding-type dependency or scheme-type branch; all-input small LWR cases, inactive slots, large weights, and batched terminal error identities pass.
8. **Finish proofs, documentation, and comparative measurement.** Update architecture/crate/API docs, remove stale code and checkpoints, check the relevant Lean targets, and record production-path performance/resource results. Accept only claims actually supported by executed checks.

Each implementation step should remain reviewable, but intermediate four-gate scaffolding from step 1 is removed in step 2. There is no requirement to preserve old public APIs or serialized formats.

## 10. Validation plan

The checks below are future implementation acceptance criteria, not a report of completed validation. Plan revision requires no tests, builds, benchmarks, or implementation changes. During implementation, use the narrowest relevant unit tests, with small overridable parameters and fresh randomized samples. Tests under `crates/exponent-lut/tests/` are integration targets and require an explicit user request before execution. Put new targeted unit tests beside the defining implementation; do not move existing integration tests merely to bypass that restriction.

Required unit coverage:

- Program validation: all four combinations of optional LUT ID and RHS, `private_mul` emitting exactly one identity LUT and no `MulPrivate`, width/stride/range mismatch, undefined RHS, private visibility, family local IDs, active ranges, zero selectors, and one-hot reconstruction.
- BGG fusion: distinct mask/payload secrets, every small binary input, unary equivalence at the plaintext/relation level, exact branch error expansion, reused table with distinct RHS instances, and invalidated caches. Old and fresh fused ciphertext bytes are not expected to agree.
- Graph structure: one fused LUT, no preliminary private GSW product, online graph contains no public-key reconstruction/decomposition, and family capacities remain structural.
- Exact arithmetic: signed monomial wraps, automorphism/NTT agreement, centered boundaries `+/- (M-1)/2`, composite-block drops, retained-base order, `t=2` and `t=17`, and values where sequential one-prime rounding or PR #157's centered-CRT-term correction differs from the specified block operation. Include the `P=65, t=2, z=1` regression above and multi-limb lifts near the whole-product midpoint.
- BGV relations: exhaustive NAND, arbitrary small tables, general supported linear payloads, `None` RHS, nonterminal `p_next != 1`, terminal `p_d=1`, Add/MulPublic, rejection of unequal-level joins, explicit identity alignment before a join, and secret/context mismatch rejection.
- Noise: reconstruct the exact error numerator using test-only secrets/errors, verify divisibility by `p_i`, compare with the output relation, and check every deterministic bound. Include non-monomial `B_mu`, reused helpers, all-slot errors, and internal public rounding.
- Artifacts/planner: public-only preparation, changed RHS/label invalidation, wrong input matrix, wrong scale/base/root/scheme, exact prime products, and depth/memory/terminal-range rejection.
- Application/terminal: one-hot sparse-LWR, zero/dummy slots, final rounding, weights exceeding one CRT limb such as `2^60`, row targets, and aggregate-before-switch behavior.

Use trusted primitive polynomial operations and exact ring relations as expected-value sources, not a second hand-written cryptosystem that merely repeats the evaluator. The PDF describes reference Python files and manifests, but only the PDF was supplied here; do not claim those files were inspected or its reported test runs reproduced. Reproduction of those exact artifacts requires obtaining the actual bundle.

For Rust edits, run `cargo +nightly fmt --all`. Required CPU and GPU workspace library compilation checks are `cargo test -r --workspace --lib --no-run` and its `--features gpu` variant, warning-free. Actual test execution remains targeted. Run GPU tests outside the sandbox; for GPU/CUDA changes build once and run the same narrow round-trip binary command 3–5 times as required by repository guidance. Record all failures. Integration tests and remote benchmark execution remain subject to their explicit task authorization.

## 11. Definition of done and known boundaries

The implementation is complete when the common Program API selects either scheme without depending on its concrete keys/encodings; `MulPrivate` primitive operations and formats are removed while builder `private_mul` emits a two-input LUT with `lut_id=None`; fused BGG evaluation and noise analysis agree; BGV uses exactly two state polynomials and the PDF's exact arithmetic; parameter/resource failures are explicit; and relevant CPU/GPU/proof checks and documentation support the delivered claims.

This plan deliberately does not promise a literal one-polynomial state, constant storage independent of CRT depth, security certification, support for an unproved strided BGV layout, or an unspecified BGV decoder/refresh construction. Those boundaries follow directly from differences between the requested summary, existing implementation, and supplied mathematical specification. They do not prevent implementing the requested ExponentLUTEncodingScheme abstraction, fused LUT gate, and PDF-defined leveled BGV backend.

### Indexed terminal target routing

`sample_terminal_material` accepts an optional public 2-by-2 polynomial-block
output transform at the destination level. It right-multiplies the plaintext
row target before sampling each branch ciphertext's fresh error. The transform
can depend on a structural family index; it is never applied to an existing
ciphertext. Its identity is covered by the caller's material ID, and noise
planning must use the transformed target coefficient bound. The fresh-error
multiplier is unchanged. A numeric fixture evaluates distinct signed,
coefficient-shifted destinations inside one indexed graph body.

This terminal sampling operation is used by indexed material families and
indexed affine readout. Aggregate-before-switch routing into refresh remains
a separate connection.

### Structural terminal aggregation

`BgvTerminalFamilyMode::Sum` reduces raw member matrices during preprocessing
and raw encoding rows during online evaluation before performing one outer
block modulus switch. The result is a singleton output family; the stored
input range and branch intermediates retain the original member count.
`Independent` preserves per-member finalization in the existing indexed body.
The mode is included in the prepared stage identity and artifact descriptor,
and import uses the corresponding output cardinality. Shared ciphertext lifts
and online mask preparation remain outside the member evaluation body.
Scalar raw-accumulator comparisons cover both public and encoding outputs.
Both shared and indexed terminal material support this aggregation. Aggregate
affine readout is connected below; production refresh routing remains outstanding.

### Indexed terminal material families

`sample_terminal_material_family` builds fresh branch ciphertexts for a public
family of output transforms in one indexed graph body. Its terminal-specific
carrier prevents ordinary LUT ciphertext families from being substituted as
row targets. `prepare_terminal_family` and `evaluate_terminal_family` accept
both shared and indexed material, with independent or aggregate output modes.
Shared material retains its single stored ciphertext set and loop-invariant
lifts; indexed material stores a structural ciphertext family and lifts each
member's ciphertexts in the member body. Both use one shared online mask bank.

The prepared-family descriptor records whether ciphertexts are shared or
indexed. Export/import uses matching scalar-block or family ports. Numeric
scalar comparisons cover indexed preparation/evaluation and raw aggregation;
A file-store reopen test covers both output modes with two fresh online secrets,
checks their plaintext rows against the public targets, and applies the
corresponding scalar/aggregate noise bounds. Indexed affine readout uses these
artifacts as described below.

### Indexed affine readout

`BgvReadoutWeights` accepts either shared constant row polynomials or a public
indexed row family with a setup-owned coefficient bound. Indexed readout
samples fresh terminal targets `w_i*(2*f-1)` and prepares independent family
outputs through the same readout trait. It stores each `w_i` as a public offset
family and restores those offsets together with the indexed terminal material.
Online evaluation computes `(c_i+w_i)/2`; finalized public matrices `A_i/2`
remain preprocessing artifacts. For the scalar signed base table, the target
bound is its coefficient bound times the supplied row coefficient bound;
fresh-error multipliers remain unchanged. No runtime coefficient scan is added.

The full-domain fixture includes changing signs, polynomial destinations, and
coefficient shifts across members, including the complementary second half,
and closes/reopens the artifact store before evaluation. Aggregate affine readout uses terminal sum mode as described below;
independently finalized readout rows must not be substituted for raw
aggregate-before-switch terminal terms.

### Aggregate affine readout

The readout material's optional terminal family mode selects independent
outputs or one aggregate. In sum mode, public preparation and online terminal
evaluation reduce raw members before one outer block modulus switch. Public
preparation also computes `sum_i w_i` (or `count*w` for a shared weight), stores
it as a separate polynomial-block offset, and finalizes `A_sum/2`. Online
readout returns `(c_sum + sum_i w_i)/2`, with payload `sum_i w_i*f_i` and the
same normalized aggregate terminal error bound. It neither recomputes the
public offset sum nor repeats public modulus switches.

Artifact descriptors retain the terminal input range and mode; public output
and encoding output use a singleton range in sum mode. Source weight families
remain distinct from the saved aggregate offset. The reopen fixture covers
both shared and indexed weights and evaluates the full signed native domain.
Its aggregate bound counts every member's branch errors and only one outer
rounding. Production refresh routing and the BGV decoder remain unimplemented;
this readout capability does not itself establish BGV self-refresh.

### Low-digit decoder compatibility audit

The remaining decoder is a mathematical interface gap, not an unimplemented
call to the existing BGG decoder. Keep the following moduli distinct:

- `Q_term`: the modulus in which the PRF terminal/readout row is evaluated.
- `q_t`: one CRT modulus of the encoding to be restored.
- `t_prf`: the plaintext/error multiplier of the BGV PRF evaluator.

The implemented affine terminal relation is
`c_m = s*A_m + q_t*M + (t_prf/2)*e (mod Q_term)` when the public row weights
route `q_t` times the mask digits. Weighting the fresh plaintext target does
not weight the fresh ciphertext error. For direct low-digit extraction of
`R_t + q_t*(M+E)`, the PRF error must therefore either be canceled by a specified
decoder or have a multiplier divisible by `q_t`. For example, with `q_t=17`,
`t_prf=2`, and normalized error `e=1`, the affine readout error has residue 1,
regardless of the mask weight. This example addresses the PRF error term;
removing inherited error from the refresh target is a separate obligation.

Setting `t_prf=2*q_t` is not a valid standalone fix under the PDF's terminal
modulus contract. PDF equation (2.2) requires every ciphertext CRT prime to be
coprime to `t_prf`, enforced by `BgvParameters::new` in
`crates/exponent-lut/src/bgv/parameters.rs`. PDF section 10.3 instead places the
terminal at the restoration modulus `Q_term=q_star=product_t q_t`. Thus a
restoration CRT prime would divide both `Q_term` and `t_prf`. Do not remove the
coprimality requirement to make the parameters constructible, and do not
claim that `q_t`-weighted targets alone remove this obstruction.

PDF equations (10.4)-(10.5) specify high-spacing extraction and explicitly
leave decoder generation, preimage norms, and security outside the evaluation
core. The requested low-digit protocol needs an additional construction that
establishes its post-decoder equation, including inherited state error,
terminal error, decoder error, and fresh-error terms. Its public target must
include the masks before a decoder is issued; the existing generic refresh
traits preserve this ordering but do not supply that construction for BGV.

An auxiliary `Q_term` coprime to `q_t` could avoid the immediate parameter
conflict, but this would change the PDF's refresh interface. Reduction of a
residue modulo such a `Q_term` is not a well-defined map to residues modulo
`q_t`: replacing an integer representative by that integer plus `Q_term`
changes its `q_t` residue. A construction using this route must prove a
centered no-wrap bound for the decoded integer and explain the conversion of
the old state and the final CRT reconstruction. Neither that bound nor that
conversion has been established. This audit does not prove all possible
low-digit decoders impossible; it rules out the direct parameter/mask-only
substitution and records the missing proof obligation.

### Grouped terminal noise accounting

`terminal_noise_bound` accepts a group's total branch count. For `g` members
with identical input bounds, RHS gains, and `W` branches, use one group with
`g*W` branches rather than enumerating `g` identical records. The numerator is
linear in each group's branch count; its outer rounding term remains outside
the group loop. Distinct input bounds or RHS gains require distinct groups.
An expanded-versus-grouped test checks equality of normalized bounds and total
branches for heterogeneous groups and non-power-of-two member counts.
`BgvReadoutMaterial::target_bound` remains a per-member bound, including in sum
mode; aggregate payload bounds must separately sum all member target bounds.
This avoids confusing the number of PRF outputs with LUT width or counting
outer rounding once per member. Production BGV refresh/decoder routing remains
blocked on the low-digit interface conditions above.

### Implementation checkpoint: decoder clarification required

The current implementation is not accepted as complete. The program scheme
boundary, unified LUT representation, BGV arithmetic, terminal/readout families,
artifact persistence, and aggregate affine readout have implementations and
component validations. `ExponentLUTRefreshScheme` and its setup/online decoder
capabilities are implemented for BGG only; there is no BGV decoder adapter.

Work on the complete BGV self-refresh path is paused pending the low-digit
post-decoder equation/construction requested from the user. The direct
`q_t`-weighted-mask substitution and the proposed `t_prf=2*q_t` substitution
under the PDF's `Q_term=q_star` contract are insufficient for the reasons in
the compatibility audit. No claim is made that every alternative construction
is impossible. Required clarification is the intended decoder construction,
or an additional specification establishing removal of inherited error and
compatibility of the PRF terminal modulus with low-digit extraction.

The original objective remains unchanged. After resolving this interface,
remaining work includes production refresh routing and s/t phase connections,
complete refresh noise/resource accounting, end-to-end self-refresh validation,
and the full acceptance audit. Component tests and arithmetic build checks do
not discharge those requirements. No BGG decoder is substituted for BGV and no
PDF coprimality condition is relaxed to bypass the unresolved construction.
