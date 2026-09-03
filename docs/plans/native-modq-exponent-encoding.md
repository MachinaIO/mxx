# Native mod-Q Exponent Encoding and Artifact-Backed BGG Cache

## Status

This document is the self-contained implementation specification for replacing
the sparse-key LWR PRF path in `mxx-exponent-lut` with native mod-Q exponent
encoding. It incorporates the design decisions made after the original native
mod-Q proposal.

This document does not authorize implementation by itself. No backward
compatibility is required: obsolete APIs, artifact layouts, helper names, and
simulation paths are deleted rather than retained behind adapters.

The principal decisions are:

1. the PRF accumulator is represented as a strided monomial and wraps modulo
   Q through negacyclic ring arithmetic;
2. the PRF chain contains no reduction or rounding LUT;
3. every LUT is evaluated by one single-layer automorphism sum; there is no LUT
   evaluation mode and no public identifier containing `Flat`;
4. the frozen cryptographic gate set is `MulPrivate`, `MulPublic`, `Add`, and
   `Lut`;
5. two-input LUTs and one-hot selection are public builder APIs expanded into
   those four gates;
6. the encodings of the PRF coordinates are supplied as online BGG+ encoding
   inputs;
7. a public attribute is represented only by the existing
   `BggEncodingWire::plaintext: Option<Mat>` field;
8. preprocessing evaluates the public-key projection of the program and builds
   decoder preimages, while online evaluation computes the encoding-vector
   projection and consumes those preprocessing artifacts;
9. only gadget decompositions are caches; public-key matrices remain ordinary
   protocol artifacts;
10. decomposition caches use the existing typed `Preimage` artifact path and
    are loaded in bounded waves instead of remaining resident in RAM or VRAM;
11. Exponent-LUT acceptance uses only the application-specific simulator in
    `mxx-exponent-lut`; the generic noise simulator is not an authority or a
    fallback for this application.

## Scope

The work changes:

- exponent layout and monomial construction;
- Exponent-LUT program declarations, public APIs, and lowering;
- BGG public/private multiplication interfaces needed for explicit cached
  decompositions;
- sparse-key LWR PRF setup and online evaluation;
- value-form PRF readout;
- mask and fresh-error digit routing;
- preprocessing artifact production and online artifact imports;
- artifact storage for large deterministic decomposition caches;
- Exponent-LUT-specific worst-case and average-case noise simulation;
- parameter search, storage accounting, tests, and benchmark estimation.

The work does not:

- add a compatibility layer for the old PRF reduction path;
- synthesize full value encodings from bit encodings in the first version;
- add a sequential or hybrid LUT implementation;
- redesign the generic noise simulator;
- require an adversarial commitment or force a trusted local simulator to
  recompute logged results;
- change matrix dimensions to make benchmark estimates fit in memory; or
- introduce a CPU fallback for GPU failures.

## Notation

Let:

- `n` be the power-of-two ring dimension;
- `R_q = Z_q[X] / (X^n + 1)`;
- `m` be the exponent stride logarithm;
- `n' = n / 2^m`;
- `Q = 2n' = 2n / 2^m` be the native exponent modulus;
- `p'` be the positive half-width of the balanced PRF output;
- `P = 2p'` be the complete balanced digit base;
- `G` be the configured gadget matrix;
- `ell_beta` be the total gadget decomposition digit count across active CRT
  limbs; and
- `B_chi` be the common helper-error cutoff.

Define

\[
  \operatorname{mono}_m(z) = X^{2^m z},
\]

including the sign induced by reduction modulo `X^n + 1`. Then

\[
  \operatorname{mono}_m(a)\operatorname{mono}_m(b)
  = \operatorname{mono}_m(a+b \bmod Q).
\]

Every strided exponent attribute belongs to

\[
  S_m = Z_q[X^{2^m}] \subseteq R_q.
\]

## Exponent layout

Add `crates/exponent-lut/src/layout.rs` with:

```rust
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExpLayout {
    pub stride_log2: u32,
    pub width_log2: u32,
}
```

For a concrete ring dimension, construction validates:

\[
  0 \le m < \log_2 n,
  \qquad
  m + \texttt{width\_log2} \le \log_2 n.
\]

The layout exposes:

```rust
impl ExpLayout {
    pub fn validate(&self, ring_dimension: usize) -> Result<(), LayoutError>;
    pub fn stride(&self) -> usize;
    pub fn width(&self) -> usize;
    pub fn effective_modulus(&self, ring_dimension: usize) -> usize;
    pub fn mono(&self, ring: &Ring, exponent: &BigInt) -> Result<Mat, LayoutError>;
    pub fn automorphism_indices(
        &self,
        ring_dimension: usize,
    ) -> Result<Vec<usize>, LayoutError>;
}
```

For `w = width_log2`, the automorphism indices are

\[
  J_{m,w}
  = \left\{
      j_S = \prod_{i\in S}
      \left(\frac{2n}{2^{i+1}}+1\right) \bmod 2n
      : S\subseteq[m,m+w)
    \right\}.
\]

Construction asserts that the set has exactly `2^w` distinct odd elements and
contains identity exactly once. The old congruence-class enumeration is
deleted. For `m = 0`, the resulting ordering and values must reproduce the
current implementation bit-for-bit.

`width_log2` describes an operation domain, not a permanent bound on every
wire. Wires retain the stride needed to establish the `S_m` invariant.

## Program wire knowledge

The existing plaintext metadata is the sole source of public attribute values:

```rust
pub struct BggEncodingWire {
    pub vector: Mat,
    pub pubkey: BggPublicKeyWire,
    pub plaintext: Option<Mat>,
}
```

No parallel `known_values`, `public_exponent`, or provenance-value field is
introduced. Program input declarations state whether public plaintext is
required:

```rust
pub enum AttributeKnowledge {
    Private,
    Public,
}

pub struct ProgramWireType {
    pub stride_log2: u32,
    pub knowledge: AttributeKnowledge,
}
```

`Public` requires `plaintext.is_some()` at binding time. `Private` permits
`None`. The current blanket rejection of all plaintext metadata at the
Exponent-LUT boundary is removed.

Knowledge propagation is:

- `MulPrivate`: private unless both operand relations explicitly expose their
  plaintexts;
- `MulPublic`: the public operand must be `Some`; the output is public only if
  the other operand is also public;
- `Add`: public only if both operands are public; and
- `Lut`: preserves whether its input is public and computes the known output
  through the table when it is public.

## Frozen gate set

Replace the current unary, binary, and one-hot gate variants with:

```rust
pub enum ProgramGate {
    MulPrivate {
        lhs: ProgramWireId,
        rhs: RhsInputId,
        output: ProgramWireId,
    },
    MulPublic {
        private: ProgramWireId,
        public: ProgramWireId,
        output: ProgramWireId,
    },
    Add {
        lhs: ProgramWireId,
        rhs: ProgramWireId,
        output: ProgramWireId,
    },
    Lut {
        input: ProgramWireId,
        lut: LutId,
        output: ProgramWireId,
    },
}
```

### MulPrivate

`MulPrivate` multiplies an encoding attribute by the attribute carried by a
typed RHS/GSW package:

\[
  \operatorname{attr}(out)
  = \operatorname{attr}(lhs)\operatorname{attr}(rhs).
\]

If both are monomials, this is exponent addition. If the RHS carries a hidden
selector bit `b`, this is the gate

\[
  b\operatorname{mono}_m(z).
\]

The name describes visibility rather than an Exponent-LUT-specific selector
role. The RHS remains a typed `RhsInputId` because private multiplication
requires the special RHS representation.

### MulPublic

`MulPublic` multiplies by a BGG+ encoding whose attribute is public. For a
public encoding `(c_x, A_x, Some(x))` and a private encoding `(c, A, _)`, it
computes

\[
  c_{out} = c_x G^{-1}(A) + x c,
  \qquad
  A_{out} = A_x G^{-1}(A).
\]

The operation rejects a public operand with `plaintext == None`. `Public`
refers to the attribute value, not to an unencrypted carrier or to a public
key alone.

### Add

`Add` is ordinary addition of encoding relations:

\[
  c_{out}=c_1+c_2,
  \qquad
  A_{out}=A_1+A_2,
\]

and therefore adds the represented ring attributes. It is not exponent
addition. Family sums are structured reductions of this primitive.

### Lut

`Lut` maps

\[
  \operatorname{mono}_m(u)
  \longmapsto
  \operatorname{mono}_m(L(u))
\]

by one single-layer automorphism sum. There is no evaluation mode. Types,
functions, artifacts, report fields, and logs use `Lut`, `LutHelper`,
`LutMaskBank`, and corresponding public variants; they do not include `Flat`.

The coefficient for an automorphism `sigma` is

\[
  D_{\sigma,L}
  = \frac{1}{W'}
    \sum_{k=0}^{W'-1}
      \operatorname{mono}_m(L(k))
      \sigma\!\left(\operatorname{mono}_m(-k)\right).
\]

## Structured program control

Cryptographic semantics remain in `ProgramGate`. Generic structural nodes
describe repetition and reduction without introducing a one-hot cryptographic
gate:

```rust
pub enum ProgramStep {
    Gate(ProgramGate),
    FamilyMap { /* typed bindings and body */ },
    FamilyReduceAdd { /* input family and output wire */ },
}
```

The implementation reuses the existing DSL `Family` map, zip, gather, and
balanced-sum support. It must not emit one host-built graph body per slot or
label.

Program identity is computed from the canonical expanded gate/structural IR,
layout, tables, shapes, and knowledge requirements. It does not include actual
label values.

## Public builder APIs

The public builder exposes convenient compound operations while the frozen
gate set remains minimal:

```rust
impl ExponentProgramBuilder {
    pub fn lut(
        &mut self,
        input: Wire,
        table: UnaryLutTable,
    ) -> Result<Wire, ProgramError>;

    pub fn lut2(
        &mut self,
        lhs: Wire,
        rhs: PrivateRhsInput,
        table: BinaryLutTable,
    ) -> Result<Wire, ProgramError>;

    pub fn select(
        &mut self,
        input: Wire,
        selectors: PrivateRhsFamily,
        public_values: PublicWireFamily,
    ) -> Result<Wire, ProgramError>;
}
```

### Two-input LUT

For inputs `u,v in [0,B)`, the RHS package carries
`mono_m(Bv)`. The builder constructs the bundled table

\[
  L'(u+Bv)=L(u,v)
\]

and expands

```rust
let bundled = builder.mul_private(lhs, rhs)?;
builder.lut(bundled, bundled_table)
```

because

\[
  \operatorname{mono}_m(u)\operatorname{mono}_m(Bv)
  = \operatorname{mono}_m(u+Bv).
\]

The RHS declaration records its exponent scale and validates

\[
  m + 2\log_2B \le \log_2n.
\]

### One-hot selection

The public selection API represents

\[
  \operatorname{select}(x,\{b_i\},\{v_i\})
  = x\sum_i b_i v_i,
  \qquad
  b_i\in\{0,1\},\quad \sum_i b_i=1.
\]

It expands in this mandatory order:

```rust
let branches = zip(selectors, public_values).map(|selector, value| {
    let selected = builder.mul_private(input, selector)?;
    builder.mul_public(selected, value)
});
branches.reduce_add(builder)
```

Applying `MulPublic` before `MulPrivate` is forbidden in this lowering because
it would amplify the fresh value-encoding error by the private multiplication
gain. Selection does not implicitly apply a LUT. A caller that needs a LUT
calls `lut` on the selected result.

## Native sparse-key LWR PRF profile

Replace the old profile and reduction plan with:

```rust
pub struct SparseLwrPrfProfile {
    pub layout: ExpLayout,
    pub p_prime: usize,
}
```

For the PRF, `layout.width_log2 = log2(n) - m`, so

\[
  n'=2^{\texttt{width\_log2}},
  \qquad Q=2n',
  \qquad P=2p'.
\]

`q_l`, `lut_width`, reduction intervals, and terminal LUT width are not
independent parameters and are deleted.

## PRF online inputs

For every PBC block `j` and slot `i`, online evaluation receives a BGG+
encoding of

\[
  \operatorname{mono}_m(a'_j[i])=X^{2^m a'_j[i]}.
\]

The relation is

\[
  c_{j,i}
  = s A_{j,i}
    - \operatorname{mono}_m(a'_j[i])sG
    + e_{j,i},
\]

where `A_{j,i}` is setup-fixed and independent of the value and label. Dummy
slots use `a'_j[i]=0` and may share a setup-time encoding of one.

The logical input is:

```rust
pub struct NativePrfLabelInputs {
    pub exponent_encodings: Family<BggEncodingWire>,
}
```

For this PRF, `a'_j[i]` is public to the evaluator, so each imported wire has

```rust
plaintext: Some(layout.mono(ring, a_prime_j_i)?),
```

with no duplicate clear-value field. If another application supplies a hidden
attribute, it leaves `plaintext` as `None` and cannot use that wire as the
public operand of `MulPublic`.

At the transport boundary, setup-fixed public matrices are not duplicated for
every label. A label payload contains the private vector and a setup public-key
slot reference; import reconstructs the full `BggEncodingWire`. The public
plaintext is derived from the label/hash computation rather than transmitted
as a matrix artifact.

## PRF accumulation

Initialize the state with attribute `mono_m(0)=1`. For each block, invoke the
public `select` API. Its slot-level expansion is

\[
  f_i = c\,G^{-1}(C_{j,i}),
  \qquad
  A_{f_i}=A\,G^{-1}(C_{j,i}),
\]

followed by

\[
  g_i
  = c_{j,i}G^{-1}(A_{f_i})
    + \operatorname{mono}_m(a'_j[i])f_i,
\]

\[
  A_{g_i}=A_{j,i}G^{-1}(A_{f_i}),
\]

and

\[
  c_{next}=\sum_i g_i,
  \qquad
  A_{next}=\sum_i A_{g_i}.
\]

Exactly one selector is one. Therefore the next attribute is

\[
  \operatorname{mono}_m(acc+a'_j[i^*] \bmod Q).
\]

No LUT is called inside the block sequence.

The public recurrence contains only setup-fixed `A`, `A_{j,i}`, and
`C_{j,i}`. It is independent of `a'_j[i]` and therefore independent of the
label. Preprocessing evaluates this public recurrence once.

## Value-form readout

The accumulator program is followed by a separate `ValueReadout`; it is not a
fifth `ProgramGate` because it exits the exponent-encoded domain and returns a
constant ring attribute.

```rust
pub struct SparseLwrPrfProgram {
    pub accumulation: ExponentLutProgram,
    pub readout: ValueReadout,
}

pub struct ValueReadout {
    pub layout: ExpLayout,
    pub table: Mat,
}
```

Define

\[
  T'
  = \sum_{k=0}^{n'-1}
      (2f(k)-1)\operatorname{mono}_m(-k).
\]

Readout performs:

1. public multiplication `c <- T'c`, `A <- T'A`;
2. one single-layer trace over all indices `J_{m,log2(n')}`, including the
   identity term;
3. normalization by the decomposition of `(2n')^{-1}G`.

The trace has `n'` terms and requires `n'-1` plain automorphism helpers. There
is no sequential part or split parameter. For `n=2^16` and `m=5`, this means
2048 terms and 2047 non-identity helpers.

The output attribute is

\[
  g=f(z)-\frac12.
\]

## Public preprocessing and online evaluation boundary

Preprocessing performs:

1. setup of selector RHS/GSW material;
2. setup of the value-encoding public matrices `A_{j,i}`;
3. evaluation of the complete public-key projection of the PRF accumulation;
4. evaluation of the public-key projection of value readout;
5. generation of decoder preimages;
6. generation and export of only the gadget decompositions required by online
   right-hand-side multiplications.

Online evaluation performs:

1. label/hash evaluation producing the public `a'_j[i]` values;
2. import of the separately supplied encoding vectors;
3. reconstruction of BGG+ input wires using setup-fixed public keys and public
   plaintext metadata;
4. vector-side execution of the PRF accumulation;
5. vector-side value readout;
6. digit routing and public half-offset addition; and
7. application of preprocessing decoder preimages to obtain refreshed
   encodings.

In cache-required online mode, public-key arithmetic and gadget decomposition
are not recomputed. Missing required artifacts are errors; there is no silent
fallback.

## Public keys versus decomposition caches

The distinction is normative:

- a public-key matrix `A` is protocol state and is not a cache;
- `G^{-1}(A)` is deterministic acceleration data and is a cache.

`BggPublicKeyWire` remains unchanged and does not contain an inline
`Option<Preimage>`:

```rust
pub struct BggPublicKeyWire {
    pub matrix: Mat,
    pub reveal_plaintext: bool,
}
```

Public matrices that must cross the preprocessing/online boundary are exported
once as ordinary public matrix artifacts. They are never duplicated inside a
cache artifact and are not counted as cache bytes.

Only decompositions needed by later operations are exported as cache
artifacts. Public matrices and decomposition caches are associated by setup
identity, program gate, family coordinate, and operand role. This association
is ordinary artifact wiring, not an adversarial proof or a new cryptographic
commitment.

## Reusing the typed Preimage artifact path

No new cache payload or decomposition wire type is introduced. `Mat::decompose`
already returns the typed `Preimage` used by `mul_small_rhs`:

```rust
let decomposition: Preimage =
    public_key.matrix.clone().decompose(base, digit_count);
```

Preprocessing exports scalar or family decompositions with the existing APIs:

```rust
context.public_preimage_output(name, decomposition)?;
context.public_preimage_family_output(name, decompositions)?;
```

Online imports them with:

```rust
ring.preimage_artifact_input(
    production_id,
    name,
    shape,
    coefficient_bound,
    ArtifactConfidentiality::Public,
);

ring.preimage_family_artifact_input(
    production_id,
    name,
    family_shape,
    matrix_shape,
    coefficient_bound,
    ArtifactConfidentiality::Public,
);
```

The coefficient bound remains part of the `Preimage` artifact type and is
validated before compact multiplication. A generic matrix artifact may not be
relabeled as a preimage.

## Decomposition cache planning

The public program is analyzed before preprocessing. A public matrix is
decomposed only if it is consumed as a bounded right operand during online
evaluation. Unused public outputs and unused decompositions are not exported.

For each gate:

- `MulPrivate` caches the RHS/GSW decomposition and exports the ordinary output
  public key only when it is needed across the stage boundary;
- `MulPublic` caches the private operand's public-key decomposition and exports
  its ordinary output public key when needed;
- `Add` needs no cache by itself; its output is decomposed only if a later gate
  consumes it as a right operand;
- `Lut` caches the helper RHS decompositions and the relation-bearing
  intermediate decompositions required by its branches; and
- value readout caches its automorphism helper decompositions and normalization
  decomposition.

For a PRF block and slot, the principal cached values are

\[
  D_{C,j,i}=G^{-1}(C_{j,i}),
  \qquad
  D_{f,j,i}=G^{-1}(A_{f,j,i}).
\]

They are exported as rectangular `Family<Preimage>` artifacts with structural
shape such as `[block_count, slot_capacity]`, not as thousands of separately
named scalar artifacts.

## Artifact ownership and storage

The existing workspace layers retain their responsibilities:

- `mxx-bgg` exposes cache-aware operations that accept explicit typed
  `Preimage` operands and never owns an application program ID;
- `mxx-exponent-lut` decides which public program values require decomposition,
  declares artifact names and family coordinates, and binds them to gates;
- `mxx-dsl` retains the existing preimage output/input constructors;
- `mxx-runtime` stores, loads, stages, and releases artifact payloads; and
- `mxx-primitives` owns compact bounded representations, GPU owners, and
  `mul_small_rhs` execution.

The cache uses the existing `ArtifactStore` interface and existing
`ArtifactType::Preimage` payload. `MemoryArtifactStore` remains suitable for
small unit tests only.

The repository currently has no non-memory `ArtifactStore` implementation.
To guarantee that production does not retain every cache payload in RAM, add a
filesystem-backed implementation in `mxx-runtime`. It must implement the same
artifact and session contracts, store each family member independently, write
through a temporary file followed by atomic rename, and load only the requested
member. Cache-specific file formats or cache-specific stores are forbidden.

The manifest and small metadata may remain resident. Matrix and preimage
payload bytes remain owned by the artifact store.

## Bounded load/compute/store lifecycle

Artifact handles are lazy. For each scheduled wave:

```text
load the required preimage artifact columns
transfer them to the assigned GPU
reuse them for the current label/output wave
run mul_small_rhs or mul_decomposed
store any required output artifact
release the loaded columns and temporary owners
advance to the next wave
```

The same decomposition is loaded once per label wave, not once per label. The
entire decomposition collection is never materialized simultaneously.

The logical input and output matrix sizes are unchanged. Objects whose element
count is `O(log q)` may remain resident if they fit. Objects whose row and
column counts are both proportional to `log q` are retained in compact
coefficient form and their DCRT/NTT expansion is streamed by columns.

## GPU residency and multi-GPU policy

Use only `MXX_GPU_VRAM_PERCENT` for the GPU memory policy. The percentage is
read once and fixed when the GPU context is created; changing the process
environment afterward does not alter that context or its estimator. The
default is 80 percent. Each device derives its own byte budget from its
physical total VRAM, so this is a per-device residency/workspace policy rather
than an aggregate fleet budget. There is no fixed-byte small-matrix residency
environment variable, and the policy does not permit retaining the complete
cache collection.

Column parallelism is derived dynamically per operation. A one-column warmup
measures the operation's fixed residency, incremental workspace, and allocator
high-water behavior on each GPU role. The runtime then selects the largest
safe concurrent column count within that device's snapshotted percentage
budget. No fixed column width or fixed residency-byte limit is reused across
different matrix shapes.

At context setup, determine:

1. the artifact family members needed by each operation;
2. the label/output waves assigned to each GPU;
3. the maximum columns loaded for one operation;
4. the small, frequently reused artifacts that may remain resident; and
5. the larger artifacts that must be streamed.

Each complete CRT-limb matrix remains on one device. Work is distributed
symmetrically by label/output wave where operations have the same shape.
Shared setup data is loaded once per device when it fits; otherwise each device
streams its assigned columns. No device-wide synchronization is added. Loads,
kernel work, and stores are ordered with streams and events.

## Storage accounting

Reports separate protocol material from deterministic acceleration data:

```rust
pub struct ExponentLutStorageReport {
    pub canonical_public_artifact_bytes: u64,
    pub private_label_input_bytes: u64,
    pub decoder_preimage_bytes: u64,
    pub decomposition_cache_artifact_bytes: u64,
    pub peak_loaded_host_bytes: u64,
    pub peak_loaded_device_bytes_per_gpu: u64,
    pub peak_workspace_bytes_per_gpu: u64,
}
```

Only deterministic gadget decompositions contribute to
`decomposition_cache_artifact_bytes`. Public-key matrices do not. Artifact
total bytes do not contribute in full to RAM or VRAM peak; only the maximum
simultaneously loaded wave does.

## Digit routing and half-offsets

The value readout returns

\[
  g=f(z)-\frac12,
\]

while the desired digit is

\[
  d=g+\frac12=f(z)\in\{-p'+1,\ldots,p'\}.
\]

Use digit base `B=P`. The digit counts are

\[
  C_m=\left\lceil\frac{\omega_m}{\log_2P}\right\rceil,
  \qquad
  C_e=\left\lceil\frac{\log_2B_e}{\log_2P}\right\rceil.
\]

Mask and fresh-error outputs each have `2*ell_beta` columns. The routed targets
are

\[
  -B^cX^j u_2\delta_k^T
\]

for masks and

\[
  -\frac{q}{q_t}B^cX^j u_2\delta_k^T
\]

for fresh errors.

Add the public mask offset

\[
  o_m(1+X+\cdots+X^{n-1}),
  \qquad
  o_m=\frac12\sum_{c<C_m}B^c,
\]

and the public fresh-error offset

\[
  \frac{q}{q_t}o_e(1+X+\cdots+X^{n-1}),
  \qquad
  o_e=\frac12\sum_{c<C_e}B^c.
\]

These are public vector additions. They change neither the public matrix nor
the noise.

## Exponent-LUT-specific noise authority

`crates/exponent-lut/src/noise.rs` is the only correctness noise authority for
this application. Parameter search and integration tests must not translate
the program to the generic noise simulator, compare against it, or fall back
to it. Remove the dependency on `mxx-noise-simulator` if no unrelated use
remains.

### Coordinate conventions and structural input

The simulator uses exact integer or rational arithmetic only. It maintains two
explicit channels:

- `WorstCase`, whose state `E` is an absolute coefficient bound; and
- `AverageCase`, whose state `V2` is the per-coefficient variance of the
  doubled error `2e`.

The average helper variance `V_chi,2 = Var(2e_chi)` comes from the configured
sampler distribution. It is not reconstructed from the worst-case cutoff
`B_chi`. All helper errors use the common cutoff `B_chi`, but distinct helper
instances are fresh sources in the average channel unless a structured
lowering explicitly reuses one.

For the linear map induced on error coefficients by right multiplication with
a concrete decomposition `D`, define

\[
  \gamma_1(D)=\max_y\sum_x|D_{y,x}|,
  \qquad
  \gamma_2^2(D)=\max_y\sum_xD_{y,x}^2.
\]

Here `x` ranges over all input matrix and ring-coefficient coordinates that
contribute to output coordinate `y`. The implementation computes both values
from the same balanced power-of-two decomposition used by the evaluator.
Multiplication or rotation by `+/- X^j` has exactly

\[
  \gamma_1=\gamma_2^2=1.
\]

No value proportional to `q`, `q/q_t`, an exponent, or a rotation index is
used directly as a noise gain.

Replace the current dense, detached parameter object with a setup-derived
snapshot containing:

```rust
pub struct RightActionNoiseGain {
    pub l1: BigUint,
    pub l2_squared: BigUint,
}

pub struct ExponentLutNoiseSnapshot {
    pub authority: NoiseModelKind,
    pub layout: ExpLayout,
    pub program: ExponentLutProgram,
    pub initial_state_bound: BigUint,
    pub initial_state_doubled_variance: ExactRational,
    pub helper_error_cutoff: BigUint,
    pub helper_doubled_variance: ExactRational,
    pub gate_gains: Vec<GateNoiseGains>,
    pub readout_gains: ValueReadoutNoiseGains,
    pub refresh: RefreshNoiseParameters,
}
```

`gate_gains` and `readout_gains` are generated from the same typed
decomposition plans that produce the evaluator artifacts. They are not
caller-supplied scalar estimates. Parameter search may construct the gains
without materializing GPU artifacts, but it must run the identical balanced
decomposition algorithm over the identical targets, base, digit count, shape,
and CRT moduli.

The snapshot is ordinary trusted simulator input. No commitment, digest
recalculation, or adversarial attestation is required. Given the same snapshot,
the simulator must deterministically produce the same report.

### Gate transfers

The simulator evaluates the frozen program in topological order and records a
state for every `ProgramWireId`.

For `MulPrivate`, let `D_rhs` be the typed RHS decomposition, let `E_rhs` be its
helper error bound, and let `V_{rhs,2}` be its doubled variance. Exponent-LUT
validates that the semantic left action is either zero or a signed monomial,
so its action norm is at most one. Therefore

\[
  E_{out}
  \le \gamma_1(D_{rhs})E_{lhs}+E_{rhs},
\]

\[
  V_{out,2}
  = \gamma_2^2(D_{rhs})V_{lhs,2}+V_{rhs,2}.
\]

For `MulPublic`, let `D_A=G^{-1}(A_private)` be the cached decomposition of the
private operand's public matrix and let `E_public` and `V_{public,2}` describe
the public-value encoding. Because the public plaintext is a signed monomial,

\[
  E_{out}
  \le E_{private}+\gamma_1(D_A)E_{public},
\]

\[
  V_{out,2}
  =V_{private,2}+\gamma_2^2(D_A)V_{public,2}.
\]

`V_{private,2}` is the doubled-error variance already attached to the private
operand's `ProgramWireId` immediately before this gate. It is produced by the
preceding gate transfers and includes every inherited and fresh contribution
accumulated on that wire. It is not the variance of the private key and it is
not a new sampler parameter.

These equations correspond exactly to

\[
  c_{out}=c_{public}D_A+x c_{private}.
\]

For `Add`, worst-case bounds add:

\[
  E_{out}\le E_{lhs}+E_{rhs}.
\]

Average `Add` does not classify or exploit independence. It always uses the
same coherent-sign principle as the worst-case channel. To keep the
calculation in exact squared units without square roots, a family of `r`
contributions uses the Cauchy bound

\[
  V_{sum,2}\le r\sum_{i=0}^{r-1}V_{i,2}.
\]

For binary `Add`, this is `2*(V_lhs,2 + V_rhs,2)`. For a structured family
reduction, apply the `r`-ary formula once so the result does not depend on the
shape of the balanced reduction tree. No correlation class, independence
ledger, or application-specific `Add` rule is part of the simulator.

For `Lut`, let `Sigma_L` be the one-layer automorphism set. For branch `sigma`,
let `D_C,sigma,L` be the input-error action, let `B_{C_alpha,L}` bound the LUT
coefficient/helper error that is added directly, let `D_A,sigma,L` be the
action on the automorphism helper error, and let `B_{h,sigma}` bound that
helper error. Before applying the common-cutoff assumption, the branch bound
is

\[
  E_{\sigma,L}
  \le
  \gamma_1(D_{C,\sigma,L})E_{in}
  +B_{C_{\alpha,L}}
  +\gamma_1(D_{A,\sigma,L})B_{h,\sigma}.
\]

The symbol in `B_chi` is the Greek letter chi, denoting the helper-error
distribution cutoff; it is not a polynomial `X`. Under the required common
cutoff assumption,

\[
  B_{C_{\alpha,L}}=B_{h,\sigma}=B_\chi.
\]

The first helper error enters the branch by direct addition, so its action gain
is exactly one. This is the source of the `1` in the combined expression; the
number one is not part of the definition of `B_chi`. Summing the branches gives

\[
  E_{Lut}
  \le
  \sum_{\sigma\in\Sigma_L}
  \left(
    \gamma_1(D_{C,\sigma,L})E_{in}
    +\left(1+\gamma_1(D_{A,\sigma,L})\right)B_\chi
  \right).
\]

Under independent automorphism-helper sampling, the average transfer is

\[
  V_{Lut,2}
  =
  \sum_{\sigma\in\Sigma_L}
  \left(
    \gamma_2^2(D_{C,\sigma,L})V_{in,2}
    +\left(1+\gamma_2^2(D_{A,\sigma,L})\right)V_{\chi,2}
  \right).
\]

The two-input LUT API needs no separate simulator formula: the simulator sees
the lowered `MulPrivate` followed by `Lut` and applies these transfers in that
order.

More generally, compound builder APIs never register compound noise-transfer
functions. The simulator visits only the lowered frozen gates. `lut2` is
derived from `MulPrivate` followed by `Lut`; `select` is derived from each
branch's `MulPrivate` followed by `MulPublic`, followed by `Add`. Compound
operation names may group already-derived gate rows in the report, but they
must not change the arithmetic or supply an alternative bound.

### One-hot PRF block

The equations in this subsection are the algebraic expansion of the basic
gate transfers for logging and review. They are not implemented as a separate
one-hot transfer. The final reduction uses the same coherent average `Add`
rule as every other family reduction Add.

For PRF block `j`, only the actual non-padding slots are included. For active
slot `i`, define

\[
  D_{C,j,i}=G^{-1}(C_{j,i}),
  \qquad
  D_{f,j,i}=G^{-1}(A_{f,j,i}).
\]

The `MulPrivate` stage gives

\[
  E_{f,j,i}
  \le \gamma_1(D_{C,j,i})E_j+B_\chi,
\]

and the following `MulPublic` gives

\[
  E_{g,j,i}
  \le E_{f,j,i}
     +\gamma_1(D_{f,j,i})E_{V,j,i}.
\]

The block output is the `Add` reduction

\[
  E_{j+1}\le\sum_{i\in I_j}E_{g,j,i}.
\]

In the average channel,

\[
  V_{f,j,i,2}
  =\gamma_2^2(D_{C,j,i})V_{j,2}+V_{\chi,2},
\]

\[
  V_{g,j,i,2}
  =V_{f,j,i,2}
   +\gamma_2^2(D_{f,j,i})V_{V,j,i,2},
\]

\[
  V_{j+1,2}
  \le |I_j|\sum_{i\in I_j}V_{g,j,i,2}.
\]

The report records both stages for every block: inherited state, selector
helper, value-encoding contribution, reduction total, output bits, and bit
growth. It does not report the removed grouped-reduction or terminal-LUT
stages.

### Value readout

Let

\[
  \tau_1=\gamma_1(T'),
  \qquad
  \tau_2^2=\gamma_2^2(T'),
\]

where these are the actual convolution-action norms of `T'`, not its modulus
representative. Public multiplication gives

\[
  E_T\le\tau_1E_{acc},
  \qquad
  V_{T,2}=\tau_2^2V_{acc,2}.
\]

The identity trace branch contributes exactly `E_T` and `V_{T,2}`. For every
non-identity `sigma`, use its concrete helper actions:

\[
  E_{tr}
  \le E_T+
  \sum_{\sigma\ne id}
  \left(
    \gamma_1(D_{C,\sigma})E_T
    +\left(1+\gamma_1(D_{A,\sigma})\right)B_\chi
  \right),
\]

\[
  V_{tr,2}
  = V_{T,2}+
  \sum_{\sigma\ne id}
  \left(
    \gamma_2^2(D_{C,\sigma})V_{T,2}
    +\left(1+\gamma_2^2(D_{A,\sigma})\right)V_{\chi,2}
  \right)
\]

under the same independent automorphism-helper assumption as `Lut`.

Let `D_norm=G^{-1}((2n')^{-1}G)`. Normalization is a deterministic public
right action and introduces no helper error:

\[
  E_{read}\le\gamma_1(D_{norm})E_{tr},
  \qquad
  V_{read,2}=\gamma_2^2(D_{norm})V_{tr,2}.
\]

If the evaluator implementation is later changed to use a relation-bearing
normalization helper, its fresh error must be added here in the same change.

### Routing and refresh

For CRT slot `t`, set

\[
  \kappa_t=q/q_t.
\]

Construct the actual balanced-decomposition matrices for the state scaling,
mask routes, and fresh-error routes. Their gains are computed from the
decompositions of the concrete targets; in particular, the simulator never
uses raw `kappa_t` as an error gain.

Let `R_m,1` and `R_e,t,1` be the summed worst-case route gains, and let
`E_d,t` be the decoder helper contribution. The pre-refresh operational error
excluding the nominal mask is

\[
  F_t^{worst}
  =\gamma_1(D_{\kappa,t})E_{state}
   +R_{m,1}E_{read}
   +R_{e,t,1}E_{read}
   +E_{d,t}.
\]

For average simulation, the mask and fresh routes reuse the same readout wire.
They must not be treated as independent. Form the combined linear action
`D_m + D_e,t` first and compute its actual squared gain

\[
  R_{m+e,t,2}^2=\gamma_2^2(D_m+D_{e,t}).
\]

Let `V_{m,digit,2}` and `V_{e,t,digit,2}` be the centered base-`P` digit
variances after their actual route actions, and let `V_{d,t,2}` be the decoder
variance.
Then

\[
  V_{t,2}
  =\gamma_2^2(D_{\kappa,t})V_{state,2}
   +R_{m+e,t,2}^2V_{read,2}
   +V_{m,digit,2}
   +V_{e,t,digit,2}
   +V_{d,t,2}.
\]

All digit variances are computed from the actual centered base-`P` support;
they are not inferred from the maximum digit magnitude.

The one joint tail event covers all inspected mask and fresh-error
coordinates. Its event count is derived from the actual transcript topology,
including `2*ell_beta` component columns. It is not multiplied by an
uninspected adversarial input domain. With exact rational `z_joint^2` and tail
correction `b_tail`, define

\[
  F_t^{avg}
  =\left\lceil
    \frac{\sqrt{z_{joint}^2\,4^{b_{tail}}V_{t,2}}}{2}
   \right\rceil.
\]

For the selected authority, write `F_t=F_t^worst` or `F_t=F_t^avg`. Define

\[
  M_m=P^{d_m},\quad B_m=M_m-1,
  \qquad
  M_e=P^{d_e},\quad B_e=M_e-1.
\]

Every CRT slot must satisfy the strict rounding condition

\[
  2(B_m+F_t)<\kappa_t,
\]

and the fresh-error condition

\[
  B_e<q_t.
\]

The latter is checked separately for every prime factor, not only against the
smallest factor cached elsewhere. Mask coverage and smudging require

\[
  M_m\ge Q,
\]

\[
  M_m\ge 2^{\lambda_{mask}+1}D\max_tF_t,
\]

where `D` is the actual exposed transcript-coordinate count. Equality is
accepted for these two non-strict conditions.

### Implementation replacement

Delete the old simulator logic tied to `Unary`, `Binary`, `OneHot`, grouped
mod-`Q_L` reductions, and a terminal rounding LUT. In particular, replace
`fixed_fuse_transfer`, `fixed_lut_transfer`,
`monomial_one_hot_transfer`, `simulate_sparse_prf`, and their average variants
with gate visitors for the frozen IR plus the explicit PRF-block and readout
reports above.

The PRF-block report is assembled from the gate rows emitted by those visitors;
it does not rerun a second PRF-specific recurrence. Any disagreement between a
block summary and its constituent gate rows is an internal simulator
error.

`ExponentLutNoiseParameters::dense` must not be an acceptance path. Every
acceptance gain comes from a concrete decomposition target or from a validated
structural bound produced by the identical decomposition algorithm. The
simulator must not call `mxx-noise-simulator::right_action_gain` or construct a
generic `SimulationRequest`.

The simulator models the evaluator stages directly:

```text
MulPrivate
MulPublic
Add reduction
T' multiplication
trace
normalization
base-P routing
public half-offset
decoder application and threshold checks
```

The public half-offset contributes exactly zero noise. Worst and average
authorities remain explicit and are never substituted silently.

Mandatory hard checks in both modes include:

- the fresh-error bound is strictly smaller than every CRT prime factor;
- a tight preimage cutoff below the default sampler cutoff is rejected;
- mask-domain coverage is sufficient;
- all dimensions and decomposition bounds match imported artifact schemas;
- sparse-LWR security and DCRT/BGG RLWE security meet the selected target; and
- the selected correctness authority passes its rounding and smudging tests.

Existing useful log fields are retained. Add per-block and per-stage inherited,
fresh, total, and bit-growth fields rather than replacing the current report
with a smaller summary.

## Parameter selection

Derive `Q` and `P`; do not accept an independently inconsistent `q_l`, LUT
width, or refresh base. Security bits remain an explicit search input rather
than a production constant. Searches may evaluate 100-bit and 128-bit profiles
separately, and reports must identify which target was used.

Delete every Phase-1 assumption that the LWR output modulus is two. In
particular, remove the `p == 2` checks from sparse-LWR candidate derivation,
`SearchConfig` validation, Phase-1 tuple selection, and average-case evidence.
The replacement validates the derived native parameters

\[
  P=2p',\qquad 2\le P\le Q,\qquad P\mid Q,
\]

and passes that same `P` to the estimator error interval, base-`P` routing,
mask/fresh digit counts, reports, and checkpoint identity. Average-case
uniformity is established from the typed program, complete centered base-`P`
support, and concrete decomposition snapshot; it must not use `P == 2` as a
proxy. The parameter search must therefore accept reviewed evidence for, among
others, `P = 16` and `P = 32` without a compatibility path for the old binary
profile.

The existing `crt_bits <= 32` check is only a test-harness search restriction,
not a DCRT primitive invariant. Replace it with validation against the
per-prime width supported by `DCRTPolyParams` and the selected backend. When
both CRT and gadget-base widths are supplied explicitly, the search grid must
contain that exact pair (for example `(50, 25)`) rather than silently replacing
it with the legacy `32/30/28` grid.

The native-mod-Q proposal's example

\[
  n=2^{16},\quad m=5,\quad Q=2^{12},\quad P=2^6,\quad p'=32
\]

is a search starting point, not a hard-coded default. The search recomputes
the LWR dimension, sparse weight, PBC parameters, CRT modulus, gadget base,
digit counts, noise, and both security estimates. No implementation-level
maximum for `log2(q)` is added.

## Required code changes

### `mxx-bgg`

- keep `BggPublicKeyWire` and `BggEncodingWire` semantically unchanged;
- expose multiplication paths that accept an explicit typed `Preimage` rather
  than decomposing internally;
- allow an online caller to attach an already-preprocessed ordinary output
  public key instead of recomputing its matrix arithmetic;
- reuse one decomposition for vector and public projection in preprocessing;
  and
- reject missing decomposition artifacts in cache-required mode.

### `mxx-exponent-lut`

- add `ExpLayout` and replace automorphism enumeration;
- replace the frozen gate set and public program builder APIs;
- replace public-value monomial families with ordinary public BGG+ wire
  families;
- delete the old one-hot gate and express selection structurally;
- replace the PRF reduction plan with native accumulation;
- add `ValueReadout`;
- generate the public program trace, ordinary public artifacts, and only the
  required decomposition cache artifacts;
- import cache artifacts lazily into online evaluation;
- update routing and half-offsets;
- replace the old PRF noise recurrence; and
- update storage and benchmark reports;
- remove the binary-output assumptions from `SparseLwrCandidate::derived`,
  `SearchConfig::validate`, `select_sparse_lwr_profile`, and average-case
  acceptance evidence, validating derived `P` consistently instead; and
- remove the parameter-search-only 32-bit CRT ceiling and preserve explicitly
  requested `(crt_bits, base_bits)` pairs.

### `mxx-runtime`

- reuse `ArtifactType::Preimage`, `ArtifactStore`, and existing typed
  encoding/decoding;
- add a filesystem-backed `ArtifactStore` and matching session support for
  production and benchmark runs;
- preserve lazy family-member loading and release artifacts after their wave;
  and
- report artifact bytes separately from peak resident bytes.

### `mxx-primitives`

- reuse compact bounded preimage serialization and `mul_small_rhs`;
- stream DCRT/NTT expansion by columns for quadratic-size decompositions;
- retain full logical shapes and coefficient bounds; and
- use the existing GPU residency setting fixed at context creation.

## Deleted APIs and data

Delete rather than deprecate:

- `ProgramGate::Unary`, `ProgramGate::Binary`, and `ProgramGate::OneHot`;
- public-value-family declarations that store clear monomials outside
  `BggEncodingWire::plaintext`;
- all LUT type, helper, function, artifact, and report names containing
  `Flat`;
- LUT mode enums, sequential indices, sequential helper paths, and readout
  split parameters;
- `SparseLwrReductionPlan`;
- delayed reduction groups and mod-Q reduction LUT programs;
- terminal rounding LUT programs and their specialized helper bundles;
- generic-simulator conversion and fallback code; and
- fixture-only compatibility paths and constructors.

## Implementation order

1. Add `ExpLayout`, monomial construction, and automorphism-index tests.
2. Rename and update LUT helpers and coefficient construction.
3. Replace the frozen gate set and knowledge validation.
4. Implement `lut`, `lut2`, and structured `select` expansion.
5. Add explicit-decomposition BGG compilation paths.
6. Add public-program decomposition planning and typed artifact outputs.
7. Add filesystem-backed artifact storage and bounded family-member loading.
8. Replace sparse-LWR accumulation and online input binding.
9. Implement value-form readout and its artifacts.
10. Update base-P routing and public half-offsets.
11. Replace the Exponent-LUT-specific noise model and reports.
12. Update parameter search and storage accounting.
13. Delete all legacy paths and exports.
14. Run the validation gates below, then update benchmark estimates using the
    production artifact-backed path.

## Validation

### Unit tests

Add focused tests for:

- `mono_m` multiplication and native wrap for `m in {0,1,2}`;
- distinct odd automorphism indices of the expected cardinality;
- bit-for-bit `m=0` behavior;
- one-input LUT correctness for small complete domains;
- `lut2` equivalence to `MulPrivate` followed by `Lut`;
- `select` equivalence to `MulPrivate`, then `MulPublic`, then `Add`;
- rejection of `MulPublic` with missing public plaintext;
- plaintext knowledge propagation;
- label-independent public matrices;
- typed public `Preimage` artifact export/import;
- file-backed scalar and family artifact round trips;
- bounded wave loading without full-family materialization;
- integer digits after half-offset addition; and
- stage-by-stage agreement between evaluator operations and simulator terms;
- Phase-1 selection and both noise authorities for non-binary `P`, including
  `P = 16` and `P = 32`; and
- exact `(crt_bits, base_bits) = (50, 25)` propagation from configuration into
  every enumerated DCRT candidate.

### Tiny end-to-end test

Use at least

\[
  n=32,\quad m=1,\quad n'=16,\quad Q=32,\quad p'=2
\]

and test all `z in Z_Q`. Supply random valid value encodings, execute the full
PRF and readout, and compare decryption with

\[
  f(z)-\frac12.
\]

Run the complete refresh path and verify the final decrypted attribute, strict
fresh-error condition, and selected noise authority.

### GPU tests

GPU tests use the same artifact-backed production functions and run outside the
sandbox. They verify repeated cache artifact loading, column streaming,
multi-GPU placement, format consistency, deterministic results across wave
sizes, and stable peak residency. Tests must not add device-wide synchronization
or serialize unrelated GPU work.

### Final acceptance

Implementation is complete only when:

- native wrap and value readout are functionally correct;
- all LUT evaluation uses the single implementation and no obsolete naming or
  mode remains;
- the public matrix is identical across labels;
- online evaluation performs no public-key decomposition or public-program
  arithmetic in cache-required mode;
- only decomposition preimages are counted as cache artifacts;
- production cache payloads are artifact-backed and bounded in RAM and VRAM;
- fresh error is strictly smaller than every CRT prime;
- sparse-LWR and DCRT/BGG security pass the selected security target;
- rounding, smudging, cutoff, and domain checks pass under the declared noise
  authority;
- logs retain existing useful fields and expose the new stage breakdown; and
- benchmark estimates measure the same store/load/compute/release lifecycle as
  production.
