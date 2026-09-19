# DSL FHE crate implementation plan

Status: implemented and validated on CPU and GPU.
Baseline: main commit `daf4421ab5b742730b77eef59b24b909a6bde171`.

## Scope and architecture

Implement common FHE traits, Ring Regev encryption, Ring-GSW encryption and
external product, and leveled BGV including addition, multiplication,
relinearization, modulus switching, SIMD encoding/decoding, and rotations.
Bootstrapping is explicitly out of scope and has no placeholder API.
Modulus switching must operate directly on CRT limbs without reconstructing
ciphertext coefficients as BigIntegers, including inside runtime primitives.

All cryptographic algorithms are graph builders written with `mxx-dsl`, including
key generation, sampling, encoding, encryption, evaluation, key switching, and
decryption. The graph is validated by `mxx-ir-core` and executed by `mxx-runtime`.
Do not first implement a separate native FHE evaluator and wrap it with a DSL
node. Native polynomial operations remain the existing runtime backend's job.

Add `mxx-fhe` at `crates/fhe`. Normal dependencies are `mxx-dsl`, `mxx-ir-core`,
`mxx-primitives`
(for the existing parameter type), and ordinary utility libraries. Runtime is
initially a unit-test dependency. Use its existing execution and artifact APIs;
no new public executor wrapper is planned. No application crate or circuit-gadget
dependency is needed.
Update workspace membership and `docs/architecture.md` during implementation.

Host code may validate public parameters, calculate public roots/twiddles and
modular inverses, construct schemas, choose a fixed schedule, bind runtime inputs,
and inspect final outputs. It must not compute sample values, transform user
plaintexts, manipulate concrete ciphertexts, or decrypt outside the graph.
Initial correctness validation uses `CpuDcrtBackend`. Reusing backend-independent
DSL does not by itself establish GPU performance or complete GPU support.

Implementation files under `crates/fhe/src`: `lib.rs`, `params.rs`,
`utils.rs`, `ring_gsw.rs`, and `bgv.rs`. Noise tracking lives with each scheme; test-only utilities use `#[cfg(test)]`.
BGV batching and rotations are implemented directly in `bgv.rs`.
`utils.rs` also provides runtime fixtures behind `#[cfg(test)]`. Traits and schemas stay with their
owners instead of introducing separate modules for small declarations.
Integer artifact outputs use the existing runtime artifact boundary with a
canonical signed byte encoding. Tests explicitly materialize persisted families
before checking plaintexts; private/public output annotations remain intact.
FHE artifacts stay in `MemoryArtifactStore` or are supplied directly as runtime
inputs. FHE execution does not write artifacts to files. Separate protocol stages
share the in-memory store and its manifests.

## Existing capabilities and exact representation

- `crates/dsl/src/value.rs:3` defines `GraphValue` and `GraphValueSchema`.
  Follow `crates/bgg/src/public_key.rs:13` for composite application wires.
- `crates/dsl/src/lib.rs:617`, `:623`, and `:638` provide uniform residue,
  small interval, and bounded Gaussian sampling nodes.
- `crates/dsl/src/lib.rs:1014` exposes `Mat::ring_automorphism`;
  `crates/runtime/src/backend/poly.rs:951` executes it on native polynomials.
- `crates/dsl/src/lib.rs:1081` provides regular gadget decomposition and `:562`
  constructs its gadget. `Preimage::mul_small_rhs` at `:1322` multiplies an
  ordinary matrix by the bounded decomposition without expanding it first.
- `crates/dsl/src/lib.rs:1122` extracts coefficients; `:596` packs canonical
  coefficient bits. `crates/dsl/src/integer.rs:49` and `:55` provide division
  and nonnegative remainder. `crates/runtime/src/executor.rs:1431` confirms
  Euclidean division/remainder, including negative inputs.
- `crates/primitives/src/matrix/dcrt_poly.rs:200` currently implements
  `reduce_modulus` by exporting full coefficients. Although its mathematical
  result is correct, it violates the requested RNS-only modswitch path. Replace
  this implementation with direct copying/selecting of existing CRT towers;
  keep its public ring-reduction semantics and the existing DSL node.
- `crates/dsl/src/control.rs:4` and `:43` provide structural `parallel` and
  `iterate`; a loop body is stored once. `iterate` requires an unchanged state
  schema. Do not carry a changing-modulus ciphertext through one such loop.
- `crates/bgg/src/test_utils.rs:38` already demonstrates graph validation and
  execution with `cpu_backend`, `MemoryArtifactStore`, and `SamplingMode::Fresh`.
- `crates/primitives/native/ExactBasis.cc:221` constructs cyclotomic order `2*N`.
  Use `R_Q = Z_Q[X]/(X^N+1)`, power-of-two `N >= 2`. Correct the conflicting
  `X^N-1` documentation in `crates/primitives/src/poly/mod.rs` when implementing.
- The old Ring-GSW circuit code is not a DSL-native FHE interface. Its helper in
  `crates/gadgets/src/circuit_gadgets/fhe/ring_gsw_nested_rns.rs:243` uses the
  opposite sign. The user permits sign adjustments: use `phi_s(a,b)=b-s*a`
  consistently here and convert internally if an adapter is later requested.

Use the existing **regular exact CRT gadget**, not the prior proposal's native
full-integer radix decomposition. Set `dropped_moduli=0` at every level. If the
ordered basis is `(q_0,...,q_l)`, define CRT idempotent `E_i` by residues 1 at
limb i and 0 elsewhere, and use

```text
B = 2^base_log
w_l = ceil(max_i bit_length(q_i) / base_log)
L_l = (l+1)*w_l
g[i,j] = E_i * B^j mod Q_l, ordered limb-major
x = sum_{i,j} g[i,j] * D[i,j](x) mod Q_l
```

Each digit is a small signed integer polynomial lifted consistently to all CRT
limbs. The implementation and column order are in
`crates/primitives/native/ExactBasis.cc:260` and
`crates/primitives/src/matrix/dcrt_poly.rs:559`. Derive widths from the registered
backend layout and validate them. Use `Mat::decompose`, not `small_decompose`:
the latter has different limb semantics. All GSW and key-switch formulas below
use these same gadget weights, denoted `g_j` with one flattened index.

## Minimal data model and reuse decisions

Reuse existing types rather than add a nominal wrapper for every mathematical
object. `Mat`/`MatType` already carry Q, N, and shape;
`DCRTPolyParams` already owns the ordered CRT basis, gadget base/width, and ring
capability validation. Tuples and `Vec<T>` already implement `GraphValue` and
`GraphValueSchema` in `crates/dsl/src/value.rs:352-441`. `Family<T>` is useful
for homogeneous repeated values; `Vec<Mat>` can carry different per-entry moduli.

| Object | Representation and invariant |
| --- | --- |
| Ring Regev plaintext | Scalar `Mat` in R_q with declared centered coefficient bound |
| BGV plaintext | `Family<Int>`, 1 to N slots on input (zero-padded), N slots on output, in fixed two-row order |
| Ring-GSW multiplier plaintext | Scalar `Mat` in R_q, with declared centered coefficient bound |
| Secret key | `Mat`, shape (1,1), sampled at the top modulus |
| Ring Regev ciphertext | `RingCiphertext` with a/b parts each (1,1), noise and plaintext bounds |
| BGV public key | `Mat`, shape (2,1), rows (A,B) |
| Ring-GSW ciphertext | The same `RingCiphertext` record with a/b parts each (1,2L) |
| Relinearization or rotation key | `Mat`, shape (2,L), rows (A_j,B_j) |
| Decomposed polynomial/ciphertext | Existing `Preimage` from `Mat::decompose` |
| Keygen output | Existing tuple `(secret, encryption_key)` |
| Multiple evaluation keys | Caller-owned `Vec<Mat>`; optional maps by level for relinearization, by (level,index) for rotation |
| Level | `usize`, inferred from the exact matrix modulus and configured chain |
| Rotation request | Existing method arguments `level: usize, steps: i32` |
| Artifact identity/confidentiality | Existing `ProductionId`, artifact handles, `ArtifactConfidentiality` |

BGV encrypt/decrypt always exchange slots. Private encode/decode helpers handle
coefficient conversion inside the graph; callers do not choose an encoding mode.
N and t come from the parameter object. Inputs have a concrete length from 1 to N
and are reduced modulo t, with unused slots set to zero. Decryption returns N
slots, including positions populated by rotation. A single integer is the
one-slot case; no input-length metadata or separate scalar type is added.

The parameter and ciphertext records reuse existing DSL values and schemas:

```rust
pub struct FheCommonParams {
    ring: DCRTPolyParams, // Top Q, N, ordered CRT basis, gadget base and width.
    secret_range: SampleRange, // Existing IR type; [0,1] or [-1,1].
    error_sigma: f64,
    error_cutoff: BigUint,
}
pub struct RingGswParams {
    common: FheCommonParams,
    scale: BigUint, // Ring Regev's explicit integer message scale.
    plaintext_bound: BigUint, // Declared centered coefficient bound for fresh inputs.
}
pub struct BgvParams {
    common: FheCommonParams,
    plaintext_modulus: u64,
}

pub struct BgvCiphertext {
    components: Mat,         // Shape (2,1) or (3,1).
    correction_factor: u64,   // Compile-time public unit modulo t.
    noise_bound: BigUint, // Coefficient bound on e in v = centered(v mod t) + t*e.
}
pub struct BgvCiphertextSchema {
    components: MatType,
    correction_factor: u64,
    noise_bound: BigUint,
}

pub struct RingCiphertext {
    a: Mat,
    b: Mat,
    noise_bound: BigUint,
    plaintext_bound: BigUint,
}
// One shared record/schema; checked shapes distinguish the aliases.
pub type RingRegevCiphertext = RingCiphertext; // Each part (1,1).
pub type RingGswCiphertext = RingCiphertext; // Each part (1,2L).
```

`BgvCiphertext` is a graph-value record, and `BgvCiphertextSchema` is its
required DSL schema. Retaining f with the components prevents forgetting the
factor after multiplication/modswitch. Shape distinguishes ordinary and quadratic
ciphertexts; no `BgvQuadraticCiphertext` type, degree field, or state enum is needed.
Define the phase by evaluating the component column as descending coefficients
in the variable `-s`: ordinary rows `(a,b)` give `b-s*a`; quadratic rows
`(a1*a2, a1*b2+b1*a2, b1*b2)` give the product phase. Public methods check the
accepted shape: `mul` and rotation require two rows, `relinearize` requires three.
A generic decryption subgraph can Horner-evaluate either accepted shape.

Use `MatType` inside the new schema; do not reproduce its modulus, dimension,
and shape fields. There is no new common RLWE sample, key-domain record, Level,
plaintext type, slot layout type, secret distribution enum, Gaussian parameters,
gadget parameters, per-level parameters, modulus-switch constants struct, or
rotation-key request struct. Share FheCommonParams between the two parameter objects.
Keep scheme-specific behavior in their methods rather than adding a scheme tag
or inactive optional fields to parameters. Plaintext modulus t belongs only to
BgvParams. RingGswParams accepts and returns scalar polynomial `Mat` values in
the ciphertext ring R_q. It retains explicit scale and a declared bound on
centered plaintext coefficients; arbitrary full-range R_q values cannot be
recovered exactly under noisy scaled encryption. Negative decoded coefficients
use canonical unsigned residues modulo q. No separate plaintext modulus is added.

Derive the lower rings through `DCRTPolyParams::select_modulus`, rejecting
`dropped_moduli != 0`. Compute scale, gadget widths, SIMD root/order, and public
modswitch inverses from these parameters. Precompute repeated constants once
per built subgraph using ordinary local values/arrays; do not expose a new public
cache type. `BgvParams::new` validates the batching condition because SIMD is
the default and only public plaintext interpretation.
`RingGswParams::new(common, scale, plaintext_bound)` and
`BgvParams::new(common, plaintext_modulus)` validate their own fields and return
`Result<Self,FheError>`. These parameter objects themselves implement the graph
builders; there is no additional RingGsw or Bgv wrapper.

Validate distinct compatible primes, ordered prefix bases, N, coprimality with t,
exact gadget dimensions, ciphertext shapes, and correction factors. Match the
runtime's registered CRT order explicitly: Q alone does not identify tower order.
The common parameters need a normal dependency on `mxx-primitives`, but all
cryptographic execution still occurs through DSL and runtime, not these params.
Do not import BGG-specific sampler layout types merely to reuse a few fields.

Remove `ParamsId`, `KeyDomainId`, and `CiphertextDomain`. Existing `SpecHash`,
`ProductionId { spec_hash, execution_nonce }`, and artifact handles identify
productions. They do not prove that two matrices use the same secret key.
Bind keys and ciphertexts through explicit existing artifact handles; do not
promise automatic wrong-key rejection for otherwise valid raw Mat inputs.
A caller-owned map can organize keys without becoming a new GraphValue record;
pass the selected key Mat to the corresponding subgraph. For structured repeated
inputs, use the existing tuple/Vec GraphValue implementations.

The correction factor f and plaintext modulus t are different quantities:
`phase = f*m + t*e` modulo Q, with centered representatives and the usual bound.
At encryption f=1; multiplication multiplies f values and dropping p updates
f to f*p^(-1) modulo t. Replacing f by t loses this information and gives a
noninvertible value modulo t. For example t=17, p=13 changes f=1 to f=4 while
t remains 17. Keep `correction_factor` on the ciphertext and `plaintext_modulus`
on BgvParams. A special chain with every p=1 modulo t could avoid this metadata,
but the current design does not impose that restriction.

BgvCiphertextSchema is a DSL construction schema, not another cryptographic
ciphertext. `GraphValue::Schema` must implement
`GraphValueSchema<Value=Self>` (`crates/dsl/src/value.rs:3-13`) so Subgraph can
create placeholders and reconstruct the record while preserving public f.
MatType alone reconstructs Mat and carries no f. A tuple `(Mat,Int)` alias could
eliminate both custom structs but would move f into runtime data and change the
current validation/arithmetic contract. Keep the named record plus its schema
for now; the Ring Regev/GSW pair aliases already use existing tuple schemas and
need no new schema type.

Public f is known when constructing a fixed graph; preserve it in the custom
GraphValueSchema. This schema alone does not add metadata to a Matrix artifact.
For staged export, emit f as a separate existing public Int artifact next to
the component Matrix, and validate it against the next graph's specialized
schema at binding. N, t, and key provenance come from the declared parameters
and explicit artifact connections. Keep secret key and decrypted outputs private.
No new serialized ciphertext header, identity system, or artifact container is
required. Runtime-selected levels/factors/rotation keys remain outside this API.

## One common graph-building trait

```rust
pub trait FheScheme {
    type Plaintext: GraphValue;
    type Ciphertext: GraphValue;
    type MulRhs: GraphValue;
    type EvaluationKey;

    fn common_params(&self) -> &FheCommonParams;
    fn keygen(&self) -> Result<(Mat, Mat), FheError>; // (secret, encryption key)
    fn encrypt(&self, key: &Mat, plaintext: &Self::Plaintext)
        -> Result<Self::Ciphertext, FheError>;
    fn decrypt(&self, secret: &Mat, ciphertext: &Self::Ciphertext)
        -> Result<Self::Plaintext, FheError>;
    fn add(&self, lhs: &Self::Ciphertext, rhs: &Self::Ciphertext)
        -> Result<Self::Ciphertext, FheError>;
    fn mul(&self, lhs: &Self::Ciphertext, rhs: &Self::MulRhs,
           eval_key: &Self::EvaluationKey)
        -> Result<Self::Ciphertext, FheError>;
}
```

Implement this trait directly on RingGswParams and BgvParams:

| Implementation | Ciphertext | MulRhs | EvaluationKey | mul behavior |
| --- | --- | --- | --- | --- |
| RingGswParams | RingRegevCiphertext | RingGswCiphertext | `()` | external_product(rhs,lhs) |
| BgvParams | BgvCiphertext | BgvCiphertext | `Mat` | multiplication then relinearization |

The associated types add no runtime records. They express the actual asymmetric
Ring Regev-by-GSW multiplication instead of pretending that two Ring Regev
ciphertexts support the same operation. Use `mul(&c,&C,&())` for Ring-GSW and
`mul(&c1,&c2,&relinearization_key)` for BGV. `external_product` can remain an
inherent algorithm used by mul; do not duplicate its implementation.

RingGswParams::keygen returns two handles to the same secret key. BgvParams
returns a secret Mat and public-key Mat. Generate evaluation material only when
needed using `relinearization_key(secret,level)`,
`rotation_key(secret,level,steps)`, and `row_swap_key(secret,level)`.
They share a private helper taking the key-switch target polynomial.

No operation-specific traits are needed. Keep scheme-specific graph builders
as inherent methods: RingGswParams::encrypt_gsw/external_product and
BgvParams::mul_unrelinearized/relinearize/mod_switch_to,
match_correction_factor and rotate_rows/swap_rows. Slot encoding/decoding is private.
Methods accept the required key Mat directly rather than a key registry.
All return graph handles and compose existing Subgraph definitions. No new
keyset container or execution wrapper is introduced. Use existing DSL/runtime
errors at their boundaries and add FHE-specific validation errors only as needed.
Raw Mat shape cannot validate a key's cryptographic target.

## Coefficient algorithms expressed in DSL

Implement small reusable graph builders:

- Extract all canonical coefficients with `Mat::coefficients()` in one runtime operation.
- Center with comparison and `select`: subtract modulus if `2*x > modulus`.
- Canonicalize with `Int::rem`, whose result is nonnegative for a positive modulus.
- Import coefficient wires with `Ring::from_coefficients(&Family<Int>)`, the DSL
  equivalent of `Poly::from_biguints`. The runtime reduces inputs modulo q.
  `Ring::from_evaluations` similarly imports native-order evaluation slots;
  `Mat::evaluations()` exports them. These reuse native primitive transforms.
- Lift plaintexts to R_Q by centered conversion and packing; reducing a sampled
  secret to a lower Q uses exact divisor `Mat::reduce_modulus`, with no scaling.

Plaintext encoding and decryption can use this explicit coefficient path:
`Mat::reduce_modulus(t)` is invalid when t does not divide Q. Ordinary
`Mat::modulus_switch` also has different semantics. **Modswitch must not use
these extraction/Int/packing helpers.** It uses the RNS subgraph below and requires
the generic limb primitives to be implemented first. Coefficient extraction at
encoding/decryption boundaries may still be expensive; this is not a claim of
optimized RNS/GPU performance for every operation.

## Generic RNS operations needed by the DSL

Keep the BGV algorithm in `mxx-fhe` as a composition of generic DSL nodes:

1. Existing `Mat::reduce_modulus(target)` selects an exact ordered subset of
   source CRT towers. Fix the CPU implementation to avoid `coeffs()`,
   `exact_basis_coefficients`, and `CRTInterpolate`; preserve native NTT format
   and tower roots. Audit any other selected backend for the same requirement.
2. Add `Mat::centered_rebase(target_modulus)` for a **single-limb source**. It
   interprets each coefficient in `[-p/2,p/2]` and writes its residues directly to
   the destination basis, which need not divide p or be larger than p. Reject
   multi-limb sources initially: this operation never reconstructs a value
   modulo a product of primes. Preserve matrix shape and ring dimension.

Do not add a CRT scalar type, constructor, or IR constant variant. Existing
`Ring::polynomial([IntExpr::constant(k)])` represents a public constant polynomial
at any configured modulus (`crates/dsl/src/lib.rs:585`). For example,
`k=p^(-1) mod Q'` has the required residue `p^(-1) mod q_i` in every target limb.
Compute k once from public parameters; its import reduces a known public constant
to limbs and does not reconstruct any ciphertext coefficient. This eliminates
the earlier proposed `crt_scalar` operation without changing RNS evaluation.

For centered rebase, convert only the single source limb to coefficient format.
"Centered" specifies the integer represented by a residue, not a signed storage
type. All stored limbs remain unsigned in [0,q_i). A negative value -d is stored
as q_i-d. Across moduli this encoding changes: source p=17 residue16 represents
-1, so at target q_i=13 the result must be12, not16 modulo13=3.
Implement this transfer entirely with unsigned comparisons/subtractions.
If its coefficient is u in [0,p), emit u modulo q_i when u <= p/2, otherwise
emit -(p-u) modulo q_i. Transform destination limbs to evaluation format before
ordinary matrix arithmetic. This uses native unsigned word arithmetic and
double-width modular products, with no BigInteger coefficient intermediate,
floating-point approximation, or CPU host extraction loop over graph values.
The source limb's centering is essential; rebasing each NTT evaluation value
with this rule would implement a different, incorrect map.
Taking the canonical nonnegative representative instead is another possible
algorithm, but it changes the correction and its noise bound. Merely storing
negative values as unsigned residues does not select that different algorithm.
The interval near zero is written [q-d,q) in ordinary unsigned coordinates;
an interval ending at 0 describes a wraparound interval, not an ordered integer
range. A bound d=q/t can describe a particular message/noise range but is not
the centering threshold p/2 for this dropped-limb correction.

`crates/primitives/src/modulus.rs:26` already has centered `modulus_raise`
semantics, but requires a larger target and reconstructs full coefficients.
It cannot be used unchanged here, especially at the final drop where q_i may
be smaller than p. Reuse its mathematical convention; implement the single-limb
unsigned path under the generic conversion primitive rather than introduce
any signed polynomial type. The required new DSL capability is cross-basis
representative conversion, independent of the final method name.

Add the single new centered-rebase operation to `mxx-ir-core` node/type/parameter validation,
serialization and exhaustive graph consumers, `mxx-dsl` builders, and
`mxx-runtime` backend dispatch. Native Rust/C++ implementations belong in
`mxx-primitives`. Extend execution-relation/Lean export support as required by
the existing core architecture; do not silently emit an uninterpreted FHE call.
Process the two ciphertext components as one (2,1) matrix and process all
remaining limbs in one native batch. Reuse existing stream/format conventions
if implementing the GPU path. Unsupported backends report a capability error
and never reconstruct coefficients as a fallback.

Public Q, bounds, and compile-time modular constants may remain BigUint values;
the restriction concerns reconstructing runtime ciphertext coefficient values.
The native application of those constants still performs word-size arithmetic
on each ciphertext limb. No new wire type is needed for centered rebase: it
maps an existing Mat to another Mat.

## Ring Regev and Ring-GSW algorithms

Keygen uses a DSL binary/ternary sample. RingGswParams holds a validated
integer scale Delta>0. Encryption samples uniform a and Gaussian e and builds
`b=s*a+e+Delta*m`, for bounded signed integer coefficient messages m. Decryption
uses DSL coefficient arithmetic:

```text
v = centered(b-s*a mod Q)
m = floor((2*v+Delta)/(2*Delta))
```

Require the intended integer phase to stay within (-Q/2,Q/2) and decoding error
to be strictly below Delta/2. Addition/external-product results must satisfy the
same bounds, including growth of the integer message. There is no implicit
plaintext modulus or modular wraparound decoding in this interface. This keeps
Ring Regev scaling separate from BGV's modulo-t message convention.

For a bounded small polynomial mu, sample A,E of shape `(1,2L)` and form

```text
H = mu * ring.gadget(1, B, L)             // (1,L)
C = vertical_concat(A, s*A+E) + block_diag(H,H) // (2,2L)
c = vertical_concat(a,b)                  // (2,1)
D = c.decompose(B,L)                      // (2L,1), typed Preimage
external_product(C,c) = D.mul_small_rhs(C) // Existing Preimage method: C*D.
```

The method receiver here follows the current DSL API, even though its name is
counterintuitive: `Preimage::mul_small_rhs(self, lhs: Mat)` constructs lhs*self
(`crates/dsl/src/lib.rs:1322`). In contrast,
`Mat::mul_small_rhs(self, rhs: SmallMatrix)` accepts SmallMatrix, not Preimage
(`crates/dsl/src/lib.rs:996`). Thus D.mul_small_rhs(C) is valid today, while
C.mul_small_rhs(D) does not type-check for a Preimage D. No API redesign is
required for FHE; if that shared API is later unified, update its existing
callers together rather than add a special FHE multiplication wrapper.

Build `block_diag(H,H)` with DSL concatenation and zero blocks; multiplying a
(1,1) matrix directly by a (2,2L) matrix would be dimensionally invalid.
Thus the first L columns add `(mu*g_j,0)` and the next L add `(0,mu*g_j)`
to independent zero encryptions. There is no Delta on the GSW diagonal.
The output phase is `mu*(b-s*a)+sum_j digit_j*error_j`, retaining the input
Ring Regev scale. Expose C as its two row matrices (a_part,b_part) and the
result as two (1,1) matrices (a,b); concatenate only for the matrix operation. This adapts the encrypted-zero-plus-gadget/external-product
pattern in the [original TFHE implementation](https://raw.githubusercontent.com/tfhe/tfhe/master/src/libtfhe/tgsw-functions.cpp)
to the repository's exact CRT gadget. It is not Torus32 wire compatibility or
TFHE bootstrapping. Optional `cmux(C,c0,c1)=c0+external_product(C,c1-c0)` is a
small DSL composition for an encrypted constant bit.

## BGV encryption, multiplication, and modulus switching

Use the phase invariant, under the required small-phase bound,

```text
centered(b-s*a mod Q_l) = centered(f*m mod t)+t*e
```

Public key: sample `A,e_pk`, set `B=s*A+t*e_pk`. Encryption samples `u,e_a,e_b`
in DSL and computes `a=A*u+t*e_a`, `b=B*u+t*e_b+m`, with f=1. Decryption
extracts the centered phase and computes coefficients `f^(-1)*phase mod t`,
then evaluates the polynomial modulo t and returns logical SIMD slots.
Here m is the encoded plaintext polynomial obtained from input slot evaluations
modulo t; its coefficients are centered before lifting into R_Q.
Only the inverse of public schema metadata f is precomputed on the host.
The t-scaled BGV error convention differs from Ring Regev's unscaled error.

Addition requires matching matrix type and f, then adds components. Correct
use of the same secret key is a caller obligation supported by explicit graph
connections; it cannot be inferred from Mat shape. Provide
explicit `match_correction_factor(ct,target)` which emits DSL multiplication of
both components by the centered public scalar `target/f mod t`. Record the new
factor and its noise cost. Level/factor mismatches never silently consume levels.

`mul_unrelinearized` emits negacyclic products in R_Q:

```text
d0=b1*b2; d1=-(a1*b2+b1*a2); d2=a1*a2
quadratic.components = vertical_concat(d2,-d1,d0)
f_out=f1*f2 mod t
```

The result uses the same BgvCiphertext type with three rows. `relinearize`
reads these rows back as `(d2,-d1,d0)` and returns two rows `(a_out,b_out)`.

Keygen builds level-specific relinearization columns satisfying
`B_j-s*A_j=g_j*s^2+t*e_j`, using the same exact CRT gadget as decomposition.
For `d2=sum_j g_j*u_j`, `relinearize` emits

```text
a_out=-d1+sum_j u_j*A_j
b_out= d0+sum_j u_j*B_j
```

The phase becomes `d0+d1*s+d2*s^2+t*sum_j u_j*e_j`. `mul` composes these two
subgraphs without automatically switching modulus. Key targets are raw ring
values, not normal plaintexts scaled by f. Generate only requested level keys
with fresh samples and bounded parallelism; do not project a top-level key and
assume its noise remains suitable at all lower moduli. State the usual
secret-dependent evaluation-key assumption alongside parameter/security claims.

For a modulus step with `p=q_l`, `Q'=Q_l/p`, the mathematical definition is

```text
r = centered((-c*t^(-1)) mod p)
c' = ((c+t*r)/p) mod Q'
f' = f*p^(-1) mod t
```

The quotient above specifies the result; do not evaluate it by recovering c.
For each remaining limb, instead compute

```text
u_p = (-c_p * t^(-1)) mod p
r_i = centered(u_p modulo p) mod q_i
c'_i = (c_i + (t mod q_i)*r_i) * p^(-1) mod q_i
```

This computes the exact divisible quotient modulo q_i with word-size modular
operations. In DSL, concatenate a and b into one (2,1) matrix C and build

```text
U = C.reduce_modulus(p) * R_p.polynomial([-t^(-1) mod p])
R = U.centered_rebase(Q')
out = (C.reduce_modulus(Q') + R * R_Q'.polynomial([t]))
      * R_Q'.polynomial([p^(-1) mod Q'])
```

No coefficient-extraction, Int division, or bit-packing node occurs in this
subgraph. Register the one-limb R_p parameters as well as each destination Q'
in the runtime, preserving the original primes/roots. The numerator is exactly
divisible by p. Precompute only public inverses. The limb calculation matches
[OpenFHE's DCRT ModReduce](https://github.com/openfheorg/openfhe-development/blob/main/src/core/include/lattice/hal/default/dcrtpoly-impl.h);
the inverse-factor update also appears in
[OpenFHE's BGV evaluator](https://raw.githubusercontent.com/openfheorg/openfhe-development/main/src/pke/lib/scheme/bgvrns/bgvrns-leveledshe.cpp).
For the literature basis, use Kim, Polyakov, and Zucca,
[Revisiting Homomorphic Encryption Schemes for Finite Fields, Section 2.3](https://www.iacr.org/archive/asiacrypt2021/130900159/130900159.pdf)
for the machine-word RNS representation. This initial operation drops one prime
at a time and uses an exact single-limb correction; it does not require a general
approximate multi-prime basis extension. The linked conference paper was
consulted; the expanded [ePrint 2021/204](https://eprint.iacr.org/2021/204)
is a further reference, not an unread appendix used as evidence for this formula.
Keep the secret unchanged as an integer polynomial. `mod_switch_to` emits a
fixed chain of typed calls for successive lower levels; equal level is identity,
higher level or deleting the final prime is an error. Because the modulus changes
the wire type, do not implement this chain using a schema-invariant `iterate`.

## SIMD encoding and rotations in the first version

The existing automorphism and integer DSL facilities make a restricted, useful
SIMD implementation reasonable in the initial scope. Require prime t with
`t=1 mod 2N`; validate a primitive 2N-th root zeta. There are N independent
Z_t slots. Choose a fixed two-row layout with N/2 columns and roots

```text
root[row,j] = zeta^((-1)^row * 5^j mod 2N)
slots[row,j] = m(root[row,j]) mod t
```

This is the usual two-row batching geometry; the root condition and row operations
are documented in the [SEAL manual, Section 5.6](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/12/sealmanual.pdf).
No global N-slot cyclic rotation is implied by `rotate_rows`.

The private `encode_slots` helper zero-pads short inputs, permutes slots into
native evaluation order and calls `Ring::from_evaluations(...).coefficients()`.
`decode_slots` calls `Ring::from_coefficients(...).evaluations()` and restores
logical slot order. The generic `PolynomialFromValues` and `PolynomialValues`
DSL nodes import/export runtime integer families. The backend uses
`Poly::from_biguints_eval(...).coeffs_biguints()` or
`Poly::from_biguints(...).evals_biguints()`. There are no FHE-specific NTT
butterflies, twists, or transform stages. CPU and GPU use their existing native
NTT implementations. Shared FHE coefficient packing/extraction also uses these
operations, avoiding per-bit packing and repeated full-polynomial extraction. Only the public root/order permutation is computed while
building the graph.

Register `BgvParams::batching_parameters()` in the runtime backend alongside the
ciphertext CRT chain. The plaintext transform operates modulo t, not Q. GPU
backends register the corresponding `GpuDCRTPolyParams` with the same prime t.
The native evaluation import has matching semantics on CPU/GPU: its input is
already evaluation data, not coefficients to be forward-transformed.

Compare the encoded output to the established primitive evaluation representation
at t, with an explicit root/order adapter, in addition to slotwise runtime tests.
The [SEAL encoder](https://github.com/microsoft/SEAL/blob/main/native/src/seal/batchencoder.cpp)
provides a primary implementation reference for the transform convention.

Accept 1 to N runtime slot values and fill unused positions with DSL zeros.
Arithmetic modulo t acts independently on slots through the ordinary encrypt,
add/mul, and decrypt API. Private coefficient helpers never change the public
plaintext interpretation. Tests cover a single negative value, partial inputs,
full inputs, and rotations into initially unused slots.

```rust
fn rotate_rows(&self, key: Option<&Mat>,
               ct: &BgvCiphertext, steps: i32)
    -> Result<BgvCiphertext, FheError>;
fn swap_rows(&self, key: &Mat, ct: &BgvCiphertext)
    -> Result<BgvCiphertext, FheError>;
```

Normalize steps modulo N/2. Positive k means
`out[row,j]=in[row,(j+k) mod (N/2)]`, using automorphism index `5^k mod 2N`.
Swap rows uses index `-1 mod 2N`. Zero rotation is identity and needs no key.
For nonzero rotation, emit `a'=sigma_k(a)`, `b'=sigma_k(b)` with the existing
`Mat::ring_automorphism`; this encrypts under `s'=sigma_k(s)` and therefore
requires a key switch back to s. Rotation-key columns satisfy
`B_j-s*A_j=g_j*s'+t*e_j`. With `a'=sum_j g_j*d_j`, emit

```text
a_out = -sum_j d_j*A_j
b_out = b' - sum_j d_j*B_j
phi_s(output) = sigma_k(b-s*a) - t*sum_j d_j*e_j
```

The modulus and f do not change. Reuse the key-switch matrix/decomposition helpers
used for relinearization with the correct target and signs. Generate only requested
keys. Negative steps require the key for the inverse index. The caller supplies
that key explicitly; None is accepted only for a zero rotation. This avoids a
new key registry and does not claim automatic target verification from raw Mat.
Rotation indexes are compile-time public
arguments initially, not secret or dynamically evaluated indexes.

## Unit tests: production DSL through runtime

The required correctness tests are ordinary `#[cfg(test)]` library unit tests,
even though they exercise multiple crates. They must execute the production DSL
algorithms through runtime and compare the resulting decrypted values with the
expected values. Direct native FHE calls or host-side decryption do not satisfy
this requirement.

For each test:

1. Construct explicit small ring/chain parameters, registering every Q_l and
   every dropped-prime single-limb ring with
   `cpu_backend`. Use public NTT constants for t for BGV. Read test
   sizes from environment variables and draw fresh runtime randomness.
2. Build production DSL keygen, encrypt (including encoding), evaluate, and decrypt (including decoding)
   calls, then declare the final plaintext/slot result as a private output.
   Runtime message inputs use `DslContext::int_family_input`; do not bake input
   messages into graph constants or host-produced ciphertexts.
3. Call `DslContext::build`, `BuiltGraph::validate(&ParamEnv)`, then
   `mxx_runtime::execute` with `MemoryArtifactStore` and `SamplingMode::Fresh`.
4. Read final `RuntimeValue::IndexedFamily`/`Int` leaves and assert exact equality
   modulo t. Expected slotwise add/multiply and index permutations are elementary
   arithmetic, not a second cryptosystem. Coefficient polynomial expectations
   use existing trusted primitives.
5. Add a staged execution test that exports keygen/encryption outputs, runs an
   evaluator graph without a secret-key input, then imports its ciphertext into
   a separate DSL decryption/decoding graph. This verifies real artifact bindings,
   explicit key/artifact connections, and evaluation without accidental secret access.

Required cases:

- Noisy Ring Regev round trips and addition, exact gadget recomposition, and
  Ring-GSW external products by 0, 1, a small polynomial, and a monomial crossing
  X^N=-1. Noiseless phase fixtures only supplement noisy tests.
- BGV noisy round trip, addition, multiplication before/after relinearization,
  one-step/repeated modswitch, explicit factor alignment, and lower-level
  operations with a nontrivial f. Include p != 1 mod t so factor bugs cannot hide.
- RNS-only modswitch at Q wider than 128 bits. Compare output limbs against the
  trusted native OpenFHE ModReduce operation, including corrections on both
  sides of p/2 and unequal/reordered valid CRT bases. Separately run noisy
  DSL encrypt -> modswitch -> decrypt and compare plaintexts. Verify the isolated
  modswitch graph has no coefficient extraction, Int arithmetic, or packing
  nodes, and audit its native call chain for `coeffs`/CRT interpolation calls;
  graph inspection alone cannot prove that a backend avoids reconstruction.
  Preserve evaluation format and compare the registered output basis exactly.
- SIMD encode/decode, slotwise addition/multiplication, positive/negative/zero
  and wraparound row rotations, and row swap. Distinct values in both rows ensure
  index direction and accidental cross-row rotations are observable.
- A complete SIMD graph: encode -> encrypt -> multiply -> relinearize ->
  modswitch -> rotate -> decrypt -> decode. Assert every slot exactly.
- Invalid SIMD modulus/root, wrong slot count, missing optional rotation key,
  malformed evaluation-key shape, incompatible ring/shape/level/factor,
  incorrect explicit artifact bindings,
  and malformed artifact schemas. Do not assert universal wrong-key detection.

A useful insecure toy fixture is N=8,t=17 with two rows `[0,1,2,3]` and
`[4,5,6,7]`; rotation by +1 must return `[1,2,3,0]` and `[5,6,7,4]`.
Use larger env-selected parameters when needed for noisy evaluation; these are
correctness fixtures, not secure deployment defaults. Each test owns a fresh
in-memory store, so no file artifacts or stale checkpoints are reused.

## Implementation order and validation boundary

1. Freeze DSL schemas, exact CRT gadget layout, phase signs, public factor
   metadata, and SIMD slot/rotation ordering. Consult the
   [BGV paper](https://eprint.iacr.org/2011/277),
   [TFHE paper](https://eprint.iacr.org/2016/870), and primary implementations
   above; record precise algorithm/version/section references before implementing
   bounds. The named papers are not currently present in `references/`; keep
   that directory read-only and resolve any specification conflict before coding.
2. Add crate membership, common graph traits/types, coefficient subgraphs,
   parameter/schema validation, and the minimal runtime unit-test fixture.
   Implement and validate direct CRT projection and single-limb centered rebase
   through primitives, IR, DSL, and runtime before
   building BGV modswitch. No temporary BigInteger modswitch path is permitted.
3. Implement Ring Regev and Ring-GSW entirely as DSL subgraphs with runtime
   decryption assertions.
4. Implement BGV keygen, encryption/decryption, arithmetic, shared key switching,
   relinearization, and modulus-switch subgraphs with runtime unit tests.
5. Implement primitive-backed SIMD encode/decode and automorphism-plus-key-switch rotations;
   add the complete SIMD runtime test and staged evaluator test.
6. Add term-by-term error bounds matching the DSL graphs, including encoding
   bounds, gadget digit errors, factor alignment, modulus correction, and
   rotation key switching. Use exact integer/rational comparisons against the
   full modulus at each level. Backend feasibility and round trips alone do not
   establish a secure parameter choice. Do not add arbitrary CRT-depth margins.

At implementation time use `cargo +nightly fmt --all`, targeted
`cargo test -r -p mxx-fhe --lib` tests, and required warning-free workspace library
compile checks from `BUILDER.md`. Runtime-backed library unit tests are explicitly
requested here; integration-test targets remain unrequested. Follow `GPU.md` and
required outside-sandbox repeated tests for GPU execution. The GPU path uses
stream-ordered INTT, a single batched unsigned centered-lift kernel, and NTT;
ciphertext coefficients remain on device throughout the modulus switch.


## Validation results

- `cargo test -r -p mxx-fhe --lib`: 11 passed. This includes noisy decryption,
  explicit phase bounds, in-memory staged evaluation, SIMD arithmetic/rotations,
  exact CRT gadget recomposition, and the native ModReduce oracle comparison.
- Targeted library tests passed for runtime CRT projection/centered conversion,
  canonical integer artifacts, mixed-width primitive gadget decomposition,
  centered-rebase IR validation, and Lean export of heterogeneous CRT transforms.
- `cargo test -r --workspace --lib --no-run`: passed without warnings.
- `cargo test -r --workspace --lib --features gpu --no-run`: passed without warnings.
- `cargo +nightly fmt --all` and `git diff --check`: passed.

GPU validation on NVIDIA GeForce RTX 4080 SUPER:

- `test_gpu_matrix_centered_rebase_matches_cpu`: five identical binary runs,
  all passed. Includes both coefficient storage widths, centering boundaries,
  exact CPU comparison, evaluation format, and source lifetime checks.
- `test_gpu_fhe`: five identical binary runs, both tests passed every run.
  Ring-GSW round trip/addition/external product and staged BGV SIMD
  multiplication/relinearization/repeated modswitch/rotations/row swap all compare
  decrypted outputs exactly. All artifacts remain in memory.
- The GPU correctness fixtures configure complete output widths through the
  existing manual-width API. They remain concurrent and do not calibrate the
  process-wide CUDA pool while another context is live. Automatic calibration's
  shared-context rejection was observed in the initial run; no production
  calibration or cryptographic behavior was changed to bypass it.

GPU support is selected with the `gpu` feature and the existing runtime GPU
backend. Validation used one detected GPU; multi-device execution was not tested.
No integration targets were run. This does not claim a production security
parameter set.

### Per-ciphertext noise tracking

`BgvCiphertext::noise_bound` bounds integer noise E in
v = centered(v mod t) + t*e. With h = floor(t/2), the phase envelope is
V = h + tE. Addition, multiplication and correction-factor matching propagate
phase envelopes and convert back using floor((V+h)/t), including plaintext
reduction carries. Multiplication uses the negacyclic convolution bound N*V1*V2.
Relinearization and rotation add N*L*(B/2)*error_cutoff to E.
Dropping p uses V' = ceil((V+t*floor(p/2)*(1+N))/p), matching the
DSL's centered residue correction. `can_decrypt` checks 2*(h+tE) < Q_level.

Ring ciphertexts carry both E and a centered plaintext coefficient bound M.
External products propagate E' = N*M_gsw*E_regev + N*(2L)*(B/2)*E_gsw
and M' = N*M_regev*M_gsw; addition sums both bounds.
`can_decrypt` checks 2E < scale and 2*(scale*M+E) < q.
These are conservative sufficient conditions and depend on declared input bounds.
Schemas preserve public bounds through DSL reconstruction; separate protocol
stages must pass them with ciphertext components and correction factors. All
artifacts in runtime tests remain in memory. Tests compare decoded plaintexts
and actual coefficient noise with propagated bounds.

Validation after ciphertext-bound tracking and module consolidation:

- CPU FHE unit tests: 12 passed, including measured coefficient noise bounds.
- CPU and GPU workspace library test builds: warning-free.
- GPU FHE unit tests: both tests passed in each of five identical-binary runs
  outside the sandbox (zero failures).

Validation after making SIMD slots the default BGV plaintext interface:

- CPU FHE unit tests: 13 passed, including single/partial slot inputs and
  unchanged coefficientwise noise and native CRT modulus-switch checks.
- CPU and GPU workspace library test builds: warning-free.
- All three GPU FHE tests passed in each of five identical-binary runs outside
  the sandbox, including zero padding and rotation into an initially unused slot.
