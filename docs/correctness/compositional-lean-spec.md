# Compositional Lean correctness over the frozen IR

## 1. Decision and status

This is a replacement design proposal, not an implemented verification system. It is based on
`codex/new-IR` at `d84669f79ad2cd4c72c88970da92043187043244`, pulled on 2026-09-05.
The design branch is `codex/compositional-lean-design`. The companion implementation plan is
[compositional-lean-plan.md](../plans/compositional-lean-plan.md).

The retained requirement is:

> Generate the Lean statement and its protocol definition from the executable frozen IR. Write
> the noise calculation, intermediate mathematical invariants, and their proofs for each
> application. Reuse ordinary primitive and cryptographic lemmas at their owning crate layers.

The proposed simplification is **direct relational extraction**: translate each existing scoped SSA
body into an ordinary Lean relation definition. A scope relates its inputs to its outputs. Pure
operations become `let` bindings, sampled results become locally bound variables, a parallel loop
relates function-valued families pointwise, and a sequential loop uses one common iteration relation.
Application proofs work with those generated definitions directly.

There is no global execution trace in the public theorem, no carrier analysis, no relation search,
and no application-specific duplicate of the graph's topology. There is also no second Lean graph
AST, typed-IR elaborator, or graph interpreter. The generated relation definitions are the sole Lean
representation of the executable protocol. The application proves equations about their bound values.

The first deliverable is the Diamond WE correctness theorem, for both Boolean messages and all
valid runtime circuits and accepting witnesses of a supported shape. Future private-private BGG+
and GSW-assisted Fuse operations use additional application lemmas, without changing this semantic
interface.

This choice is justified by the code and failure modes below. It is not a claim that a globally
optimal design has been proved. The implementation plan deliberately tests its highest-risk
assumptions on small examples before rebuilding Diamond.

## 2. Baselines must not be confused

There are two distinct implementations to compare.

| Baseline | Relevant state |
| --- | --- |
| `codex/new-IR`, `d84669f79` | Executable `ParallelLoop` and `SequentialLoop`; active Rust operational checker in `mxx-correctness`; no tracked crate-local Lean implementation |
| `codex/app-specific-lean`, `7cb9d089d` | Prior application-specific Lean attempt and handoff; trace-based proof machinery and a large Diamond-specific emitter |

The old report is available from Git as
`7cb9d089d:docs/app-specific-lean-correctness-handoff.md`. Its dirty-worktree and build-failure
statements describe its earlier snapshot. They are not current validation results for `new-IR`.
The prior branch is preserved, and no part of it is merged by this proposal.

Switching branches leaves old ignored build artifacts under several `crates/*/lean/.lake`
directories. They are not tracked implementation on `new-IR` and do not establish a successful
build. They have not been deleted or reused in this task.

Concrete evidence for the redesign:

- `crates/ir-core/src/graph.rs`, `GraphScope`, `Graph::freeze`, and `child_scope_id`: structural
  child bodies already exist as shared scope definitions.
- `crates/ir-core/src/node.rs`: current control operations are `ParallelLoop`, `SequentialLoop`,
  and `SubgraphCall`, not the old `ParallelGrid` representation.
- `crates/we/src/diamond/parameter_search.rs`, `evaluate_candidate`: acceptance currently calls
  `check_operational_noise_candidate`; it is not a Lean verification path.
- `crates/correctness/src/operational_noise`: approximately 46,800 Rust lines including tests at
  this snapshot; the largest components are lowering and normalization. This is a maintenance
  measurement, not a runtime benchmark or proof that every existing path is inefficient.
- The prior branch's `ScopeInvariant.lean` has 3,991 lines and its Diamond `emit.rs` has 6,095
  lines. Much of that machinery identifies occurrences and transports values out of traces.
- `crates/bgg/src/encoding.rs`, `BggEncodingCompiler::mul`, and
  `crates/bgg/src/boolean.rs`, `encoding_multiply`: the useful local algebra is much smaller than
  the trace reconstruction required to reach it in the old proof.

## 3. What to retain and what to replace

Retain the frozen graph as the authoritative protocol, exact artifact linking, crate ownership,
ordinary Lean algebra, bounded-sampler support assumptions, and the application's choice of
invariants. Retain matrix/gadget/preimage runtime semantics and GPU parallelism.

Replace the following correctness mechanisms:

| Mechanism | Replacement |
| --- | --- |
| Global Large-term normal form and cancellation search | Local exact equations in application proofs |
| Carrier/source provenance interpreter | Ordinary values and source/preimage equations |
| Full trace, fuel, reached-node occurrence bundles | Scope-local input/output derivations |
| Diamond role-map validators that restate internal topology | Proofs directly about generated scope definitions |
| A second Lean graph AST and dependent environment interpreter | Direct Lean `let`, existential, function, and relation definitions |
| Caller-supplied execution records and output equalities | Values introduced by relational semantics |
| Materialized family certificates or index-use tables | A theorem for an arbitrary index |
| Step-by-step numeric simulation of uniform long loops | Proved fixed-dimensional recurrence summaries |
| Generic bound-expression language shared with the symbolic engine | Diamond-owned arithmetic functions and their Lean proofs |

Correctness metadata such as source restrictions, explicit carrier restoration, or bans on
concatenating source-bearing values was introduced to compensate for information loss in an
abstract interpreter. It is not required by the new semantics. Slice and concat retain their usual
matrix meaning. Whether a useful theorem holds after them is an ordinary mathematical question.

The same applies to index-independent sources: it is a useful hypothesis for a particular
preimage-family lemma, not a restriction on all IR values. A source may depend on outer loop
parameters while remaining independent of an inner branch selector.

## 4. Minimal semantic vocabulary

The core needs the following concepts. The Rust graph remains unchanged; its nodes are exported as
Lean expressions, not as another interpreted data structure.

| Concept | Purpose |
| --- | --- |
| Mathematical wire types | Matrices, scalars, tuples, trapdoors, and function-valued families |
| `PrimitiveRuns` | Fixed primitive relations for successful sampling and partial operations |
| `Rel A B := A -> B -> Prop` | The type of a generated scope relation |
| `IterRuns` | One generic relation for a counted carried-state loop |
| `ScopeSpec` | An ordinary predicate over a generated relation's inputs and outputs |
| `CoeffBound`, `Approx` | Integer coefficient bound and modular approximation |

Do not add `SemanticValueId`, `Carrier`, `NoiseState`, `Reached*`, `TraceView`, a symbolic matrix
expression arena, or a generic collection of application contract records. Existing source IDs in
the frozen graph still identify wires; they are not new semantic identities.

The declarations below specify mathematical interfaces. They are not claimed to be compiler-tested
Lean source. Concrete syntax, universe parameters, and routine dependent type transports remain
implementation work. The interfaces and quantifiers are normative.

### 4.1 Rings, matrices, and bounds

Reuse mathlib's quotient-ring construction rather than reprove ring laws for a custom convolution
datatype. The prior primitive package already used `AdjoinRoot`; selectively port its useful
algebra and the tight convolution bound after review.

```lean
abbrev RingPoly (n : Nat) (R : Type) [CommRing R] :=
  AdjoinRoot (Polynomial.X ^ n + Polynomial.C (1 : R))

abbrev ModMatrix (q n rows cols : Nat) :=
  Matrix (Fin rows) (Fin cols) (RingPoly n (ZMod q))

abbrev IntMatrix (n rows cols : Nat) :=
  Matrix (Fin rows) (Fin cols) (RingPoly n Int)

def CoeffBound (e : IntMatrix n rows cols) (B : Nat) : Prop :=
  ∀ (r : Fin rows) (c : Fin cols) (k : Fin n),
    Int.natAbs (coeff (e r c) k) ≤ B

def Approx (actual ideal : ModMatrix q n rows cols) (B : Nat) : Prop :=
  ∃ e : IntMatrix n rows cols,
    actual = ideal + reduce q e ∧ CoeffBound e B
```

For `n > 0`, let `f = X^n + C 1` and prove `f.Monic`. The representative of `x : RingPoly n R`
is `AdjoinRoot.modByMonicHom` applied to that monicity proof and `x`; `coeff x k` is coefficient
`k.val` of this remainder. Prove its degree is less than `n` and that its image in the quotient is
`x`. These fix the coefficient convention used by every bound lemma; no arbitrary lift is chosen.

`reducePoly : RingPoly n Int →+* RingPoly n (ZMod q)` is induced by coefficient reduction
`Int →+* ZMod q`: map a polynomial's coefficients and then take its quotient class. Prove this
is well-defined by mapping `X^n + 1` to zero in the target quotient. Define matrix reduction by
`reduce q e r c = reducePoly (e r c)`, and expose lemmas for preservation of zero, addition,
negation, and matrix multiplication. This reuses the quotient universal property, not a second
polynomial representation. Require `n > 0` and `q > 1` at the relevant parameter boundary.
Do not carry exceptional zero-dimensional ring branches through application proofs.
Matrix row/column dimensions may be zero only where the actual IR permits them.

A magnitude bound on a modular value is simply `Approx value 0 B`. There is no second magnitude
record. A preimage relation is the equation `B * K = T`, optionally a notation, not a certificate
containing independently chosen matrices.

Bounds are on integer error witnesses before modular reduction. They are not absolute norms of
arbitrary canonical residues. The final centered modular distance follows from the existence of
this witness; the witness need not equal the centered lift coefficient by coefficient.

### 4.2 Values and families

Lean mirrors the current resolved wire-type constructors, including matrices, small matrices,
preimages, trapdoors, integers, Booleans, reals, bytes, blobs, and nested families. Compile-time
constants may share the same mathematical value domain as their runtime counterparts while their
wire-type tags remain distinct in the graph.

The critical type translation is:

```lean
IndexedFamily element N  -->  Fin N → LeanType(element)
input/output port list  -->  an ordinary typed tuple in port order
```

Matrix, small-matrix, and preimage wires have the same underlying
modular matrix representation. Their different primitive construction rules establish the required
facts. No conversion can delete an equation already in the Lean context. In particular, the theorem
does not rely on a `Preimage` wrapper storing a hidden carrier relation.

A trapdoor value contains its public matrix and a private abstract token. The token's contents are
not used by noise algebra. Sampling relates the result to that very public matrix, rather than to a
same-shaped matrix found elsewhere.

Retain the trapdoor's layout and whether it is a sampled secret trapdoor or the public gadget
trapdoor. The latter invokes deterministic decomposition in the current runtime; it is not a new
random choice. A declared output cutoff is not itself a proof of a digit bound: derive the bound
from the actual layout and check that it fits the declaration.

Nested families are nested functions; reindexing is function composition. A uniform family bound is
`∀ i, Approx (actual i) (ideal i) B`. It is one proposition and one proof abstraction, not a list of
`N` certificates.

### 4.3 Parameter and type resolution

Translate `IntExpr`, `RealExpr`, and lexical loop bindings mechanically to Lean expressions. Do not
substitute every possible loop index. The initial exporter may resolve candidate parameters and
uniform matrix geometry once, giving ordinary concrete matrix types. Loop indices remain symbolic
binders in operation arguments, selectors, and payload expressions. Parameterizing generated
definitions over a record of dimensions is also allowed where it reduces repeated proof work.

The initial acceptance artifact may be parameter-specific. Reusable algebra and map/iteration lemmas
are parametric. An application-owned proof script elaborates the same generated scope proof template
for each candidate; it does not prove each concrete runtime lane or run. No universal theorem about
all numeric candidates is required before obtaining the first real verified candidate.

A named subgraph used with different compile bindings remains one definition with explicit formal
parameters. Each call supplies its own actual bindings. Candidate-specific root specialization must
not overwrite one shared scope's parameter environment or clone it for every call/lane. In
particular, the existing validator's representative scope bindings are not authority for all
invocations; export follows the original scope expressions and actual call bindings.

Types are emitted from validated wire descriptors. The Lean type checker checks actual argument and
result types directly. If an expression equality requires transport, emit its local kernel-checked
arithmetic proof; do not ask the application to identify a matrix through `HEq`. Uniform output
geometry is required for the initial family export. A geometry condition that depends on the index
must be proved uniformly, or that case is explicitly unsupported. It must never trigger lane expansion.

Preserve the exact existing expression semantics:

- `IntExpr::Div` requires divisibility and rejects zero denominators;
- `RoundDiv(a,b)` for `b > 0` is `floor((2*a+b)/(2*b))`, including negative inputs;
- `Log2Ceil` requires a positive argument;
- a child environment first binds the loop index, then evaluates every named binding against the
  same snapshot, as in `Runtime::child_env`; bindings are not sequential assignments;
- signed runtime indices must be proved nonnegative and in range before conversion to `Fin`;
- compile-time exact division and runtime integer division are separate operations and must follow
  their respective existing definitions; and
- index-dependent payloads are allowed even when the output family element type is uniform.

For the initial checker, variable geometry must be justified symbolically. Unsupported symbolic
type constraints produce an explicit error, never an enumeration of every lane as a fallback.

## 5. Direct semantics of the existing graph

### 5.1 Lexical variables implement SSA sharing

The type of a generated scope is `Rel Inputs Outputs`, where `Rel A B := A -> B -> Prop`.
Parameters and the fixed hash/token interpretation can be additional ordinary arguments. The exporter
visits stored SSA nodes in dependency order and maps each result wire to one Lean variable. This
wire-to-variable map exists only during export and has size proportional to stored syntax.

For example, a scope sampling `K` and computing `(s*B+e)*K` is emitted in the following shape:

```lean
def generatedScope (p : Params) (inputs : Inputs p) (out : Output p) : Prop :=
  ∃ K,
    preimageRuns inputs.public inputs.trapdoor inputs.target K ∧
    let c := inputs.s * inputs.public + inputs.e
    out = c * K
```

The source sampler payload, including its cutoff, is an argument of `preimageRuns`; omitted fields
in this illustration are not optional in generated code. Pure operations use `let`. Partial pure
operations include their domain guards. A child call introduces one output tuple and its generated
child relation before the continuation. The final tuple consists of the actual stored output wires.

Every reachable sampler and every retained/effect root is represented, even if its value is unused
in the final tuple. In particular, an unused partial pure operation still contributes its domain
guard. `Select` does not delete eager dependencies.

Eliminating this proposition gives exactly the sampled `K`, its relation, and the actual product
equation. Lean variable binding supplies sharing without a semantic environment lookup. Two uses
of a wire reuse one variable. Multiple outputs of a primitive or child call are bound together.
Different invocations introduce fresh logical variables even when they call the same scope definition.

Only the counted iteration relation needs an inductive execution derivation. Parallel output relations
use universal quantification. There is no fuel, trace, or interpreted node-sequence datatype.

### 5.2 Primitive rules and sampling

Deterministic primitives are equations with exact operands, type/layout parameters, and results.
All fusions retain their algebraic meaning. For example,
`MatrixMulAccumulate` means `bias + sum(coeff[t] * left[t] * right[t])`, and
`MatrixMulSmallRhs` means the same ordered product as matrix multiplication.

For bounded probabilistic sampling, `PrimitiveRuns` describes possible successful returned values:

| Primitive | Required relation |
| --- | --- |
| Gaussian with cutoff `B` | There exists an integer lift with coefficient bound `B` |
| Uniform interval | An integer lift lies in the exact interval defined by the primitive |
| Uniform residue | Any value in the actual residue ring |
| Preimage from trapdoor with public matrix `S`, target `T`, cutoff `B` | `S * K = T` and `Approx K 0 B` |
| Gadget decomposition | The actual deterministic digit function of the actual target; reconstruction and digit-bound lemmas are proved about it |

The preimage relation and error bound share the same `K`, `S`, and `T` introduced by that operation.
The target is not assumed to have zero error. A later application lemma may prove
`Approx T P E` from the operations that constructed it.

In the current IR, `PreimageSample` has three ordered arguments: an explicit public matrix, a
trapdoor, and a target. The rule requires equality between the explicit public matrix and the
trapdoor's public matrix, matching `PreimagePublicMismatch` in the runtime. Do not drop the explicit
public operand merely because the trapdoor also contains it. `TrapdoorSample` introduces its public
matrix and trapdoor token in one result tuple, with the same public matrix in both ports.

Gadget decomposition and preimage sampling reuse the equation and multiplication lemma. Gadget
decomposition must additionally remain deterministic: replacing it by an arbitrary small preimage
would lose equality between repeated decompositions of the same input.

Hash sampling uses one globally fixed mathematical hash interpretation of the actual tags, types,
and inputs. Equal requests give equal values; distinct requests need not differ. It is not an
independent arbitrary sample on every call. Plain, decomposed, and small-decomposed variants must
preserve the actual hash/decomposition relationship. Do not model finite-family hashing by assuming
that inspecting one element determines the whole family's hash.

Fresh sampling does not require a global occurrence-ID registry for deterministic bounds. Distinct
node/call binders may choose different allowed values, while uses of a bound result share it. If
replay mode adds equalities, the bounded relation can overapproximate it; universal correctness over
that larger relation remains sound. No distributional independence is assumed.

Preserve quantifier nesting exactly. Sampling one shared secret before a loop yields
`exists s, sample(s) and forall i, body(s,i,...)`. Sampling inside its body yields
`forall i, exists s_i, sample(s_i) and body(s_i,i,...)`. These are different programs. Likewise,
bind an intermediate output family once before multiple consumers use it; never reintroduce its
sampler existential independently at each consumer.

### 5.3 Parallel loops

For a loop of length `N`, output families `ys`, and stored child scope `body`, the defining rule is:

```text
forall i : Fin N,
  generatedBody model (childEnv params i) (laneInputs i) (laneOutputs ys i).
```

For each stored input mode, `laneInputs` is exactly:

- `Broadcast x`: `x`;
- `Zip xs`: `xs i`;
- `ZipOffset offset xs`: `xs (i + offset)`, with the actual bound proof.

The index expression above means natural-number addition, not addition in `Fin`. More precisely,
for `xs : Fin M → A` and a validated bound `h : ∀ j : Fin N, j.val + offset < M`, use
`xs ⟨i.val + offset, h i⟩`. A static inequality such as `N + offset ≤ M` supplies this proof
without enumerating indices. Preserve the existing IR validation restrictions even for empty
families; this semantic rule does not relax the runtime's checked-index boundary.

The runtime's integer loop binding is `Int.ofNat i.val`; a sequential step binds `Int.ofNat i`
for its natural index `i`. Resolve counts and offsets to nonnegative integers fitting the runtime
index representation, and preserve its checked-addition requirement. Negative or overflowing
indices are rejected, not coerced by truncation or modular arithmetic.

All output families are projected from that one child output tuple. Never choose independently
sampled vectors and public keys that merely have the same shape.

The generic proof rule requires one body specification for arbitrary `i` and gives the pointwise
family postcondition. It does not require or construct an array of body proofs. A zero-length
parallel loop has no lanes and executes no child sampler. It returns empty function-valued families.

For example, the semantic shape of a generated preimage family followed by multiplication is:

```text
exists K : Fin N -> PreimageMatrix,
  (forall i, preimageRuns actualPublic actualTrapdoor (targets i) (K i))
  and out = (fun i => inputs i * K i).
```

The same family `K` appears in the sampling facts and multiplication. A proof introduces one
arbitrary `i`, applies the local preimage lemma, and generalizes over `i`. This example does not
require a relation registry, a per-lane record, or a global valuation. The exporter preserves the
actual intermediate scopes rather than performing this algebraic fusion automatically.

`minimum_count`, parent bindings, and any count-expression restrictions retain their existing IR
validation meaning; they are not permission to run phantom lanes.

### 5.4 Selection, packing, and reindexing

`FamilyGetStatic` and `FamilyGetDynamic` read the actual family and actual index. A bound
`∀ i, Approx (xs i) (ideal i) B` immediately specializes to that index; there is no selector state.

For related families, preserve the same selector:

```text
forall i, B * K i = T i
---------------------------------
B * K (selector j) = T (selector j).
```

An arbitrary sequence of gathers is ordinary function composition. Repeated, noninjective, and
nested selectors require no provenance extension. Out-of-range selectors are rejected, never
clamped or assigned zero.

`Select` chooses one of its actual already-computed inputs. The rule must not erase evaluation of
other reachable inputs or their sampler obligations by treating an eager graph as a lazy branch.

`FamilyPack` with `N` explicit argument wires has an intrinsically `O(N)` input description. This is
different from a structural parallel loop with `N` logical lanes and one stored body. The checker
must read all explicit wires, but must not expand a structural family into such a pack.

### 5.5 Sequential loops

Preserve the actual initial carried tuple, loop-invariant arguments, count, and index binding.
The body returns precisely `carried_count` outputs, which become the next carried tuple. At zero
iterations the result is the initial state.

Use a single invariant `I i state` and the ordinary induction rule:

```text
I 0 initial
forall i < L, forall state next,
  I i state -> generatedBody (paramsAt i) (state, invariants) next -> I (i+1) next
--------------------------------------------------------------------------------
IterRuns body L initial final -> I L final.
```

The common relation has the following shape:

```lean
inductive IterRuns (body : Nat → State → State → Prop) : Nat → State → State → Prop
  | zero (initial) : IterRuns body 0 initial initial
  | step : IterRuns body i initial current → body i current next →
      IterRuns body (i + 1) initial next
```

`body` is the actual generated child relation with invariant inputs captured as ordinary function
arguments. Its index is a natural number; a use requiring `Fin L` derives the bound from the loop
step hypothesis, never from modular index arithmetic.

The proof size is independent of `L`. Proving the step must preserve all carried fields together.
There is no list of iteration certificates and no expansion of the graph for every iteration.

## 6. How proofs attach to the generated program

Rust exports every unique frozen scope once as a relation definition. Node payloads become arguments
to fixed semantic functions/relations, wires become variable uses, and child references become calls
to generated definitions. Bindings, effect roots, and the linked workflow mapping follow the same
mechanical translation. The exporter contains no Diamond inference.

The generated Lean module contains these definitions and the linked program relation, for example:

```lean
def encryption (...) (inputs : EncInputs) (out : EncOutputs) : Prop := ...
def decryption (...) (inputs : DecInputs) (out : DecOutputs) : Prop := ...
def program (...) (external : ExternalInputs) (out : Outputs) : Prop :=
  ∃ encrypted, encryption ... encrypted ∧
    decryption (linkedInputs external encrypted) out
def claim (...) := DiamondCorrect program ...
```

The export sidecar maps scope IDs, nodes, ports, and named roots to generated definitions/binders or
output tuple projections. It is useful for regeneration and debugging; it is not a second Lean
semantic representation. Each operation must have one documented translation rule and source-map
coverage. Repeated scope definitions are shared. Private artifact values are ordinary linked values,
not separately sampled external assumptions.

An application theorem is about this `program` or its generated child relations, not a separately
generated expected Diamond shape. Any generated local equation lemma follows from those definitions.
There is at most one such lemma per static node, never one per dynamic occurrence.

This architecture explicitly trusts the mechanical Rust exporter and its mapping to the primitive
library, just as the previous proposal trusted the Rust-to-Lean data emitter. Hashes detect staleness;
they do not prove translation correctness. Adding a second Lean graph interpreter would only improve
this boundary if accompanied by a checked translation-equivalence proof. That additional scope is
unnecessary for the initial user-requested theorem and is deliberately omitted.

Use generic scope decomposition rules, ordinary `simp` with a restricted lemma set, and application
module lemmas. If repetition warrants a tactic, it may choose existing kernel-checked rules but
cannot introduce axioms, synthesize an ideal protocol, or normalize arbitrary large expressions.
Start with handwritten tactic sequences; build no tactic framework before two actual uses justify it.

The scope specification is only an abbreviation for a proposition:

```lean
def ScopeSpec {Config : Type} {Inputs Outputs : Config → Type}
    (run : (cfg : Config) → Rel (Inputs cfg) (Outputs cfg))
    (P : (cfg : Config) → Inputs cfg → Prop)
    (Q : (cfg : Config) → Inputs cfg → Outputs cfg → Prop) : Prop :=
  ∀ cfg inputs outputs, P cfg inputs → run cfg inputs outputs → Q cfg inputs outputs
```

`Config` denotes the ordinary parameters already needed by the generated scope: candidate geometry,
compile bindings, and any explicitly abstract hash/token model. It need not be a new stored record.
The input and output types may depend on these parameters. Specializing a candidate fixes geometry
without accidentally fixing runtime inputs or dropping the model's quantifier.

Keeping `inputs` in `Q` permits exact input/output identities without introducing a ghost execution
record. The fixed runtime primitive interpretation supplies semantics; any quantified hash/token model
only fills the explicitly abstract primitive domains, not arbitrary replacements for multiplication,
preimage sampling, or the decoder.

Existing DSL subgraphs and structural loop bodies provide theorem boundaries. For inlined matrix
operations, obtain their equations within the containing scope and use an ordinary algebra lemma.
No new region extraction certificate or user-supplied contract annotation is needed initially.

Proof authors may name local variables, define ghost invariants, or use mathematical reference
functions such as Boolean circuit evaluation. They may not replace the generated executable program
with a handwritten one or assume an output equation they are supposed to prove.

## 7. Diamond and BGG+ local mathematics

### 7.1 Preimage consumption

Let `B : d x m`, `K : m x p`, `L : r x d`, and let the stored primitive establish `B*K=T`.
Suppose local proofs give:

```text
c = L*B + reduce(e),       CoeffBound e E_c
T = P + reduce(E),         CoeffBound E E_T
L = reduce(l),             CoeffBound l L_B
K = reduce(k),             CoeffBound k K_B.
```

Then the actual product has the integer error witness `l*E + e*k`:

```text
c*K = L*P + reduce(l*E + e*k)
errorBound <= d*n*L_B*E_T + m*n*E_c*K_B.
```

This theorem is a direct distributivity/associativity and homomorphism proof. It neither discovers
`L` nor classifies `B` as Large. The input-injector invariant supplies `L`, and the target producer's
local proof supplies `P` and `E`.

If `E_T=0`, the first noise contribution vanishes. If `P=U*B_next`, the output ideal is
`(L*U)*B_next`; the exact target source is preserved through ordinary algebra. The special case
`L=1` follows from the identity matrix theorem; do not pay an unnecessary `d*n` factor for multiplying
by an identity. Constant-polynomial factors likewise use the tighter constant-side lemma.

### 7.2 The full BGG+ invariant

The handoff's `something*G + error` describes the carrier-bearing payload part, not the entire
BGG+ ciphertext. The full local invariant, consistent with the prior Lean encoding and current
sampling/multiplication code, is:

```text
c = s*A - x*(t*G) + reduce(e).
```

Keep `s`, `A`, `x`, `t`, and `G` as ordinary values in the BGG module theorem. A zero message is
simply `x=0`; do not append an artificial zero gadget or maintain a carrier tag.

For multiplication, the compatible operand invariants are:

```text
c_L = s*A_L - x_L*(t*G) + e_L
c_R = t*A_R - x_R*(u*G) + e_R
G*D = A_R
A_out = A_L*D
c_out = c_L*D + x_L*c_R.
```

All equations are in the modular matrix ring; displayed errors abbreviate their reductions.
The two `x_L*t*A_R` terms cancel locally, giving:

```text
c_out = s*A_out - (x_L*x_R)*(u*G) + e_L*D + x_L*e_R.
```

Thus the decomposition's zero target error removes its target-error contribution, but a complete
BGG multiplication can still contain the additional `x_L*e_R` term. Omitting this second summand
would undercount noise. The Boolean layer specializes compatible secrets and messages appropriately.

The exact `D` in the vector and key products is shared by the actual family graph, or equality of
separately recomputed deterministic decompositions must be proved. Same shape and bound are not
enough. Select the vector, public key, and message with the same actual selector.

### 7.3 Bit reconstruction and future private-private multiplication

Digit decomposition has its actual semantics, including base, digit count, signed/canonical
convention, and layout. Prove its reconstruction theorem at the gadget layer. This includes inputs
equal to a gadget matrix; no special rule about tracing `G` is needed.

For a future GSW-assisted multiplication, prove the application-level routed-sum identity that
the encoded digit computation reconstructs the required multiple of `C`. Combine it with
`t*C = y*(u*G) + e_C`. If the reconstructed multiple is `x*C`, its decryption contributes `x*e_C`,
in addition to errors from evaluating and combining the encodings themselves. Those errors are
derived from the actual operations, not erased by reconstruction.

This proof may use finite sums and induction without enumerating the symbolic family at generation
time. The generic semantic core sees only matrix operations, family indexing, and scope composition.

### 7.4 Diamond proof decomposition

The application proof has five boundaries:

1. Input preprocessing and injection: one initial encoding and one transition theorem, lifted
   pointwise and then through the input loop.
2. Initial BGG encodings: instance/witness selection, actual public keys, and mask/payload identities.
3. One Boolean gate and the layer loop: six gate cases, coupled tuple selection, active padding,
   and the circuit-evaluation invariant.
4. Projection/final residual: connect the accepting output to the actual projection and subtraction
   in the decryption graph.
5. Decoder: prove the whole-polynomial approximation and the exact Boolean interval result.

Upper-layer theorems consume lower-layer scope specifications. No lower layer receives a proof of
the final Diamond output or a generic "this value has the desired encoding" callback.

## 8. Final statement and exact decoder

The generated statement uses the linked encryption and decryption scopes. Artifact inputs are the
actual exported values of the producer run. Circuit and instance inputs are shared through the
existing protocol input mapping; witness inputs remain decryption-only.

Extract the existing declaration's pure input-validity and satisfaction graphs by the same rules.
`ValidExternalInputs` and `CircuitAccepts` use those generated predicates and their actual input
mappings. The application may prove equivalence to its convenient mathematical circuit model, but
must not substitute a separate unconnected acceptance predicate into the final statement.

Conceptually, the statement is:

```text
forall model params message externalInputs outputs,
  ValidParameters params ->
  ValidExternalInputs generatedRefs params message externalInputs ->
  CircuitAccepts generatedRefs externalInputs ->
  generatedProgram model params externalInputs outputs ->
  Approx (noisyOutput generatedRefs outputs)
         (constant (floor(q/2) * boolToNat message)) (diamondBound params)
  and diamondBound params < decoderRadius q
  and decodedOutput generatedRefs outputs = message.
```

The initial theorem can fix the exported candidate parameters and quantify over the remaining
runtime inputs. A generic parameter theorem is optional when the same extraction template admits
one easily. The theorem takes a proved arithmetic acceptance predicate as part of `ValidParameters`; it does
not prove that arbitrary parameter choices are correct. No final-noise hypothesis is accepted.
The output references are exported from the graph's actual named roots. The decoder operand must
be checked against the actual extraction/comparison edges, not an independently supplied matrix.

This remains a theorem about successful bounded-support executions. It does not prove sampler
termination, probability of a good event, cryptographic security, or Rust/CUDA refinement. The
relation does not silently assume totality. Inconsistent sampler cutoffs may admit no run; detecting
all such cases is not accomplished by a partial-correctness theorem. Structural/type validity and
non-vacuous simple execution fixtures are separate required checks.

The concrete decoder in `crates/we/src/diamond/graph.rs::decode_boolean_interval` uses
`RoundDiv(q-2,4)` and inclusive endpoints. With the exact `RoundDiv` semantics this is:

```text
a = floor(q/4), h = floor(q/2)
decode(z) = (a <= canonicalCoeff0(z) and canonicalCoeff0(z) <= 3*a).
```

For `q >= 4`, define

```text
decoderRadius(q) = min(a, q-3*a, h-a+1, 3*a-h+1).
```

For an integer error bound `B`, `B < decoderRadius(q)` expresses all four interval conditions:

```text
B < a,  B < q-3*a,  B <= h-a,  B <= 3*a-h.
```

Prove the equality to the generated comparisons once for arbitrary admissible `q`, including odd
moduli and inclusive endpoints. Whole-polynomial `Approx` implies the needed coefficient-zero
distance. The encryption-side negative `ceil(q/2)` term yields `floor(q/2)` modulo `q`; use that
identity rather than changing the sign or rounding convention.

The current `new-IR` GPU integration test asserts message equality for `true`. It does not currently
assert observed noise below a Lean-proved bound. The new target explicitly includes both messages and
the noise proposition; any future runtime noise measurement must be added and reported separately.

## 9. Efficiency contract

Use different symbols for different sources of work:

- `S`: bytes/nodes/edges in the finite exported syntax, including explicitly stored arrays;
- `N`: logical family cardinality represented by a structural body;
- `L`: number of sequential iterations;
- `b`: bit length of numeric parameters and bounded arithmetic intermediates;
- `k`: fixed dimension of an application recurrence state, independent of `N` and `L`.

For a fixed body and proof template:

| Component | Required dependence |
| --- | --- |
| IR export and static correspondence | `O(S)` or `O(S log S)`, not `O(N*S)` |
| Generated Lean syntax | `O(S)` plus binary numeral sizes |
| Uniform family proof | One quantified body proof, not `N` instances |
| Dynamic gather/select proof | Function application and the actual index bound |
| Sequential semantic proof | One step proof plus induction, not `L` copies |
| Numeric uniform affine recurrence | `O(k^3 log L)` integer operations via repeated squaring |
| Kernel checking | No reduction of runtime families, circuits, or loops into concrete elements |

These are engineering acceptance conditions for supported proof templates. They are not a general
upper bound on arbitrary theorem proving. Explicit input data of size `N` must be read if it is part
of the theorem artifact; the uniform candidate theorem instead quantifies over runtime circuit and
family values. Runtime validation/execution may of course process `N` actual values.

### 9.1 Numeric summaries, not per-lane noise states

Use a small application-owned bound vector. For an injector transition with uniform selector bound
`U_B`, target error `E_T`, and preimage bound `K_B`, the general bound above gives:

```text
[prefixBound']   [d*n*U_B       0    ] [prefixBound]
[errorBound' ] <=[d*n*E_T   m*n*K_B ] [errorBound ].
```

Use the appropriate constant-side improvement where proved. This illustrates the two needed scalar
quantities; the final recurrence must be derived from the actual Diamond target and state dimensions.
Extra additive terms use one constant coordinate, giving a fixed affine matrix. Never add a coordinate
for every public source, wire, lane, or loop iteration.

For Boolean BGG multiplication with small Boolean `x_L`,
`B_product <= (m*n*D_B + 1)*B` if both input errors are bounded by `B`. XOR then has bound
`2*B + 2*B_product`. A uniform layer majorant may cover all six gate cases, including the constant-one
encoding, using a fixed-dimensional affine recurrence. Prove every row of that majorant and quantify
any conservatism. Do not substitute an arbitrary loose majorant merely to make the proof easy.

Uniform recurrences are composed by matrix powers. Nonuniform depth-specific data supplied explicitly
can require linear work in its input size. The initial candidate checker uses a proved uniform majorant;
it does not promise sublinear evaluation of an arbitrary list of unrelated transitions.

### 9.2 Bound bit growth matters

`O(log L)` multiplications alone do not give sublinear bit complexity: an exact value such as `a^L`
has `Theta(L log a)` bits. The checker only needs to decide whether the final bound is below the
decoder radius `C`. Evaluate the nonnegative bound arithmetic in capped naturals:

```text
cap_C(x) = min(x,C)
x +_C y = min(x+y,C)
x *_C y = min(x*y,C).
```

Prove that capping commutes with nonnegative addition and multiplication, and therefore with the
fixed-dimensional matrix-power bound evaluation. Always preserve `0 *_C C = 0`; a saturated
intermediate must not force rejection if later multiplication by zero makes it irrelevant.
Apply the same exact capped semantics in Rust and Lean. No subtraction or signed cancellation is
permitted in this numeric bound calculation.

If the capped final result is less than `C`, it equals the uncapped bound exactly. Otherwise the
candidate fails this sufficient bound. Accepted artifacts report that exact final value.
Intermediate integers need only `O(log C)` bits; operation cost still includes big-integer arithmetic
and the bit length of `q`. This removes the hidden `O(L)` output-size cost for rejected huge candidates.

### 9.3 Prevent hidden expansion in Lean

Keep family and loop semantics abstract in application proofs. Do not use `decide`, `simp`, or
definitional reduction to enumerate `Fin N`, evaluate the runtime circuit, unfold a concrete
iteration count, expand quotient coefficients, or materialize every primitive occurrence.

Use named body lemmas and the generic map/fold rules. Static node lookup proofs may inspect the
stored syntax, but avoid repeated whole-graph unfolding and quadratic scans. Reuse generated scope
definitions and local node equations. A kernel proof of a numerical candidate uses verified capped
binary exponentiation, not reduction of the semantic loop.

## 10. Search and verification API

Do not create another universal noise-simulator crate. The application's Rust code implements its
own small numeric bound function and prefilter. Its Lean package independently proves the bound for
the generated graph and checks the candidate inequality. Agreement follows at verified acceptance,
not from trusting a shared handwritten Rust expression interpreter.

Planned Rust-facing types:

```rust
pub enum CandidateCheck {
    RejectedByBound { capped_bound: BigUint, threshold: BigUint },
    Verified {
        parameters: DiamondParameters,
        noise_bound: BigUint,
        artifact: LeanArtifact,
    },
}

pub fn check_candidate(
    linked: &LinkedFrozenProgram,
    params: &DiamondParameters,
) -> Result<CandidateCheck, VerificationError>;
```

`LeanArtifact` holds the source/program digest, parameter binding, theorem name, relevant toolchain
and dependency digests, and output location. It is a process/cache receipt, not a new semantic proof
language. In the first implementation it need not cache anything; add caching only after a real
generated theorem succeeds.

`VerificationError` separates unsupported semantics, export/type failure, Lean failure, timeout,
and stale artifact. None becomes bound rejection or verified acceptance. A Rust bound underestimate
must fail Lean's independently checked bound/acceptance equality. Overestimates may conservatively
reject candidates and should be caught by ordinary differential unit tests.

Do not assume acceptance is monotone in CRT depth when ring size, digits, sampler bounds, and
security-selected parameters also change. Existing binary search may remain a heuristic for finding
a candidate, but every returned candidate must be independently verified. Claims of minimality or
exhaustion require monotonicity for that searched family or an exhaustive policy.

## 11. Ownership and libraries

| Location | Contents |
| --- | --- |
| `crates/ir-core/lean/MxxIR` | Small relational combinators, parameter arithmetic, pointwise/fold rules |
| `crates/ir-core/src/lean` | Mechanical frozen-IR exporter |
| `crates/primitives/lean/MxxPrimitives` | Quotient ring, coefficient view, reduction, exact matrix norms |
| `crates/runtime/lean/MxxRuntime` | Concrete primitive relations and fixed hash interpretation interface |
| `crates/gadgets/lean/MxxGadgets` | Gadget reconstruction, preimage consumption, input-injector algebra |
| `crates/bgg/lean/MxxBgg` | Full BGG+ invariant, compatible-secret multiplication and gate lemmas |
| `crates/we/lean` | Generated Diamond program, application scope proofs, bound, decoder, final theorem |
| `crates/we/src/diamond/correctness` | Application bound calculator, candidate invocation, artifact receipt |

The relational combinators are polymorphic in ordinary Lean input/output types, so `ir-core` need
not depend on crypto libraries. Generated modules import the concrete runtime primitive semantics;
application theorems do not quantify over arbitrary unspecified primitive rules. If byte/blob
operations reached by Diamond need semantics,
provide explicit abstract domains and deterministic operations, never a default `Unit` or zero value.

Use Lean and mathlib directly. The official `Std.Do`/`mvcgen` system is an optional automation layer:
it generates verification conditions and leaves invariants to the proof author. It does not infer
cryptographic noise laws. A direct relational scope theorem requires very little framework, so
`mvcgen` is not a mandatory dependency of this design. If later adopted, prove its adequacy bridge
to the same generated relation semantics once, and measure its effect on proof size.

Sources for that library assessment:
[Lean verification conditions](https://lean-lang.org/doc/reference/latest/The--mvcgen--tactic/Verification-Conditions/)
and [Lean predicate transformers and adequacy](https://lean-lang.org/doc/reference/latest/The--mvcgen--tactic/Predicate-Transformers/).
Toolchain support must be checked against the pinned version; these documentation pages are not a
claim that the old worktree's Lean version exposes every current API.

## 12. Why this is a smaller solution

The program has to describe what was computed. The mathematics has to explain why that computation
has the desired error. A global trace adds a third task: recovering the same local computation from
an occurrence log. The new relational semantics makes local operands and outputs available at the
point where the proof needs them.

Families do not become a collection of discovered facts. Their meaning is a function and a universally
quantified relation. Index composition is handled by ordinary substitution. Sequential execution has
an invariant and an induction theorem; its bound uses a separate fixed-size numeric recurrence.

The design still has real costs: faithful typed extraction for all reached primitives, the small
family/iteration proof library, and application mathematics. It does not promise to automate those away.
It removes the trace-recovery and generic symbolic-discovery problems that were overwhelming them.

Alternatives considered:

- Repairing the old trace implementation keeps large occurrence/type transports on the proof path.
- Direct shallow relational extraction is selected because the exporter was already outside the
  formal trust boundary. It removes the second Lean AST and the typed environment interpreter.
- A shallow deterministic executable would also need an oracle/effect representation for sampling.
  The relational definition expresses bounded successful results more directly for this theorem.
- A deep Lean graph interpreter plus an intrinsically typed compiler could reduce translation trust
  only with a verified correspondence bridge. It adds scope that the initial theorem does not require.
- A generic proof certificate plus replay reintroduces a second collection of facts whose connection
  to the source graph must itself be proved.
- Carrier annotations, special bit-recomposition nodes, e-graphs, and module contracts in the DSL
  reintroduce the abandoned semantic-discovery/duplication burden.

## 13. Acceptance gates and remaining design risk

Before calling this design implemented, require:

1. one exported sampler/preimage graph with a proved local noise theorem;
2. one parallel family and nested gather with a quantified proof, including zero count and duplicate
   selectors;
3. one carried-state loop with a single step proof and capped logarithmic bound evaluation;
4. artifact producer/consumer identity preserved without occurrence records;
5. the full BGG multiplication theorem with both error summands;
6. the generated Diamond output/decoder theorem for both Boolean messages;
7. malformed-wiring and arithmetic-bound mutation tests;
8. explicit measurements for `N,L = 1, 16, 2^10, 2^20`, showing no lane/iteration enumeration in
   generated syntax, proof instances, or checker work;
9. a full Lean build, no `sorry`/`admit`, and only the agreed standard logical axioms;
10. separate reporting of the formal IR theorem and any later authorized GPU validation.

The most important unresolved implementation risk is faithful typed relational extraction. The plan
tests it first with parameters, a sampler, multiple outputs, and a child loop. If it requires another
large hierarchy of application-specific views, stop and revise that boundary before porting Diamond.
The second risk is kernel cost for exact capped arithmetic; measure it on the prototype, not after
the full emitter and cache are written.

This document changes the proposed proof semantics and rejects the old trace-first implementation
plan. It does not authorize source implementation in the current task. Only this specification and
the companion plan are added after the requested pull.
