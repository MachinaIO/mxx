# Generated Lean correctness specification

## 1. Status and objective

This document specifies the first implementation of generated Lean correctness checking for the
typed executable IR. It is an implementation specification, not a description of currently
implemented behavior.

The first application target is the Diamond witness-encryption round trip. For every accepted
parameter candidate, Lean must prove that, whenever the supplied instance and witness make the
Boolean circuit evaluate to one, the decryption graph's noisy plaintext is close to the constant
polynomial `floor(q / 2) * message`. The proved error must be small enough for the exact Boolean
decoder interval used by the generated decryption IR. The decoded output must therefore equal the
encrypted message.

The protocol appearing in the theorem statement must be generated from the same frozen IR used by
the Rust runtime. Application code may provide a theorem template and application-specific proofs,
but it must not restate the encryption or decryption computation as a separate handwritten Lean
program.

This design deliberately does not rebuild a generic symbolic noise simulator. The generated IR has
an exact operational meaning. Reusable primitive, gadget, and BGG+ libraries prove local algebraic
and bound lemmas. Diamond-specific proofs choose the useful intermediate invariants and compose
those lemmas.

`docs/noise-simulator-spec.md` remains the description of the currently implemented numeric
simulator. Its statement that the simulator does not generate Lean remains true: Lean generation
is a separate crate-local correctness system specified here. When this design is implemented,
`docs/architecture.md` and the parameter-search section of `docs/noise-simulator-spec.md` must be
updated in the same change; they must not claim that simulator acceptance alone is the final
Diamond correctness verdict.

## 2. Scope

### 2.1 Required in the first implementation

The first implementation must provide:

1. a concrete-only Lean representation of the reached frozen encryption and decryption graphs;
2. a Rust emitter that produces that representation from linked `mxx-ir-core::ValidatedGraph`
   values and their concrete `ParamEnv` bindings;
3. generated, typed references to the noisy-plaintext and decoded outputs;
4. exact Lean semantics for every IR operation reached by the Diamond graphs;
5. reusable exact and coefficient-bound lemmas at the owning primitive, gadget, or BGG+ layer;
6. an application-specific Diamond proof over the generated program;
7. an exact Boolean decoder-radius definition equivalent to the current four-condition parameter
   acceptance check;
8. a candidate-check entry point usable by `DiamondParameterSearch`;
9. regeneration, stale-output, no-`sorry`, and axiom gates; and
10. one generated and checked theorem for the parameter set used by the Diamond integration test.

### 2.2 Not required initially

The first implementation does not have to prove:

- correctness of the Rust parser, graph freezer, Rust-to-Lean emitter, Rust executor, CUDA kernels,
  DCRT representation, or lattice-security estimator;
- a probability bound for the sampler good event;
- correctness for a rejecting witness;
- correctness of every application in the workspace;
- a generic algorithm that discovers BGG+, GSW, carrier, or cancellation invariants;
- a generic application-independent noise-bound checker; or
- equality between concrete GPU memory contents and the Lean semantic value.

The initial theorem is a theorem about the exact generated frozen-IR semantics. Runtime-versus-IR
implementation refinement is a separate future scope.

## 3. Trust boundary and protocol identity

### 3.1 Authoritative source

The authoritative protocol object is the validated frozen `Graph` plus its concrete `ParamEnv`.
The emitter must consume that object directly. It must not reconstruct the graph by calling BGG+ or
Diamond helper functions a second time.

Before emission, the linker validates every artifact input against the actual producer manifest:
the full `ProductionId`, artifact name, type, shape, and confidentiality must match. It then builds
a mathematical linked program in which each validated artifact input is replaced by a typed
reference to the producer-stage output. The execution nonce and artifact content provenance do not
occur in this mathematical program.

The emitter computes, rather than accepts from the caller, the identities:

```rust
pub struct LeanSemanticIdentity {
    pub ir_version: u32,
    pub linked_program_sha256: [u8; 32],
}
```

`linked_program_sha256` hashes the canonical concrete program after artifact-link resolution. It
commits to every operation, type, parameter binding, structural scope, typed producer output, and
consumer use, but not to `ProductionId.execution_nonce`. Erasing the nonce is allowed only after
the full runtime link has been validated. Any other difference remains hash-visible.

The raw producer and consumer `SpecHash`/`ProductionId` values remain runtime-integrity data. They
may be logged beside a particular execution, but are neither emitted into checked-in Lean source
nor used as a reusable proof-cache key. This lets one theorem cover repeated executions of the
same concrete protocol without claiming that two different linked computations are identical.

### 3.2 Generated theorem target

Every final theorem refers to its generated candidate program definition. For the checked-in
fixture this is:

```lean
Mxx.We.Golden.DiamondWE.candidate.program
```

and must obtain the values under test through generated typed output references:

```lean
Mxx.We.Golden.DiamondWE.candidate.refs.noisyPlaintextOutput
Mxx.We.Golden.DiamondWE.candidate.refs.decodedOutput
```

The theorem must not contain a separately handwritten definition of Diamond encryption,
decryption, or the noisy plaintext expression.

### 3.3 Stale-generation rule

The emitter supports two modes:

```text
--write    atomically replace generated files
--check    regenerate in memory and fail on any byte difference
```

CI and correctness checks use `--check`. A checked-in generated theorem is invalid if the current
frozen graph, parameter environment, output reference, or application claim template regenerates a
different file.

The generated file header contains the IR version and `linked_program_sha256`. It contains no raw
execution nonce, timestamp, or absolute path.

## 4. Repository and Lean package layout

Proofs live beside the Rust abstraction they justify. There is no repository-wide directory that
accumulates all mathematical definitions and application proofs.

```text
crates/ir-core/lean/
  lakefile.toml
  MxxIrCore/
    Expr.lean
    Types.lean
    Program.lean
    Eval.lean
    Family.lean
    Structural.lean

crates/primitives/lean/
  lakefile.toml
  MxxPrimitives/
    Negacyclic.lean
    Reduction.lean
    Matrix.lean
    Bounds.lean
    Sampling.lean
    Preimage.lean
    Radix.lean

crates/runtime/lean/
  lakefile.toml
  MxxRuntime/
    IrBackend.lean
    LinkedProgram.lean
    Artifacts.lean

crates/gadgets/lean/
  lakefile.toml
  MxxGadgets/
    GadgetMatrix.lean
    Decomposition.lean
    InputInjector.lean
    BooleanCircuit.lean

crates/bgg/lean/
  lakefile.toml
  MxxBgg/
    Encoding.lean
    PublicKey.lean
    Multiplication.lean
    Boolean.lean

crates/we/lean/
  lakefile.toml
  MxxWe/
    DiamondWE/
      Model.lean
      Exact.lean
      Noise.lean
      Decoder.lean
      Correctness.lean
    Golden/
      DiamondWE/
        Program.lean
        Parameters.lean
        Claim.lean
        Check.lean

lean-toolchain

crates/we/test-data/
  diamond_lean_golden.json
```

The Lake dependency direction mirrors the Cargo dependency direction. IR and primitives are
independent lower layers; runtime is their bridge:

```text
MxxIrCore       MxxPrimitives
       \         /
        MxxRuntime

MxxPrimitives -> MxxGadgets -> MxxBgg
          MxxRuntime, MxxBgg, and MxxGadgets -> MxxWe
```

`MxxIrCore` must not import `MxxPrimitives`. Its evaluator is parameterized by a semantic backend.
`MxxRuntime` imports both packages and supplies the concrete polynomial/matrix backend used by the
generated Diamond program. BGG+ algebraic lemmas operate directly on `MxxPrimitives` values and do
not depend on the runtime evaluator.

`crates/we/lean/lakefile.toml` uses path dependencies on the lower-layer Lean packages. Building
`MxxWe` therefore checks all transitive proof libraries. No proof source is placed in a top-level
`lean/` directory.

The repository-root `lean-toolchain` pins the Lean version and is discovered from every
`crates/*/lean` working directory. The two independent lowest Lake packages pin the same exact
mathlib Git revision, and CI rejects revision disagreement; all higher packages receive mathlib
transitively. Floating branches are forbidden.

`MxxWe/Golden/DiamondWE` is the single checked-in regeneration fixture. The reusable
`MxxWe.DiamondWE` library does not import it. Parameter search generates a cache-local Lake package
under `target/mxx-lean/diamond/<semantic-hash>/` with a unique namespace, imports the reusable
library by path, and checks that candidate there. It never shadows or overwrites the checked-in
golden modules.

If `mxx-noise-simulator` remains as a fast candidate prefilter, it does not become part of the Lean
trust chain. If a future generic checker is made authoritative, its checker-soundness proof belongs
under `crates/noise-simulator/lean/`; application theorems still remain in their application crates.

## 5. Concrete generated IR model

### 5.1 Concrete-only expressions

Rust resolves every compile-time parameter in a reached `IntExpr`, `RealExpr`, matrix dimension,
family extent, loop count, and sampler cutoff before Lean generation. The generated program uses
closed expression ASTs for genuinely structural values: their only non-literal leaves are declared
loop/grid slots. Thus a family map may depend on its current coordinate, while no parameter name or
undeclared dynamic variable reaches Lean.

Each structural slot is declared by its enclosing loop/grid with its kind and finite exclusive upper
bound. The emitter validates slot scope and bounds before rendering; it never evaluates a structural
expression using a default value or silently drops it. Expressions required to be compile-time
constants (matrix dimensions, sampler cutoffs, and type shapes) must be literal after substitution.

Generation fails if any reached expression cannot be resolved or if the graph is not valid under
the supplied `ParamEnv`.

### 5.2 Lean wire types

`crates/ir-core/lean/MxxIrCore/Types.lean` defines:

```lean
namespace Mxx.IR

structure MatrixType where
  modulus : Int
  ringDimension : Nat
  rows : Nat
  columns : Nat

def MatrixType.Valid (matrixType : MatrixType) : Prop :=
  1 < matrixType.modulus ∧
  0 < matrixType.ringDimension ∧
  0 < matrixType.rows ∧
  0 < matrixType.columns

structure TrapdoorType where
  matrix : MatrixType
  sigma : RealExpr
  gadgetBase : StructuralIntExpr
  digitCount : StructuralIntExpr
  preimageMaxCoefficientBound : StructuralIntExpr

def TrapdoorType.Valid (trapdoorType : TrapdoorType) : Prop :=
  trapdoorType.matrix.Valid ∧
  0 < trapdoorType.sigma ∧
  1 < trapdoorType.gadgetBase ∧
  0 < trapdoorType.digitCount

inductive WireType
  | constantInt
  | constantReal
  | constantBool
  | int
  | real
  | bool
  | bytes (length : Nat)
  | typedBlob (typeName : String) (schemaHash : List UInt8)
  | matrix (matrixType : MatrixType)
  | trapdoor (trapdoorType : TrapdoorType)
  | preimage (matrixType : MatrixType)
  | family (shape : List Nat) (element : WireType)

structure WireRef where
  scope : Nat
  node : Nat
  port : Nat
  deriving DecidableEq, Repr

end Mxx.IR
```

This is the concrete counterpart of every `ConcreteWireType` variant, including constant-valued
wires, exact rationals, typed blobs, and trapdoors. Numeric data and validity proofs are
deliberately separate so the types remain simple serializable data. The generated program contains
one decidable `Program.Valid` proof covering all reached matrix, trapdoor, and family types. A Rust
inventory test must fail when a newly reached concrete wire or node variant has no emitter and Lean
semantics; omission is not treated as an opaque value.

The generated concrete dimensions are part of `MatrixType`. Proofs may introduce readable local
abbreviations such as `SecretRows` or `GadgetColumns`, but those names do not replace the generated
dimensions.

### 5.3 Families

Logical family axes are represented without flattening:

```lean
def Mxx.IR.FamilyIndex : List Nat → Type
  | [] => Unit
  | extent :: rest => Fin extent × FamilyIndex rest

def Mxx.IR.Family (shape : List Nat) (element : Type) : Type :=
  FamilyIndex shape → element
```

`FamilyPack`, static/dynamic get, selection, gather, reindex, and parallel-grid semantics operate on
`FamilyIndex`. The Rust emitter converts the frozen row-major maps to functions on this index type.

Families use uniform application assumptions and uniform bounds in the first implementation. A
family bound never depends on its selected index.

The structural semantics are definitionally:

```text
FamilyPack(shape, flat)[u]       = flat[rowMajorOffset(shape, u)]
FamilyGetStatic(X, u)            = X[u]
FamilyGetDynamic(X, u)           = X[u]
FamilySelectAxis(X, a, s)[u]     = X[insertAxis(u, a, s[u])]
FamilyReindex(X, f)[u]           = X[f(u)]
FamilyGather(X, s_0, ..., s_r)[u]= X[s_0[u], ..., s_r[u]]
Select(selector, x_0, ..., x_k)  = x_selector
```

For `FamilySelectAxis`, a scalar selector is treated as the constant function `s[u]`; a selector
family must have exactly the output shape obtained by removing axis `a`. Static coordinates and
index maps are proved in range by `LinkedProgramData.Valid`. Dynamic coordinates and `Select`
indices are checked during evaluation and return `EvalError.indexOutOfRange` when invalid. These
operations preserve the selected value itself, including a `PreimageValue`; they do not weaken or
invent sampler facts. Uniform family bounds let the proof select any valid index without a new
bound expression.

### 5.4 Program data

The generated program is data, not only a sequence of handwritten Lean `let` expressions:

```lean
structure Mxx.IR.Node where
  kind : ConcreteNodePayload
  arguments : Array WireRef
  outputs : Array WireType

structure Mxx.IR.Scope where
  id : Nat
  structuralSlots : Array StructuralSlotDecl
  nodes : Array Node
  inputs : Array WireRef
  outputs : Array WireRef

structure Mxx.IR.Stage where
  name : String
  bindings : Array (String × Int)
  scopes : Array Scope
  root : Nat
  namedOutputs : Array NamedOutput

structure Mxx.IR.LeanSemanticIdentity where
  irVersion : Nat
  linkedProgramSha256 : ByteArray

structure Mxx.IR.LinkedProgramData where
  identity : LeanSemanticIdentity
  stages : Array Stage
  artifactLinks : Array ArtifactLink

def Mxx.IR.LinkedProgramData.Valid (data : LinkedProgramData) : Prop :=
  -- finite acyclic node order, valid structural scopes and occurrence paths,
  -- type-correct arguments/outputs, in-range indices, and valid typed artifact links
  ...

structure Mxx.IR.LinkedProgram where
  data : LinkedProgramData
  valid : data.Valid
```

`LinkedProgram` always means this validated subtype. Convenience projections expose
`program.identity`, `program.stages`, and `program.artifactLinks` from `program.data`; no public
evaluator accepts raw `LinkedProgramData`.

Stage integer bindings record concrete bindings used to freeze the stage. Real values are emitted
as closed `RealExpr` expressions in node and trapdoor descriptors; no unresolved parameter name is
emitted.
`ConcreteNodePayload` is a closed, exhaustive counterpart of the reached
`mxx_ir_core::NodeKind` variants. Every numeric, range, family-map, hash-tag, sampler, structural,
literal, and artifact field is retained in closed form. It omits source locations and construction
identities. Unsupported reached variants make generation fail; they are never emitted as opaque
operations or represented by a placeholder/default.

### 5.5 Typed references

Generated references carry a checked type:

```lean
structure Mxx.IR.TypedWireRef
    (program : LinkedProgram)
    (wireType : WireType) where
  stage : Fin program.stages.size
  wire : WireRef
  typeCorrect : program.wireType stage wire = some wireType
```

The emitter fills `typeCorrect` with a decidable proof. Application proof code cannot use a wire
from a different generated program or silently treat a Boolean, family, or differently shaped
matrix as the claimed output.

### 5.6 Inputs, samples, and evaluation

Input and sampler nodes obtain values from an evaluation environment. Sampler results are not
computed probabilistically inside Lean. `MxxIrCore` remains independent of the concrete
polynomial implementation through this interface:

```lean
structure Mxx.IR.SemanticBackend where
  denoteMatrix : MatrixType → Type
  denoteTrapdoor : TrapdoorType → Type
  denotePreimage : MatrixType → Type
  denoteTypedBlob : String → ByteArray → Type
  matrixZero : ...
  matrixIdentity : ...
  matrixAdd : ...
  matrixSubtract : ...
  matrixMultiply : ...
  matrixScale : ...
  trapdoorPublic : ...
  materializePreimage : ...
  applyPreimage : ...
  -- one field for every non-structural reached primitive
```

The concrete signatures are dependent on the validated input and output `MatrixType`s. Every
operation returns the type declared by its node. `MxxRuntime.IrBackend` instantiates this interface
with `MxxPrimitives.ExactMatrix`. The certified Diamond evaluator uses
`MxxRuntime.irBackendWithGadgetOracle oracle`; the oracle is an explicit argument of every runtime
backend, value, environment, trace, and dependent application predicate. It is never inferred from
`Candidate`, so two different gadget interpretations cannot be mixed accidentally.

The runtime proof package uses explicit wrappers:

```lean
structure Mxx.Runtime.TrapdoorValue (trapdoorType : Mxx.IR.TrapdoorType) where
  privateState : Mxx.Runtime.AbstractTrapdoor trapdoorType
  publicMatrix : Mxx.Primitives.ExactMatrix
    trapdoorType.matrix.modulus
    trapdoorType.matrix.ringDimension
    trapdoorType.matrix.rows
    trapdoorType.matrix.columns

structure Mxx.Runtime.PreimageValue (matrixType : Mxx.IR.MatrixType) where
  matrix : Mxx.Primitives.ExactMatrix
    matrixType.modulus matrixType.ringDimension
    matrixType.rows matrixType.columns
```

`denoteTrapdoor` and `denotePreimage` are these wrappers. `TrapdoorPublic` projects
`publicMatrix`; `MaterializePreimageExact` projects `matrix`; and `ApplyPreimage left preimage`
computes ordinary matrix multiplication by `preimage.matrix`. `PreimageBinary` performs the stated
matrix operation inside the wrapper. The exact relation is deliberately not an unchecked field of
`PreimageValue`; it is occurrence-specific evidence supplied by `GoodSamples` or proved for a
deterministic gadget decomposition. The evaluator's gadget milestone is contract-parametric: a
`GadgetLayout` records regular versus small mode, base, digit count, and source/target dimensions;
the Diamond path accepts only regular layouts with positive dimensions and matching shapes. A
The backend supplies a dependent certificate family indexed by the exact gadget value, target,
and returned preimage; `gadgetDecompose` returns the preimage together with a lifted certificate.
The runtime's certified backend therefore takes a lawful gadget oracle as an explicit parameter.
The current unsigned radix routine is retained only as an experimental fixture and is not a
Rust/DCRT-exact or certified implementation. Any concrete Rust refinement must separately prove
the same `RightPreimage` reconstruction and `PreimageWithin (base - 1)` bound.

```lean
def Mxx.IR.WireType.denote
    (backend : SemanticBackend) : WireType → Type
  | .constantInt => Int
  | .constantReal => RealExpr
  | .constantBool => Bool
  | .int => Int
  | .real => RealExpr
  | .bool => Bool
  | .bytes length => Fin length → UInt8
  | .typedBlob typeName schemaHash =>
      backend.denoteTypedBlob typeName schemaHash
  | .matrix matrixType => backend.denoteMatrix matrixType
  | .trapdoor trapdoorType => backend.denoteTrapdoor trapdoorType
  | .preimage matrixType => backend.denotePreimage matrixType
  | .family shape element => Family shape (element.denote backend)

abbrev Mxx.IR.DynamicValue (backend : SemanticBackend) :=
  Σ wireType, wireType.denote backend
```

`DynamicValue` is confined to the evaluator's internal node table. Public proof code obtains a
value only through a `TypedWireRef`, so casts and failed downcasts do not leak into BGG+ or Diamond
proofs.

```lean
structure Mxx.IR.InstantiationFrame where
  call : WireRef
  iteration : Option Nat

structure Mxx.IR.WireOccurrence where
  stage : Nat
  instantiationPath : List InstantiationFrame
  wire : WireRef

structure Mxx.IR.SampleRef (program : LinkedProgram) extends WireOccurrence where
  validOccurrence : program.ValidOccurrence toWireOccurrence
  isSampler : program.IsSampler toWireOccurrence

structure Mxx.IR.EvalEnv
    (backend : SemanticBackend)
    (program : LinkedProgram) where
  externalInput :
    (ref : program.ExternalInputRef) → program.denote backend ref.wireType
  sampleOutput :
    (ref : program.SampleRef) → program.denote backend ref.wireType

def Mxx.IR.eval
    (backend : SemanticBackend)
    (program : LinkedProgram)
    (env : EvalEnv backend program) :
    Except (EvalError program) (program.Trace backend)
```

`eval` is total by structural recursion over the topological node order and structural-scope
well-foundedness contained in `program.valid`. It has no fuel parameter, default value, partial
branch, or panic case. Runtime-dependent out-of-range selectors, family coordinates, or malformed
external values return an explicit `EvalError`, matching the Rust executor's rejecting behavior.
`Trace`, typed references, occurrence validity, and node-equation theorems all take the same
validated `LinkedProgram`. Node-equation theorems take an
`hEval : eval ... = .ok trace`. The current Diamond claim is conditional on a `GoodRunPromise`
that contains such a successful trace together with its occurrence-specific sampler facts. It does
not independently construct evaluation success.

Artifact inputs are not independent environment fields. `eval` obtains them from the exact linked
producer-stage output. This preserves the encryption/decryption artifact identity in the theorem.

`InstantiationFrame` mirrors the runtime `WireId.instantiation_path`. Its `iteration` is the
row-major flattened lane used by the Rust executor for a rank-N grid, or the sequential iteration
number; `none` denotes a non-iterated subgraph call. `ValidOccurrence` reconstructs the rank-N grid
coordinate from that lane and proves it is in range. Evaluation appends a frame whenever it enters
a subgraph, grid coordinate, or sequential-loop iteration; nested structures therefore produce
distinct `SampleRef`s even when they execute the same static sampler node.
`ValidOccurrence` proves that every coordinate is in range and that the path reaches the stated
static wire. All sampler assumptions quantify over `SampleRef`, never merely over a static node.

### 5.7 Sampler descriptors

The IR emitter derives sampler metadata from node kinds and typed arguments:

```lean
structure Mxx.IR.MatrixSamplerDescriptor (program : LinkedProgram) where
  matrixType : MatrixType
  staticOutput : program.StaticWireRef (.matrix matrixType)
  cutoff : Nat

structure Mxx.IR.PreimageSamplerDescriptor (program : LinkedProgram) where
  sourceType : MatrixType
  preimageType : MatrixType
  targetType : MatrixType
  staticOutput : program.StaticWireRef (.preimage preimageType)
  staticSourceArgument : program.StaticWireRef (.matrix sourceType)
  staticTargetArgument : program.StaticWireRef (.matrix targetType)
  cutoff : Nat
  shapesValid : PreimageShapesValid sourceType preimageType targetType

structure Mxx.IR.FamilyPreimageSamplerDescriptor (program : LinkedProgram) where
  groupShape : List Nat
  branchShape : List Nat
  sourceType : MatrixType
  preimageType : MatrixType
  targetType : MatrixType
  staticOutput : program.StaticWireRef
    (.family (groupShape ++ branchShape) (.preimage preimageType))
  staticSourceArgument : program.StaticWireRef
    (.family groupShape (.matrix sourceType))
  staticTargetArgument : program.StaticWireRef
    (.family (groupShape ++ branchShape) (.matrix targetType))
  cutoff : Nat
  shapesValid : PreimageShapesValid sourceType preimageType targetType

structure Mxx.IR.SamplerDescriptors (program : LinkedProgram) where
  ordinaryMatrices : Array (MatrixSamplerDescriptor program)
  preimages : Array (PreimageSamplerDescriptor program)
  familyPreimages : Array (FamilyPreimageSamplerDescriptor program)
```

`descriptor.occurrences` is the subtype of `SampleRef program` whose static stage/output matches
the descriptor. For each ordinary preimage occurrence, `sourceOccurrence` and `targetOccurrence`
reuse the same instantiation path and resolve the node's actual argument bindings.

A `FamilyPreimageSample` is one aggregate sampler occurrence whose `sampleOutput` is the complete
family value. Its sampler fact quantifies separately over
`GroupIndex = FamilyIndex groupShape` and `BranchIndex = FamilyIndex branchShape`. For group `i`
and branch `j`, the equation is `B_i * K_i,j = T_i,j`: the source may vary with `i`, but is
independent of `j`. Lean does not create extra `SampleRef`s for the runtime executor's synthetic
per-lane draws; those draws jointly implement this one mathematical family output. A future
runtime-refinement proof must relate the flattened runtime lanes to the aggregate family by the
same row-major index map.

Trapdoor sampler occurrences remain present in `EvalEnv`, but require no standalone magnitude
assumption; every use relevant to correctness is constrained by the relation fact of a downstream
preimage-sampler occurrence.

For proof performance, the emitter may additionally generate one reducible accessor per reached
node and a theorem equating it to the successful `Mxx.IR.eval` trace. The canonical
`LinkedProgram` and generic evaluator remain the statement authority.

## 6. Polynomial and matrix model

### 6.1 Exact and witness rings

The exact runtime ring and the integer witness ring are distinct:

```lean
structure Mxx.Primitives.Negacyclic (n : Nat) (R : Type u) where
  coeff : Fin n → R

abbrev Mxx.Primitives.ExactPoly (q n : Nat) :=
  Negacyclic n (ZMod q)

abbrev Mxx.Primitives.ErrorPoly (n : Nat) :=
  Negacyclic n Int
```

`Negacyclic` has pointwise addition and negacyclic convolution implementing `X^n = -1`. The
primitive package proves its commutative-ring instance once.

Reduction is an actual ring homomorphism:

```lean
def Mxx.Primitives.reducePoly (q n : Nat) :
    ErrorPoly n →+* ExactPoly q n
```

No proof extracts noise by applying a non-homomorphic centered lift to an arbitrary exact value.
Instead, every small or noisy value carries an integer witness whose reduction is the exact value.

### 6.2 Matrices

```lean
abbrev Mxx.Primitives.ExactMatrix (q n rows columns : Nat) :=
  Matrix (Fin rows) (Fin columns) (ExactPoly q n)

abbrev Mxx.Primitives.ErrorMatrix (n rows columns : Nat) :=
  Matrix (Fin rows) (Fin columns) (ErrorPoly n)

def Mxx.Primitives.reduceMatrix (q n rows columns : Nat) :
    ErrorMatrix n rows columns → ExactMatrix q n rows columns
```

Matrix multiplication uses mathlib `Matrix`. Shape-changing operations use explicit `Fin`
reindexing functions generated from the validated IR ranges and family maps.

Lean matrices remain symbolic functions. The checker must not materialize a matrix with
`ringDimension * rows * columns` numeral entries merely to prove a protocol theorem.

### 6.3 Coefficient norm

```lean
def Mxx.Primitives.polyNorm (x : ErrorPoly n) : Nat :=
  Finset.univ.sup fun i => Int.natAbs (x.coeff i)

def Mxx.Primitives.matrixNorm (x : ErrorMatrix n rows columns) : Nat :=
  Finset.univ.sup fun row =>
    Finset.univ.sup fun column =>
      polyNorm (x row column)
```

The primitive library proves:

```lean
polyNorm (x + y) ≤ polyNorm x + polyNorm y
polyNorm (-x) = polyNorm x
polyNorm (x * y) ≤ n * polyNorm x * polyNorm y

matrixNorm (a * b) ≤
  innerDimension * n * matrixNorm a * matrixNorm b
```

It also proves constant-polynomial variants in which the convolution factor is one. Application
proofs select those stronger lemmas explicitly; constantness is not inferred by a global abstract
interpreter.

## 7. Bound representation

### 7.1 Trusted numeric type

All correctness bounds and thresholds are `Nat`. Rust `BigUint` values are emitted as decimal
natural-number literals. Trusted correctness code contains no `Float` or approximate real
comparison.

Sampler sigma values may be retained as exact rationals for provenance, but sampler cutoffs and
all values consumed by the correctness proof are integer upper bounds.

### 7.2 Bound expression

The shared bound language is intentionally small:

```lean
inductive Mxx.Primitives.BoundExpr (Parameter : Type)
  | literal (value : Nat)
  | parameter (name : Parameter)
  | add (left right : BoundExpr Parameter)
  | mul (left right : BoundExpr Parameter)
  | max (left right : BoundExpr Parameter)

def Mxx.Primitives.BoundExpr.eval
    (environment : Parameter → Nat) :
    BoundExpr Parameter → Nat
```

There is no subtraction, division, or floating-point operation in a noise-bound expression. Any
analytic quantity is converted to a proved safe integer cutoff before entering this language.

The Rust side has an isomorphic serializable `BoundExpr<Parameter>` and exact `BigUint` evaluator.
The application-specific Rust checker and generated Lean proof therefore consume the same bound
tree rather than independently transcribing a formula.

### 7.3 Approximation and bounded lifts

```lean
structure Mxx.Primitives.Approx
    (actual ideal : ExactMatrix q n rows columns) where
  error : ErrorMatrix n rows columns
  equation : actual = ideal + reduceMatrix error

def Mxx.Primitives.ApproxWithin
    (actual ideal : ExactMatrix q n rows columns)
    (bound : Nat) : Prop :=
  ∃ error,
    actual = ideal + reduceMatrix error ∧
    matrixNorm error ≤ bound

structure Mxx.Primitives.BoundedLift
    (actual : ExactMatrix q n rows columns)
    (bound : Nat) where
  witness : ErrorMatrix n rows columns
  reduce_eq : reduceMatrix witness = actual
  norm_le : matrixNorm witness ≤ bound
```

`BoundedLift` represents bounded secrets, binary matrices, Gaussian matrices, preimages, and any
other exact value whose integer magnitude is used in a later error product. Arbitrary large public
matrices need no lift unless an application proof uses them as a bounded multiplier.

To construct product-error witnesses such as `L * E`, every exact multiplier used by a noise lemma
has an integer lift. The complete fact used by multiplication lemmas is:

```lean
inductive Mxx.Primitives.SupportClass
  | constant
  | arbitrary

structure Mxx.Primitives.MagnitudeFact
    (actual : ExactMatrix q n rows columns) where
  lift : ErrorMatrix n rows columns
  reduce_eq : reduceMatrix q n rows columns lift = actual
  bound : Nat
  norm_le : matrixNorm lift ≤ bound
  support : SupportClass
  support_valid :
    support = .constant →
      ∀ row column coefficient,
        coefficient.val ≠ 0 → (lift row column).coeff coefficient = 0
```

Small secrets, preimages, and digits use small bounds. Uniform or hash-generated exact matrices
may use their canonical centered coefficient lift with bound `floor(q / 2)`. That centered lift is
only required to reduce back to the exact value; it is never treated as a ring homomorphism.

## 8. Sampling and relation facts

### 8.1 Good-sample environment

Every sampler node has an exact trace value. The generated application predicate relates that
value to the facts promised by that sampler operation. An ordinary bounded sampler contributes a
lift and the node's declared cutoff:

```lean
structure Mxx.Primitives.SampleFact
    (actual : ExactMatrix q n rows columns)
    (bound : Nat) extends BoundedLift actual bound
```

The generated Diamond predicate is a structure rather than an untyped conjunction:

```lean
structure Mxx.We.DiamondWE.GoodSamples
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : Mxx.IR.EvalEnv (RuntimeBackend oracle) candidate.program.data)
    (trace : Mxx.IR.Trace (RuntimeBackend oracle)) : Prop where
  occurrence : ∀ sample : Mxx.IR.SampleRef candidate.program.data,
    sample.Reached trace → SamplerOccurrenceFact oracle candidate env trace sample

structure Mxx.We.DiamondWE.GoodRunPromise
    (oracle : Mxx.Runtime.RuntimeGadgetOracle)
    (candidate : Candidate)
    (env : Mxx.IR.EvalEnv (RuntimeBackend oracle) candidate.program.data) where
  trace : Mxx.IR.Trace (RuntimeBackend oracle)
  evaluated : Mxx.IR.eval (RuntimeBackend oracle) candidate.program env = .ok trace
  goodSamples : GoodSamples oracle candidate env trace
```

`SampleRef.Reached trace` means that the sampler's exact occurrence is present in the actual trace.
The quantifier is extensionally exhaustive: there is no caller-provided inventory and therefore no
empty-plan witness. `SamplerOccurrenceFact` is selected deterministically from the stored
`NodePayload`; it binds the environment output to the same trace occurrence and supplies the
operation-specific cutoff or preimage facts.

`GoodRunPromise oracle candidate env` packages one successful evaluator trace with `GoodSamples`
for that exact trace, under the same explicit `oracle`. The oracle is not a field of `Candidate` and
is not inferred from any typeclass or default backend.
The first correctness theorem is conditional on this successful good run. It does not prove that
every well-typed external environment evaluates successfully. Probability of `GoodSamples` and an
independent liveness or total-environment theorem are deferred.

### 8.2 Right preimage

The preimage fact is tied to the actual IR output value:

```lean
structure Mxx.Primitives.RightPreimage
    (source : ExactMatrix q n sourceRows inner)
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (actualTarget : ExactMatrix q n sourceRows targetColumns) where
  equation : source * actualPreimage = actualTarget
```

The relation contains exactly the runtime equation `B * K = T`. The proof-side split `T = P + E`
is represented separately by `Approx actualTarget targetIdeal`. Smallness of `K` is represented by
a `MagnitudeFact` or `BoundedLift` for `actualPreimage`. The IR relation therefore does not choose
an application-specific ideal target.

The generated occurrence sets are decidable predicates over `SampleRef`, not materialized arrays;
they include every valid dynamic execution of the corresponding static sampler nodes. Uniform
bounds mean the cutoff is independent of a family index, but sampled values and relation facts are
still occurrence- and index-specific.

For every `PreimageSample` occurrence, `PreimageOccurrenceFact` contains both

```text
RightPreimage
  (trace.get (descriptor.sourceOccurrence occurrence))
  (materializePreimage (env.sampleOutput occurrence.toSampleRef))
  (trace.get (descriptor.targetOccurrence occurrence))
PreimageWithin
  (materializePreimage (env.sampleOutput occurrence.toSampleRef))
  descriptor.cutoff
```

where the source and target occurrence references are derived from that concrete node's actual
dynamic arguments. Thus an
arbitrary environment-supplied sampler result cannot be used merely because it is small: the
conditional theorem also requires the actual `B * K = T` equation. `FamilyPreimageSample` emits the
analogous pointwise `RightPreimageFamily` fact: for each source group the source is independent of
the target-branch index, while the preimage and target vary by branch.

The exact and bound consumption lemma takes:

```text
RightPreimage source K T
Approx X (L * source)
Approx T P
MagnitudeFact L
MagnitudeFact K
```

and returns `Approx (X * K) (L * P)` with the integer error witness corresponding to
`L * E + eX * K`.

`P` is an unrestricted exact matrix. In particular, if the preimage target ideal contains a public
source such as `B` or `G`, that term remains inside `L * P` after consumption; the lemma never
drops or replaces it with zero. Only the explicitly witnessed target error `E` moves to the noise
side.

Bounds remain separate from the relation:

```lean
def Mxx.Primitives.PreimageWithin
    (actualPreimage : ExactMatrix q n inner targetColumns)
    (preimageBound : Nat) : Prop :=
  Nonempty (BoundedLift actualPreimage preimageBound)
```

The local consumption lemma proves the exact equation

```text
X = L * source + eX
source * K = P + E
---------------------------------------------
X * K = L * P + (L * E + eX * K).
```

It requires bounded lifts only for `L`, `E`, `eX`, and `K`. It does not require carrier metadata on
every IR value and does not track unrelated exact terms globally.

### 8.3 Gadget decomposition

Gadget decomposition reuses `RightPreimage`:

```lean
abbrev Mxx.Gadgets.GadgetDecomposition
    (gadget target decomposition) :=
  RightPreimage gadget decomposition target
```

Whether the trapdoor is public affects which IR operations can construct the value; it does not
change the algebraic relation type.

### 8.4 Radix digits

Radix decomposition is distinct from gadget preimage decomposition:

```lean
structure Mxx.Primitives.RadixSystem (q n : Nat) where
  Limb : Type
  instFintypeLimb : Fintype Limb
  weight : Limb → ErrorPoly n
  digit : ExactPoly q n → Limb → ErrorPoly n
  reconstruct :
    ∀ x,
      x = ∑ limb, reducePoly q n (weight limb * digit x limb)
  commonDigitBound : Nat
  digit_bound : ∀ x limb, polyNorm (digit x limb) ≤ commonDigitBound
```

Column-wise decompositions used by GSW and Fuse are represented by:

```lean
structure Mxx.Primitives.ColumnDigits
    (matrix : ExactMatrix q n rows columns)
    (Limb : Type)
    [Fintype Limb] where
  digit : Fin columns → Limb → ErrorPoly n
  route : Limb → Matrix (Fin rows) Unit (ExactPoly q n)
  reconstruct :
    ∀ column,
      matrixColumn matrix column =
        ∑ limb, reducePoly q n (digit column limb) • route limb
  commonDigitBound : Nat
  digit_bound : ∀ column limb, polyNorm (digit column limb) ≤ commonDigitBound
```

For a digit decomposition of `G`, `reconstruct` is the exact fact that makes `G` reappear after
homomorphic digit computation and recomposition. No carrier annotation or special
`recompose_bits` transfer is used.

### 8.5 Preimage families and selectors

The first implementation separates a source-group axis from a target-branch axis and uses common
bounds:

```lean
structure Mxx.Primitives.RightPreimageFamily
    (GroupIndex BranchIndex : Type)
    [Fintype GroupIndex] [Fintype BranchIndex] where
  source : GroupIndex → ExactMatrix q n sourceRows inner
  actualTarget :
    GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns
  actualPreimage :
    GroupIndex → BranchIndex → ExactMatrix q n inner targetColumns
  relation :
    ∀ group branch,
      RightPreimage (source group)
        (actualPreimage group branch) (actualTarget group branch)
  commonPreimageBound : Nat
  bounded :
    ∀ group branch,
      PreimageWithin (actualPreimage group branch) commonPreimageBound

structure Mxx.Primitives.TargetApproxFamily
    (actualTarget :
      GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns) where
  targetIdeal :
    GroupIndex → BranchIndex → ExactMatrix q n sourceRows targetColumns
  targetApprox :
    ∀ group branch,
      Approx (actualTarget group branch) (targetIdeal group branch)
  commonTargetNoiseBound : Nat
  bounded :
    ∀ group branch,
      matrixNorm (targetApprox group branch).error ≤ commonTargetNoiseBound
```

The sampler-generated `FamilyPreimageOccurrenceFact` contains only `RightPreimageFamily`, hence only
the actual relations and the common preimage bound. Diamond proves `TargetApproxFamily` separately
from the generated target-expression node equations; the IR sampler metadata is never asked to
invent an application-specific ideal target. Selecting `(group, branch)` is function application
and preserves both common bounds immediately.

## 9. BGG+ semantic structures

These structures are proof-side semantic witnesses. They are not additional DSL contracts.

```lean
structure Mxx.Bgg.Encoding
    (ciphertext : ExactMatrix q n 1 gadgetColumns)
    (maskSecret payloadSecret : ExactMatrix q n 1 secretColumns)
    (publicMatrix gadget : ExactMatrix q n secretColumns gadgetColumns)
    (message : ExactPoly q n) where
  error : ErrorMatrix n 1 gadgetColumns
  equation :
    ciphertext =
      maskSecret * publicMatrix -
      message • (payloadSecret * gadget) +
      reduceMatrix error

structure Mxx.Bgg.GswCiphertext
    (ciphertext : ExactMatrix q n secretColumns gadgetColumns)
    (payloadSecret outputSecret : ExactMatrix q n 1 secretColumns)
    (gadget : ExactMatrix q n secretColumns gadgetColumns)
    (message : ExactPoly q n) where
  error : ErrorMatrix n 1 gadgetColumns
  equation :
    payloadSecret * ciphertext =
      message • (outputSecret * gadget) +
      reduceMatrix error
```

Error bounds are predicates over the structure's `error` field rather than fields duplicated in
the exact structure.

`MxxBgg/Multiplication.lean` proves BGG+ multiplication from ordinary matrix algebra, a gadget
preimage equation, and the two input `Encoding` equations. `MxxBgg/Boolean.lean` proves the BGG+
Boolean gate lemmas used by Diamond.

Application proofs instantiate these structures with values obtained from the generated trace.
They never use independent ciphertext variables disconnected from an IR wire.

## 10. Proof ownership by operation scope

The following ownership is mandatory:

| Fact | Owning Lean package |
|---|---|
| Concrete IR syntax, typed refs, graph evaluation | `MxxIrCore` |
| Integer/Boolean/family/selector/loop semantics | `MxxIrCore` |
| Concrete matrix backend and linked artifact evaluation | `MxxRuntime` |
| Negacyclic ring and reduction homomorphism | `MxxPrimitives` |
| Matrix add/subtract/multiply/scale exact laws | `MxxPrimitives` |
| Coefficient and matrix norm inequalities | `MxxPrimitives` |
| Gaussian, interval, hash, and preimage sampler facts | `MxxPrimitives` |
| `ApplyPreimage` exact and bound lemma | `MxxPrimitives` |
| Gadget matrix and exact decomposition | `MxxGadgets` |
| Input-injector exact and bound theorem | `MxxGadgets` |
| Boolean circuit functional evaluation | `MxxGadgets` |
| BGG+ encoding and multiplication | `MxxBgg` |
| BGG+ Boolean gates and public-key transitions | `MxxBgg` |
| Diamond linked-stage invariants | `MxxWe` |
| Diamond final noisy-plaintext equation and bound | `MxxWe` |
| Diamond decoder implication | `MxxWe` |

An application theorem may use lower-level facts, but it must not copy their proofs into
`crates/we/lean`.

## 11. Generated node equations

The emitter generates a typed equation theorem for every reached node output. For example:

```lean
theorem node_184_port_0_eq
    (oracle) (env) (trace) (hEval :
      Mxx.IR.eval (RuntimeBackend oracle) program env = .ok trace) :
    trace.get node184Port0 =
      trace.get node181Port0 - trace.get node183Port0 := by
  simpa [Mxx.IR.eval] using successful_node_equation hEval
```

For structural nodes, the equation uses the generic subgraph, family, grid, or sequential-loop
evaluator rather than an expanded copy for every index. These equations are derived from the
generated program definition and evaluator. They are not assumptions; the only premise is that
the exact generated evaluation produced the stated trace.

Application proofs use these equations to establish `Mxx.Bgg.Encoding`, preimage, digit, and
Diamond invariants for selected generated wires. No generic engine decides which invariant a wire
should satisfy.

## 12. Diamond theorem statement

The semantic theorem is parameterized only by the generated IR `Candidate` and
`ParametersMatchProgram`.  A backend CRT layout is a separate refinement layer:
`RuntimeDcrtRepresentation` records the ordered moduli, base-bit metadata, and
derived modulus/depth facts.  `CertifiedCandidate` combines both layers only
when an explicit representation-matching proof is available.

### 12.1 Generated application parameters

Each generated candidate namespace contains the exact candidate values. The checked-in fixture
uses `Mxx.We.Golden.DiamondWE`; a cache candidate uses a namespace derived from its semantic hash.
Both instantiate the same reusable type:

```lean
structure Mxx.We.DiamondWE.ParametersData where
  modulus : Nat
  ringDimension : Nat
  gadgetBase : Nat
  gadgetDigitCount : Nat
  witnessWidth : Nat
  errorCutoff : Nat
  preimageCutoff : Nat

def Mxx.We.DiamondWE.ParametersData.Valid (data : ParametersData) : Prop :=
  -- positivity, matrix-shape, radix, and DecoderGeometryValid conditions
  ...

structure Mxx.We.DiamondWE.Parameters where
  data : ParametersData
  valid : data.Valid

def Mxx.We.Golden.DiamondWE.parameters : Parameters := ...
```

Convenience projections expose `parameters.modulus`, `ringDimension`, and the remaining fields from
`parameters.data`; theorem statements never accept unvalidated `ParametersData`.

It also contains the exact generated Diamond bound expression and its evaluated value. Candidate
parameters may therefore occur definitionally in the claim, as permitted by this specification.

The reusable theorems in `Exact.lean`, `Noise.lean`, `Decoder.lean`, and `Correctness.lean` quantify
over arbitrary natural-number `ParametersData` through the validated `Parameters` wrapper; they
are not duplicated
for each modulus, ring dimension, or CRT depth. The generated candidate supplies concrete values
and discharges decidable shape and arithmetic premises in `Check.lean`. A lemma may be specialized
to those numerals only when generalization materially prevents Lean from elaborating or reducing
it; such specialization stays in the generated cache/golden check and does not fork the protocol
proof.

For a Lean-verified candidate, `DiamondSelectedParameters.noise_bound` is exactly the `BigUint`
evaluation of this same Diamond-owned `BoundExpr`. The emitter serializes that expression and value
into `Parameters.lean`; it does not recompute or transcribe a second formula. Consequently the
runtime integration test's comparison against `selected.noise_bound` uses the identical numeric
bound proved by `CorrectnessClaim`. A value accepted by the migration-only generic noise simulator
must not be stored in this field.

### 12.2 Exact ideal plaintext

For message `m`, the ideal plaintext is a `1 × 1` matrix whose sole entry is the constant
polynomial

```text
coefficient 0: floor(q / 2) * Bool.toNat m
all other coefficients: 0.
```

This deliberately uses `floor(q / 2)`. The encryption graph may use rounded-up half modulus in the
sample target so that subtraction leaves this canonical center for odd `q`.

### 12.3 Decoder-safe threshold

Let:

```text
quarter = RoundDiv(q - 2, 4)
half    = floor(q / 2)
```

The current parameter search accepts a coefficient-error radius `B` exactly when:

```text
quarter > B
q - (3 * quarter + B) > 0
half >= quarter + B
3 * quarter >= half + B.
```

Lean defines the equivalent strict threshold:

```lean
def Mxx.We.DiamondWE.decoderNoiseThreshold (q : Nat) : Nat :=
  min quarter
    (min (q - 3 * quarter)
      (min (half - quarter + 1)
        (3 * quarter - half + 1)))
```

The truncated subtraction conditions are explicit data:

```lean
structure Mxx.We.DiamondWE.DecoderGeometryValid (q : Nat) : Prop where
  quarter_le_half : quarter q ≤ q / 2
  half_le_three_quarters : q / 2 ≤ 3 * quarter q
  three_quarters_le_q : 3 * quarter q ≤ q

def Mxx.We.DiamondWE.DecoderSafe (q noise : Nat) : Prop :=
  quarter q > noise ∧
  q - (3 * quarter q + noise) > 0 ∧
  q / 2 ≥ quarter q + noise ∧
  3 * quarter q ≥ q / 2 + noise
```

`ParametersData.Valid` contains `DecoderGeometryValid modulus`. `Decoder.lean` must prove

```lean
decoder_safe_iff :
  DecoderGeometryValid q →
  (noise < decoderNoiseThreshold q ↔ DecoderSafe q noise)

decoder_threshold_le_half :
  DecoderGeometryValid q → decoderNoiseThreshold q ≤ q / 2
```

The second lemma supplies the `bound < q / 2` fact needed when converting an integer error witness
to centered modular distance. The implementation must not silently replace the four exact
conditions with an approximate `Noise < q / 4` test.

### 12.4 Whole-polynomial distance

The integration test checks every coefficient, while the decoder reads only coefficient zero. The
Lean theorem preserves the stronger property:

```lean
def Mxx.Primitives.centeredMatrixDistance
    (q n rows columns : Nat)
    (x y : ExactMatrix q n rows columns) : Nat :=
  Finset.univ.sup fun row =>
    Finset.univ.sup fun column =>
      Finset.univ.sup fun coefficient =>
        ZMod.valMinAbs
          ((x row column).coeff coefficient -
           (y row column).coeff coefficient) |>.natAbs
```

If direct use of `ZMod.valMinAbs` does not fit the concrete ring wrapper, the primitives package
provides an equivalent proved definition. Application code must not define a second distance.
For the Diamond noisy-plaintext wire, `rows = columns = 1`; retaining the matrix dimensions here
matches the runtime diagnostic, which checks every coefficient of every matrix entry.

### 12.5 Authoritative generated claim

The reusable application layer prescribes the typed references that generation must fill:

```lean
structure Mxx.We.DiamondWE.BooleanCircuitInputRefs
    (program : Mxx.IR.LinkedProgram) where
  activeGateCountsInput : TypedWireRef program ...
  circuitGateKindsInput : TypedWireRef program ...
  circuitLeftSourcesInput : TypedWireRef program ...
  circuitRightSourcesInput : TypedWireRef program ...
  circuitOutputSourceInput : TypedWireRef program ...

structure Mxx.We.DiamondWE.Refs (program : Mxx.IR.LinkedProgram) where
  messageInput : TypedWireRef program .bool
  encryptionInstanceBitsInput : TypedWireRef program (.family instanceShape .int)
  decryptionInstanceBitsInput : TypedWireRef program (.family instanceShape .int)
  witnessBitsInput : TypedWireRef program (.family witnessShape .int)
  encryptionCircuit : BooleanCircuitInputRefs program
  decryptionCircuit : BooleanCircuitInputRefs program
  noisyPlaintextOutput : TypedWireRef program (.matrix plaintextMatrixType)
  decodedOutput : TypedWireRef program .bool

structure Mxx.We.DiamondWE.BoundData where
  expression : BoundExpr DiamondBoundParameter
  environment : DiamondBoundParameter → Nat
  value : Nat
  evaluated : expression.eval environment = value

structure Mxx.We.DiamondWE.Candidate where
  program : Mxx.IR.LinkedProgram
  parameters : Parameters
  refs : Refs program
  bound : BoundData
```

Every field is rendered from the linked validated IR, concrete bindings, named inputs/outputs, and
the single Rust `BoundExpr`. There is no application callback that can substitute a different
wire. Generated `Claim.lean` only constructs this data and aliases the reusable proposition:

```lean
def Mxx.We.DiamondWE.CorrectnessClaim (candidate : Candidate) : Prop :=
  ∀ (oracle : Mxx.Runtime.RuntimeGadgetOracle) (message : Bool)
    (env : Mxx.IR.EvalEnv (RuntimeBackend oracle) candidate.program.data)
    (run : GoodRunPromise oracle candidate env),
    ValidExternalInputs oracle candidate env message →
    BooleanCircuitEvaluatesToOne oracle candidate env →
    ∃ (noisy : RuntimeMatrixValue candidate.plaintextMatrixType)
      (decoded : Bool)
      (exactNoisy : ExactMatrix candidate.parameters.modulus
        candidate.parameters.ringDimension 1 1),
      traceTypedValueAt run.trace #[] candidate.refs.noisyPlaintextOutput = some noisy ∧
      traceTypedValueAt run.trace #[] candidate.refs.decodedOutput = some decoded ∧
      HEq noisy exactNoisy ∧
      Nonempty (ApproxWithin exactNoisy
        (idealPlaintext candidate.parameters message) candidate.bound.value) ∧
        candidate.bound.value < decoderNoiseThreshold candidate.parameters.modulus ∧
          decoded = message
```

This is the current provisional public boundary. Typed trace references tie both observations to
the generated program; `HEq` bridges the runtime descriptor-indexed matrix to the parameter-indexed
exact matrix; and `ApproxWithin` carries the final error witness and its bound. The repeated
explicit norm proposition is unnecessary because the same inequality is already a field of
`ApproxWithin`.

These premises are transparent prescribed definitions, not application-selected propositions:

- `ValidExternalInputs oracle candidate env message` reads only generated `ExternalInputRef`
  values. It
  states that the generated Boolean message input equals `message`; the encryption and decryption
  circuit-input values are equal; the two instance arrays are equal; all generated
  instance/witness integer inputs are zero or one; and the
  circuit reconstructed from the generated active-gate counts, gate kinds, sources, and output
  source satisfies `BooleanCircuit.Valid`. Active-gate counts are authoritative per-layer inputs;
  they must not be inferred from padded opcode values, because a constant-false gate and an unused
  padded slot can have the same opcode. The predicate also checks the DSL's canonical zero encoding
  for every inactive padded gate/source slot;
- `BooleanCircuitEvaluatesToOne oracle candidate env` likewise reads those exact external inputs,
  constructs the `BooleanCircuit`, evaluates it on the generated instance and witness values, and
  requires the result to equal `true`; and
- `GoodRunPromise oracle candidate env` contains one actual successful trace and exhaustive
  occurrence-specific `GoodSamples` for that same trace. Sampler kind, cutoff, and source/target
  arguments are derived from each stored sampler payload rather than from caller-supplied
  descriptors.

Thus neither a handwritten `False` premise nor a separately restated circuit can make the theorem
vacuous. The circuit may be supplied as ordinary DSL/IR inputs, but the theorem reads precisely
those external input values. A program that always returns `EvalError` has no `GoodRunPromise`, so
this theorem makes no independent liveness claim about it.

The intended application-specific proof over a generated structural view has this target shape:

```lean
theorem Mxx.We.DiamondWE.correct
    (candidate : Candidate)
    (view : Candidate.HasDiamondGraphShape candidate)
    (valid : CandidateValidity candidate view) :
    CorrectnessClaim candidate := by
  -- application-specific proof
```

This `correct` theorem is not yet implemented. The current crate declares `CorrectnessClaim` and
proves only projection/consequence lemmas from an assumed claim. The future proof must use the
exact trace contained in `GoodRunPromise` and apply the exact/noise/decoder lemmas. Proving that
every valid external environment produces such a run is a separate liveness and total-environment
obligation, intentionally outside the first theorem.

`HasDiamondGraphShape candidate : Type` is a Type-valued structural view containing typed
references, recurrence data, and equations derived from `candidate.program`; it does not assert a
second protocol semantics. Its decidable constructor checks the concrete node kinds, arguments,
scopes, and requires `candidate.refs` to equal the named references derived from the program. It is
Type-valued so `deriveOutputNoiseBound` may compute from its finite recurrence data; its equation
fields remain propositions checked by the kernel.

`Noise.lean` defines `deriveOutputNoiseBound parameters view` and
`deriveBoundEnvironment parameters view` from the Diamond recurrence. `CandidateValidity` contains
kernel-checked proofs of all non-topological candidate data:

```lean
structure Mxx.We.DiamondWE.CandidateValidity
    (candidate : Candidate)
    (view : Candidate.HasDiamondGraphShape candidate) : Prop where
  parameters_match_program :
    ParametersMatchProgram candidate.parameters candidate.program view
  bound_expression_eq :
    candidate.bound.expression = deriveOutputNoiseBound candidate.parameters view
  bound_environment_eq :
    ∀ parameter,
      candidate.bound.environment parameter =
        deriveBoundEnvironment candidate.parameters view parameter
  decoder_safe :
    candidate.bound.value < decoderNoiseThreshold candidate.parameters.modulus
```

`ParametersMatchProgram` equates every semantic IR field—modulus, ring dimension, gadget
base/digit count, witness width, matrix shapes, sampler cutoffs, and decoder literals—to the
concrete bindings, types, shapes, and node literals exposed by the generated program/view. It is
not merely a positivity predicate. `Mxx.IR.deriveSamplerDescriptors` scans the validated program's
sampler node kinds and typed arguments, so it cannot be empty when the program contains a reached
sampler.

The Diamond noise proof establishes distance at most the derived expression; `bound_expression_eq`,
`bound_environment_eq`, and `candidate.bound.evaluated` connect it to the exact emitted/Rust value.
`decoder_safe` is the concrete arithmetic obligation discharged in generated `Check.lean`. Thus the
same program cannot be packaged with bound zero or an unsafe bound and passed to `correct`.

Generated `Check.lean` constructs the structural view and `CandidateValidity` by kernel-checked
reduction and applies `correct`. A topology change unsupported by the proof template, a modified
parameter/program mismatch, a modified bound recurrence/environment, or a failed decoder
inequality therefore fails generation or proof checking, while the claim remains directly about
the emitted IR evaluator. The sole identity is `candidate.program.identity`; `Candidate` carries
no duplicate caller-provided identity or sampler inventory.

The proof may strengthen the generated claim by constructing an integer error witness:

```lean
∃ error,
  noisy = ideal + reduceMatrix error ∧
  matrixNorm error ≤ bound
```

The whole-polynomial centered-distance result follows from this witness and `bound < q / 2`.

### 12.6 Initial integration-test instance

`crates/we/test-data/diamond_lean_golden.json` is the deterministic source for checked-in golden
generation. It contains no expressions or environment-variable defaults; it records concrete
decimal values for every `Parameters` field, the full serialized `BoundExpr`, the AND-circuit data,
instance width/value, witness width/value, and the expected linked semantic hash. It is populated
from one parameter-search result only after that result is Lean-verified, then reviewed and
committed. Regeneration is exactly:

```text
cargo run -p mxx-we --example emit_diamond_lean_correctness -- \
  --fixture crates/we/test-data/diamond_lean_golden.json --write
```

CI runs the same command with `--check`. It never runs live parameter search or reads `MXX_*`
environment overrides to determine golden contents.

The fixture uses the same logical case as `crates/we/tests/test_gpu_diamond_we.rs`:

- one AND gate with inputs zero and one;
- one true instance bit;
- all witness bits true;
- circuit output true;
- message true; and
- the concrete parameters recorded in the committed Lean-verified fixture.

`CorrectnessClaim` must quantify over both Boolean messages: neither `ValidExternalInputs` nor any
other premise may restrict the quantified message to `true`. The current integration-test case is
obtained by instantiating this reusable theorem with message `true`; it is not a weaker, separately
proved single-message theorem.

The proof-linked GPU regression loads the committed parameter fixture rather than relying on a
fresh search to rediscover it. A separate parameter-search test may search and Lean-check new
candidates. Both runtime paths compare against the `noise_bound` obtained from the exact emitted
Diamond `BoundExpr`.

GPU availability, GPU timings, repetition count, and cost-estimator assertions are not part of the
Lean correctness claim.

## 13. Diamond proof decomposition

The application proof is divided as follows.

### 13.1 `Model.lean`

- defines the transparent `Candidate`, `Refs`, and public-input predicates consumed by every
  generated candidate;
- defines the ideal plaintext;
- defines input-only `ValidExternalInputs`, `BooleanCircuitEvaluatesToOne`, and
  occurrence-exhaustive `GoodSamples`;
- defines `GoodRunPromise`, which packages one successful trace with its `GoodSamples`; and
- exposes no alternative protocol evaluator.

### 13.2 `Exact.lean`

- consumes the exact successful trace packaged by `GoodRunPromise`;
- proves input-injector encodings from generated node equations;
- proves BGG+ encoding invariants for initial values and each Boolean layer;
- proves digit reconstruction where a matrix or gadget is decomposed;
- proves exact preimage-consumption equations locally;
- proves the accepting circuit output has the required BGG+ payload; and
- proves the final noisy plaintext equals the ideal plaintext plus a reduced integer error.

Only exact terms needed by those local module equations are expanded. Unrelated public exact terms
may be bundled into locally defined public matrices. The proof does not construct a graph-wide
normal form or prove every possible public-term cancellation.

### 13.3 `Noise.lean`

- defines the Diamond application-specific bound tree;
- applies primitive, input-injector, BGG+, and preimage norm lemmas;
- proves the final error witness is within the generated bound; and
- proves the Rust-isomorphic bound expression evaluates to that bound.

### 13.4 `Decoder.lean`

- proves equivalence between `decoderNoiseThreshold` and the four Rust interval conditions;
- proves that the whole-polynomial bound implies the required coefficient-zero bound; and
- proves the generated `ThresholdDecode` result equals the message.

### 13.5 `Correctness.lean`

- currently declares the provisional run-conditional `CorrectnessClaim` and proves consequence
  lemmas from an assumed claim;
- targets a future proof combining `Exact`, `Noise`, and `Decoder` to establish that claim from
  `HasDiamondGraphShape candidate` and `CandidateValidity candidate view`;
- does not claim independent evaluator liveness or totality for arbitrary external environments;
  and
- contains no low-level matrix expansion that belongs in a lower package.

## 14. Rust generation API

Generic IR emission belongs to `mxx-ir-core`:

```rust
pub fn render_lean_program(
    program: &ValidatedLinkedProgram,
) -> Result<RenderedLeanProgram, LeanEmissionError>;
```

This renderer accepts only the already linked and validated program. It owns canonical graph,
concrete binding, type, literal, scope, node, and typed-reference emission. Program constructors,
output references, and artifact links are emitted from that object in one deterministic traversal;
the caller cannot provide a second graph, semantic stage name, artifact-link list, or raw producer
node number. A matching hash beside a handwritten Lean program is not sufficient. The renderer
uses the validated runtime links only to construct the typed semantic projection, then computes and
renders the same `LeanSemanticIdentity`; the execution nonce and artifact content provenance are
not part of the emitted semantic data. The renderer knows no Diamond theorem.

Diamond claim generation belongs to `mxx-we`:

```rust
pub struct DiamondLeanClaimRequest<'a> {
    pub program: &'a RenderedLeanProgram,
    pub parameters: &'a DiamondSelectedParameters,
    pub inputs: DiamondNamedInputs<'a>,
    pub outputs: DiamondNamedOutputs<'a>,
}

pub struct DiamondNamedOutputs<'a> {
    pub noisy_plaintext: LeanNamedOutputSelector<'a>,
    pub decoded: LeanNamedOutputSelector<'a>,
}

pub fn emit_diamond_lean_correctness(
    request: &DiamondLeanClaimRequest<'_>,
    mode: EmitMode,
) -> Result<DiamondLeanArtifact, DiamondLeanError>;
```

`LeanNamedOutputSelector` and the corresponding input selector are owned by `mxx-ir-core`; they
contain a stage name and a declared input/output name, never a noise-simulator root or raw node
number. The application emitter resolves every circuit/message/instance/witness input and both
outputs to exact generated `TypedWireRef` values before rendering. Missing, ambiguous, wrongly
typed, or cross-program references are errors. `mxx-we` therefore has no generation-time type
dependency on `mxx-noise-simulator`.

The application emitter calls the Diamond-owned `derive_output_noise_bound(program, parameters)`;
the request cannot inject an arbitrary bound. It emits the derived expression, environment, and
exact `BigUint` evaluation as `BoundData`, and rejects a nonmatching
`parameters.noise_bound`. Lean's `CandidateValidity` independently checks the same derivation.

Generation is exposed by an explicit example or binary owned by `mxx-we`; Cargo `build.rs` must not
modify checked-in source files.

## 15. Parameter-search correctness check

Parameter search may use `ParametersMatchProgram` to validate the generated
semantic graph.  It must not claim CRT implementation refinement merely from
that result.  CRT modulus ordering, product, coprimality, and actual bit widths
are checked separately by `RuntimeDcrtRepresentation`; backend-specific NTT
suitability remains a future refinement obligation.

### 15.1 Authoritative first implementation

Initially, a candidate is correctness-accepted only after:

1. constructing and validating its linked frozen graphs;
2. generating a unique cache-local Lake package containing `Program.lean`, `Parameters.lean`,
   `Claim.lean`, and `Check.lean`, keyed by `LeanSemanticIdentity.linked_program_sha256`;
3. checking that the application proof theorem typechecks against those generated definitions; and
4. checking the concrete numeric threshold side condition.

The checker invokes Lean as a subprocess. It does not embed an unstable Lean runtime library in
Rust.

```rust
pub enum DiamondCorrectnessVerdict {
    Rejected {
        bound: BigUint,
        decoder_threshold: BigUint,
    },
    LeanVerified {
        semantic_identity: LeanSemanticIdentity,
        claim_instance_sha256: [u8; 32],
        theorem: String,
        artifact_directory: PathBuf,
    },
}
```

Infrastructure errors are distinct from mathematical rejection. A timeout, emitter failure, stale
file, unsupported node, or Lean failure must never be returned as `Rejected` or `LeanVerified`.

### 15.2 Fast application-specific prefilter

Repeated Lean elaboration for every rejected CRT depth is unnecessary. The preferred search path
is:

1. evaluate the Diamond-owned Rust `BoundExpr` using exact `BigUint` arithmetic;
2. reject candidates that fail the exact four decoder inequalities;
3. run lattice security estimation only according to the existing search policy;
4. generate and check Lean for the first candidate that passes the fast checks; and
5. return the candidate only after Lean succeeds.

The fast checker is a prefilter, not the final trust anchor. It is acceptable for it to reject a
candidate conservatively. It must not allow `DiamondParameterSearch::search` to return a candidate
that lacks a matching Lean-verified artifact.

The current `mxx-noise-simulator` may be used as a migration prefilter, but its accepted result is
not a substitute for the new Diamond theorem. Once the Diamond-owned bound implementation covers
the full graph, the generic simulator dependency may be removed from `mxx-we` in a separate change.
Regardless of whether that prefilter remains, the returned
`DiamondSelectedParameters.noise_bound` is populated only from the Diamond-owned `BoundExpr` that
was emitted and proved for the returned candidate.

### 15.3 Cache behavior

The correctness cache key includes:

```text
IR_VERSION
Lean source schema version
linked-program semantic hash
application claim-template hash
bound-expression hash
generated claim-instance hash
Lean toolchain version
mathlib revision
```

The cache is stored below Cargo `target/` and is never committed. Each cache package uses a
namespace containing a prefix of the semantic hash and imports `MxxWe` by a generated path
dependency. A cache hit is valid only when the recorded theorem name and all key components match;
the fixed checked-in golden namespace is never on the candidate package's module path.

The claim-instance hash is computed over the canonical rendered contents of `Parameters.lean` and
`Claim.lean`. It therefore commits to every resolved typed input/output reference, sampler
descriptor, concrete bound-parameter value, evaluated bound, and candidate namespace. On a cache
hit the checker regenerates these two files in memory, recomputes the hash, and compares it before
returning `LeanVerified`; a linked-program hash alone is insufficient.

## 16. Generation and proof gates

The implementation is complete only when all of the following pass:

1. Rust unit tests for lowering and rendering every reached Diamond `NodeKind` into
   `ConcreteNodePayload`;
2. Rust golden tests for the linked semantic identity and typed input/output references;
3. `--check` regeneration of the checked-in Diamond generated files;
4. `lake build` from `crates/we/lean`;
5. zero `sorry`, `admit`, or custom axioms in the transitive proof packages;
6. an axiom report containing only standard Lean/mathlib axioms accepted by repository policy;
7. a negative test showing that changing a reached IR operation changes generated output and makes
   the stale check fail;
8. a negative test showing that changing a candidate modulus changes the linked semantic hash and
   the generated parameter theorem;
9. a negative test showing that an insufficient decoder threshold is rejected; and
10. the existing GPU integration test, when explicitly run, still checks decoded-message equality
    and observed whole-polynomial noise against the selected bound.
11. the Diamond inventory test writes the generated source to an isolated temporary Lake module,
    imports the crate-local `MxxIrCore` package, and invokes Lean to elaborate it successfully.
    The temporary module is not checked in and is removed after the test.

Lean proof checking does not replace the runtime integration test. The two tests establish
different boundaries: generated-IR mathematics and concrete runtime behavior.

## 17. Implementation phases

### Phase 1: Lean foundations

- create the six crate-local Lean packages;
- implement concrete negacyclic rings, reduction, matrices, norms, and bounds;
- implement concrete IR types and straight-line evaluation; and
- implement closed structural expressions, slot declarations, and exhaustive node payload lowering;
- prove primitive matrix and preimage lemmas.

### Phase 2: structural IR and generation

- implement families, selectors, subgraphs, `ParallelGrid`, and `SequentialLoop`;
- implement the generic Rust graph emitter;
- run the generated-source Lean elaboration gate from the Diamond inventory test;
- emit typed node equations and output references; and
- add stale-generation tests.

### Phase 3: reusable crypto layers

- prove gadget decomposition and input injection in `MxxGadgets`;
- prove BGG+ encoding, multiplication, and Boolean gates in `MxxBgg`; and
- ensure each theorem is expressed over ordinary IR semantic values.

### Phase 4: Diamond

- generate the linked Diamond program and claim;
- prove the exact final plaintext equation;
- prove the application-specific noise bound;
- prove the exact decoder implication; and
- check the current integration-test instance.

### Phase 5: parameter search

- add the Diamond-owned exact `BoundExpr` evaluator;
- use it to prefilter candidates;
- invoke Lean for a passing candidate;
- cache the verified artifact by linked semantic hash; and
- require the verified artifact before returning selected parameters.

## 18. Target acceptance criteria

These are target completion criteria, not a statement of the current implementation status. The
`CorrectnessClaim` boundary exists, but the application theorem `Mxx.We.DiamondWE.correct` is still
absent. Completion requires a new checkout to perform the following without handwritten protocol
duplication:

1. build the Diamond encryption and decryption graphs from the DSL;
2. select a concrete parameter candidate;
3. emit the concrete linked Lean program directly from those frozen graphs;
4. regenerate the application claim that references the emitted noisy and decoded outputs;
5. typecheck the crate-local reusable proofs and Diamond proof with no `sorry`;
6. obtain the run-conditional `Mxx.We.DiamondWE.correct` theorem for the exact trace and exhaustive
   sampler facts packaged by `GoodRunPromise`;
7. obtain a theorem proving whole-polynomial distance from
   `floor(q / 2) * message` is within the application bound for an accepting witness;
8. obtain the strict decoder-threshold inequality and decoded-message equality;
9. reject any stale generated artifact after a reached IR or parameter change; and
10. return the parameter candidate only with its matching Lean verification artifact.

An independent theorem constructing a successful run for every valid external environment is a
separate future liveness/total-environment goal and is not required by the first run-conditional
correctness theorem.

The implementation must not require adding a separate DSL contract for every BGG+, gadget, or
Diamond module. Those semantic structures exist only in the proof libraries and are established
from the generated IR node equations.
