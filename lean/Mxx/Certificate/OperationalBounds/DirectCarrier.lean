import Mxx.Certificate.OperationalBounds.Core

namespace Mxx.Certificate

open Mxx.Ir

/-! ## Selection-preserving operational expressions

The executable checker still evaluates ordinary operations into the flat facts above.  This
request-local arena is the compact boundary for unresolved dynamic selections.  Arena indices are
allocation identities only: they are never matrix, relation, or symbolic-equality evidence. -/

/-- Lossless descriptor for one delayed matrix operation.  It records every input needed to invoke
the existing concrete transfer later; it never carries a replacement noise formula. -/
inductive PrimitiveOperationKind where
  | add (subtract : Bool)
  | multiply (rule : DerivationRule) (rightWire : WireRef)
  | tensor
  | concat (axis : ConcatAxis)
  | transform (operation : OperationalFactorTransform)
  | slice (rows columns : Option (IntExpr × IntExpr))
  | scale (scalar : IndexedParameterExpr)
      (loopDomains : List IndexedOperationalParameterDomain)
  | bggGrouping
  deriving BEq

structure PrimitiveOperation where
  kind : PrimitiveOperationKind
  outputType : IndexedMatrixTypeExpr
  /-- The Graph-IR shape checked when constructing a delayed carrier.  The owner-aware descriptor
  above is evaluated again at each fixed assignment before reduction; this schema is never a
  substitute for that evaluation. -/
  outputSchema : MatrixTypeExpr
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  parameterEnvironment : ParamEnvironment
  deriving BEq

/-! ## Direct indexed-value carrier

This replacement foundation has one authoritative payload.  Fixed references point only into a
selection-free arena; mapped and delayed pointwise payloads never carry a forgeable output ref. -/

inductive OperationalFixedScalarSchema where
  | integer | boolean | real
  | trapdoor
      (matrixType : MatrixTypeExpr)
      (sigma : RealExpr)
      (gadgetBase digitCount preimageMaxCoefficientBound : IntExpr)
  | bytes (length : Int)
  | typedBlob (typeName : String) (schemaHash : List Nat)
  | unknown (wireType : WireTypeExpr)
  deriving BEq

inductive OperationalIndexedPayloadSchema where
  | matrix (matrixType : MatrixTypeExpr)
  | scalar (schema : OperationalFixedScalarSchema)
  deriving BEq

inductive FixedOperationalPayloadRef where
  | matrix (id : Nat)
  | scalar (id : Nat)
  deriving BEq, DecidableEq, Repr

structure FixedOperationalPayloadArena where
  matrices : Array OperationalMatrixFact := #[]
  scalars : Array OperationalScalarFact := #[]
  deriving BEq

def operationalScalarSchema : OperationalScalarFact → OperationalFixedScalarSchema
  | .integer _ => .integer
  | .boolean => .boolean
  | .real => .real
  | .trapdoor fact => .trapdoor fact.matrixType fact.sigma fact.gadgetBase fact.digitCount
      fact.preimageMaxCoefficientBound
  | .bytes fact => .bytes fact.length
  | .typedBlob typeName schemaHash => .typedBlob typeName schemaHash
  | .unknown wireType => .unknown wireType

/-- Construct the complete scalar schema declared by one executable Graph-IR output.  Integer,
boolean, real, byte, blob, and trapdoor selection all compare this exact contract against every
fixed lane before the direct table is formed. -/
def operationalScalarWireSchema
    (environment : ParamEnvironment) : WireTypeExpr → Option OperationalFixedScalarSchema
  | .constantInt | .integer => some .integer
  | .constantBool | .boolean => some .boolean
  | .constantReal | .real => some .real
  | .bytes length => (.bytes ·) <$> length.evaluate environment
  | .typedBlob typeName schemaHash => some (.typedBlob typeName schemaHash)
  | .trapdoor matrixType sigma gadgetBase digitCount preimageMaxCoefficientBound =>
      some (.trapdoor matrixType sigma gadgetBase digitCount preimageMaxCoefficientBound)
  | _ => none

def FixedOperationalPayloadArena.refHasSchema
    (arena : FixedOperationalPayloadArena)
    (schema : OperationalIndexedPayloadSchema)
    (reference : FixedOperationalPayloadRef) : Bool :=
  match schema, reference with
  | .matrix expected, .matrix id => arena.matrices[id]?.any (fun fact => fact.matrixType == expected)
  | .scalar expected, .scalar id => arena.scalars[id]?.any (fun fact => operationalScalarSchema fact == expected)
  | _, _ => false

def FixedOperationalPayloadArena.pushMatrix
    (arena : FixedOperationalPayloadArena)
    (fact : OperationalMatrixFact) : FixedOperationalPayloadArena × FixedOperationalPayloadRef :=
  ({ arena with matrices := arena.matrices.push fact }, .matrix arena.matrices.size)

def FixedOperationalPayloadArena.pushScalar
    (arena : FixedOperationalPayloadArena)
    (fact : OperationalScalarFact) : FixedOperationalPayloadArena × FixedOperationalPayloadRef :=
  ({ arena with scalars := arena.scalars.push fact }, .scalar arena.scalars.size)

inductive DirectRelationOperationKind where
  | preimage (maximum : IndexedParameterExpr) (loopDomains : List IndexedOperationalParameterDomain)
  | decomposition (declaredType : IndexedMatrixTypeExpr) (base : IndexedParameterExpr) (small : Bool)
      (digitCount : IndexedParameterExpr) (loopDomains : List IndexedOperationalParameterDomain)
      (layouts : List Mxx.GadgetLayoutDescriptor)
  deriving BEq

structure DirectRelationOperation where
  kind : DirectRelationOperationKind
  outputType : IndexedMatrixTypeExpr
  outputSchema : MatrixTypeExpr
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  parameterEnvironment : ParamEnvironment
  deriving BEq

/-- Scalar primitive metadata belongs to the executable producer just like matrix primitives.
The direct scalar evaluator must not manufacture a temporary subject when a loop body produces
an integer used by a later indexed lookup. -/
structure DirectScalarOperation where
  kind : OperationalScalarPrimitiveKind
  ownerScope : Option ScopeTemplateKey
  ownerNode : Nat
  outputPort : Nat
  deriving BEq

inductive OperationalIndexedPointwiseOperation where
  | matrix (operation : PrimitiveOperation)
  /-- Relation-producing matrix kernels retain every graph operand as a direct carrier input.
  This is distinct from ordinary matrix-only pointwise operations because preimage sampling also
  consumes a trapdoor scalar at the same indexed assignment. -/
  | relation (operation : DirectRelationOperation)
  | scalar (operation : DirectScalarOperation)
  | matrixToScalar (operation : DirectValueScalarOperation)
  | matrixFromScalar (operation : DirectValueMatrixOperation)
  deriving BEq

inductive OperationalIndexedPayload where
  | shared
      (schema : OperationalIndexedPayloadSchema)
      (reference : FixedOperationalPayloadRef)
  | explicit
      (schema : OperationalIndexedPayloadSchema)
      (binder : IndexVariable)
      (references : Array FixedOperationalPayloadRef)
  /-- An ordered direct-value table is the authoritative representation when a packed family
  contains delayed or already-mapped values.  It remains `explicitTable` storage: the IDs are
  the exact lane payloads, not an alternate selection semantics. -/
  | explicitValues
      (schema : OperationalIndexedPayloadSchema)
      (binder : IndexVariable)
      (values : Array OperationalIndexedValueId)
  | mapped
      (schema : OperationalIndexedPayloadSchema)
      (source : OperationalIndexedValueId)
      (map : IndexMap)
  /-- A wire-bound view over one direct carrier root.  Rebinding is intentionally lazy: the
  subject overlay is applied only after all pending maps have reached a fixed leaf, where the
  relation/provenance invariant is checked before replacement. -/
  | rebound
      (schema : OperationalIndexedPayloadSchema)
      (source : OperationalIndexedValueId)
      (subject : WireRef)
  /-- A root-result annotation preserves the source carrier and replaces only the complete
  fixed-assignment result's total hard bound.  Sequential recurrences use this after the body
  has already consumed its internal relations, so no leaf relation is retained or rewritten. -/
  | matrixResultBound
      (schema : OperationalIndexedPayloadSchema)
      (source : OperationalIndexedValueId)
      (totalHardBound : OperationalBoundExpr)
  | pointwise
      (schema : OperationalIndexedPayloadSchema)
      (operation : OperationalIndexedPointwiseOperation)
      (inputs : Array OperationalIndexedValueId)
  deriving BEq

def OperationalIndexedPayload.schema :
    OperationalIndexedPayload → OperationalIndexedPayloadSchema
  | .shared schema _ | .explicit schema _ _ | .explicitValues schema _ _ | .mapped schema _ _ |
      .rebound schema _ _ | .matrixResultBound schema _ _ | .pointwise schema _ _ => schema

def OperationalIndexedPayload.storage : OperationalIndexedPayload → IndexedStorage
  | .shared .. => .sharedTemplate
  | .explicit .. | .explicitValues .. => .explicitTable
  | .mapped .. | .rebound .. | .matrixResultBound .. | .pointwise .. => .mappedTemplate

abbrev OperationalIndexedValue := IndexedFact OperationalIndexedPayload

structure DirectOperationalIndexedArena where
  fixed : FixedOperationalPayloadArena := {}
  values : Array OperationalIndexedValue := #[]
  /-- The one executable integer producer for each dependent gather owner.  This is a registry
  rather than a cache: owner identity is part of the indexed semantics, and an ambiguous owner
  is rejected before any matrix lane can consume it. -/
  gatherIntegerRoots : Std.HashMap GatherLookupOwner OperationalIndexedValueId := {}
  deriving BEq

def DirectOperationalIndexedArena.valueAt?
    (arena : DirectOperationalIndexedArena)
    (id : OperationalIndexedValueId) : Option OperationalIndexedValue := arena.values[id]?

/-- Register one executable integer-family root for a gather owner.  Repeating the exact same
registration is idempotent because multiple dynamic family gets may consume one executable index
wire.  A different root is rejected: choosing either would erase owner provenance. -/
def DirectOperationalIndexedArena.registerGatherIntegerRoot
    (arena : DirectOperationalIndexedArena)
    (owner : GatherLookupOwner)
    (root : OperationalIndexedValueId) : Option DirectOperationalIndexedArena := do
  let value ← arena.valueAt? root
  match value.payload.schema, value.context.binders.toList with
  | .scalar .integer, [_] =>
      match arena.gatherIntegerRoots[owner]? with
      | none => some { arena with gatherIntegerRoots := arena.gatherIntegerRoots.insert owner root }
      | some existing => if existing == root then some arena else none
  | _, _ => none

/-- Resolve only an unambiguous owner registration.  The registry constructor above prevents
duplicates, but this still rejects a malformed arena assembled by a fixture or future caller. -/
def DirectOperationalIndexedArena.gatherIntegerRoot?
    (arena : DirectOperationalIndexedArena)
    (owner : GatherLookupOwner) : Option OperationalIndexedValueId :=
  arena.gatherIntegerRoots[owner]?

def explicitCountValid
    (environment : ParamEnvironment)
    (binder : IndexVariable)
    (references : Array FixedOperationalPayloadRef) : Bool :=
  match binder.count.evaluate environment with
  | some count => 0 < count && count == Int.ofNat references.size
  | none => false

def operationalMatrixTypeEqual (left right : MatrixTypeExpr) : Bool :=
  operationalSameRing left right && operationalDimensionEqual left.rows right.rows &&
    operationalDimensionEqual left.columns right.columns

/-- Closed type registry for delayed matrix operations.  Unsupported descriptors fail closed until
their fixed-assignment type rule is moved before this carrier. -/
def matrixOperationSchemasValid
    (operation : PrimitiveOperation)
    (inputs : Array OperationalIndexedPayloadSchema)
    (output : MatrixTypeExpr) : Bool :=
  if !operationalMatrixTypeEqual operation.outputSchema output then false else
  match operation.kind, inputs with
  | .add _, #[.matrix left, .matrix right] =>
      operationalMatrixTypeEqual left output && operationalMatrixTypeEqual right output
  | .multiply .., #[.matrix left, .matrix right] =>
      match inferOperationalProductMode left right with
      | .ok (_, inferred) => operationalMatrixTypeEqual inferred output
      | .error _ => false
  | .tensor, #[.matrix left, .matrix right] =>
      operationalSameRing left right && operationalSameRing left output &&
        operationalDimensionEqual output.rows (.multiply left.rows right.rows) &&
        operationalDimensionEqual output.columns (.multiply left.columns right.columns)
  | .concat axis, inputs =>
      match inputs.toList with
      | [] => false
      | .matrix first :: rest =>
          let matrices := first :: rest.filterMap fun schema => match schema with
            | .matrix matrixType => some matrixType
            | .scalar _ => none
          if matrices.length != inputs.size || !matrices.all (operationalSameRing first) then false else
          match axis with
          | .rows =>
              matrices.all (fun matrixType =>
                operationalDimensionEqual matrixType.columns first.columns) &&
                operationalSameRing first output &&
                operationalDimensionEqual output.columns first.columns &&
                operationalDimensionEqual output.rows
                  (matrices.foldl (fun rows matrixType => .add rows matrixType.rows) (.constant 0))
          | .columns =>
              matrices.all (fun matrixType => operationalDimensionEqual matrixType.rows first.rows) &&
                operationalSameRing first output &&
                operationalDimensionEqual output.rows first.rows &&
                operationalDimensionEqual output.columns
                  (matrices.foldl (fun columns matrixType => .add columns matrixType.columns) (.constant 0))
          | .diagonal =>
              operationalSameRing first output &&
                operationalDimensionEqual output.rows
                  (matrices.foldl (fun rows matrixType => .add rows matrixType.rows) (.constant 0)) &&
                operationalDimensionEqual output.columns
                  (matrices.foldl (fun columns matrixType => .add columns matrixType.columns) (.constant 0))
      | _ => false
  | .transform transform, #[.matrix input] =>
      match transform with
      | .negate => operationalMatrixTypeEqual input output
      | .transpose =>
          operationalSameRing input output &&
            operationalDimensionEqual output.rows input.columns &&
            operationalDimensionEqual output.columns input.rows
      | .rowSlice start stop =>
          operationalSameRing input output &&
            operationalDimensionEqual output.rows (.subtract stop start) &&
            operationalDimensionEqual output.columns input.columns
      | .columnSlice start stop =>
          operationalSameRing input output &&
            operationalDimensionEqual output.rows input.rows &&
            operationalDimensionEqual output.columns (.subtract stop start)
      | .rowEmbed _ _ | .columnEmbed _ _ => false
  | .slice rows columns, #[.matrix input] =>
      let expectedRows := match rows with
        | some (start, stop) => .subtract stop start
        | none => input.rows
      let expectedColumns := match columns with
        | some (start, stop) => .subtract stop start
        | none => input.columns
      operationalSameRing input output && operationalDimensionEqual output.rows expectedRows &&
        operationalDimensionEqual output.columns expectedColumns
  | .scale _ _, #[.matrix input] => operationalMatrixTypeEqual input output
  | .bggGrouping, #[.matrix vector, .matrix _, .matrix _] =>
      operationalMatrixTypeEqual vector output
  | _, _ => false

/-- Closed schemas for direct relation producers.  The concrete kernel performs the remaining
parameter, identity, and relation-inventory checks after correlated lane alignment. -/
def relationOperationSchemasValid
    (operation : DirectRelationOperation)
    (inputs : Array OperationalIndexedPayloadSchema)
    (output : MatrixTypeExpr) : Bool :=
  if !operationalMatrixTypeEqual operation.outputSchema output then false else
  match operation.kind, inputs with
  | .preimage .., #[.matrix publicType,
      .scalar (.trapdoor trapdoorType _ _ _ _), .matrix targetType] =>
      operationalMatrixTypeEqual publicType trapdoorType &&
        match inferOperationalProductMode publicType output with
        | .ok (_, productType) => operationalMatrixTypeEqual productType targetType
        | .error _ => false
  | .decomposition declaredType .., #[.matrix _] =>
      declaredType.closedIr?.any (operationalMatrixTypeEqual · output)
  | _, _ => false

def scalarOperationSchemasValid
    (kind : OperationalScalarPrimitiveKind)
    (inputs : Array OperationalIndexedPayloadSchema)
    (output : OperationalIndexedPayloadSchema) : Bool :=
  match kind with
  | .boolToInt => inputs == #[.scalar .boolean] && output == .scalar .integer
  | .intBinary _ => inputs == #[.scalar .integer, .scalar .integer] && output == .scalar .integer
  | .intCompare _ => inputs == #[.scalar .integer, .scalar .integer] && output == .scalar .boolean
  | .bitExtract _ => inputs == #[.scalar .integer] && output == .scalar .boolean
  | .intToReal => inputs == #[.scalar .integer] && output == .scalar .real
  | .realBinary _ => inputs == #[.scalar .real, .scalar .real] && output == .scalar .real
  | .realSqrt => inputs == #[.scalar .real] && output == .scalar .real

def pointwiseSchemasValid
    (operation : OperationalIndexedPointwiseOperation)
    (inputs : Array OperationalIndexedPayloadSchema)
    (output : OperationalIndexedPayloadSchema) : Bool :=
  match operation, output with
  | .matrix operation, .matrix matrixType =>
      matrixOperationSchemasValid operation inputs matrixType
  | .relation operation, .matrix matrixType =>
      relationOperationSchemasValid operation inputs matrixType
  | .scalar operation, _ => scalarOperationSchemasValid operation.kind inputs output
  | .matrixToScalar operation, .scalar output =>
      let oneMatrix := inputs.size == 1 && inputs.all (fun schema => match schema with
        | .matrix _ => true
        | .scalar _ => false)
      match operation.kind with
      | .extractCoefficient _ => oneMatrix && output == .integer
      | .thresholdDecodeBool _ _ _ => oneMatrix && output == .boolean
      | .thresholdDecodeInt _ _ _ => oneMatrix && output == .integer
  | .matrixFromScalar operation, .matrix output =>
      match operation.kind with
      | .liftIntegerToConstantPolynomial matrixType =>
          inputs == #[.scalar .integer] && operationalMatrixTypeEqual matrixType output
      | .trapdoorPublic matrixType => match inputs with
          | #[.scalar (.trapdoor trapdoorType _ _ _ _)] =>
              operationalMatrixTypeEqual matrixType trapdoorType &&
                operationalMatrixTypeEqual matrixType output
          | _ => false
  | _, _ => false

def DirectOperationalIndexedArena.pushValue
    (arena : DirectOperationalIndexedArena)
    (context : IndexContext)
    (payload : OperationalIndexedPayload) : DirectOperationalIndexedArena × OperationalIndexedValueId :=
  let value : OperationalIndexedValue := { context, payload, storage := payload.storage }
  ({ arena with values := arena.values.push value }, arena.values.size)

def DirectOperationalIndexedArena.pushShared
    (arena : DirectOperationalIndexedArena)
    (context : IndexContext)
    (schema : OperationalIndexedPayloadSchema)
    (reference : FixedOperationalPayloadRef) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) :=
  if validateContext context && arena.fixed.refHasSchema schema reference then
    some (arena.pushValue context (.shared schema reference))
  else none

def DirectOperationalIndexedArena.pushExplicit
    (environment : ParamEnvironment)
    (arena : DirectOperationalIndexedArena)
    (context : IndexContext)
    (binder : IndexVariable)
    (schema : OperationalIndexedPayloadSchema)
    (references : Array FixedOperationalPayloadRef) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) :=
  if context == { binders := #[binder] } && validateContext context &&
      explicitCountValid environment binder references &&
      references.all (arena.fixed.refHasSchema schema) then
    some (arena.pushValue context (.explicit schema binder references))
  else none

/-- Store an ordered table of already-authoritative direct values.  All lane schemas must agree;
their contexts are merged with the fresh family binder in first-occurrence order, so shared
selector variables remain one correlated dimension rather than a Cartesian expansion. -/
def DirectOperationalIndexedArena.pushExplicitValues
    (environment : ParamEnvironment)
    (arena : DirectOperationalIndexedArena)
    (binder : IndexVariable)
    (values : Array OperationalIndexedValueId) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let entries ← values.toList.mapM arena.valueAt?
  let first ← entries.head?
  if values.isEmpty || !explicitCountValid environment binder (Array.replicate values.size (.matrix 0)) ||
      !entries.all (fun entry => entry.payload.schema == first.payload.schema) then none
  let context ← mergeIndexContextsN (entries.map (·.context))
  let context ← extendContext context binder
  if !validateContext context then none
  some (arena.pushValue context (.explicitValues first.payload.schema binder values))

def DirectOperationalIndexedArena.pushMapped
    (arena : DirectOperationalIndexedArena)
    (source : OperationalIndexedValueId)
    (map : IndexMap) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let sourceValue ← arena.valueAt? source
  if !map.transportValid || map.source != sourceValue.context then none else
  match sourceValue.payload with
  | .mapped schema base prior => do
      if !prior.isDirectCarrierContextLift && prior.validate && map.validate then
        match composeIndexMap prior map with
        | some composed => some (arena.pushValue map.destination (.mapped schema base composed))
        | none => none
      else
        some (arena.pushValue map.destination (.mapped schema source map))
  | payload => some (arena.pushValue map.destination (.mapped payload.schema source map))

/-- Store a mapped view whose result schema has already crossed the same capture-free transport.
Only the compact schema is rebuilt at view construction; fixed facts, relation inventories, and
delayed pointwise children remain behind the view until one fixed-assignment reduction. -/
def DirectOperationalIndexedArena.pushMappedWithSchema
    (arena : DirectOperationalIndexedArena)
    (source : OperationalIndexedValueId)
    (map : IndexMap)
    (schema : OperationalIndexedPayloadSchema) : Option
    (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let sourceValue ← arena.valueAt? source
  if !map.transportValid || map.source != sourceValue.context then none else
  match sourceValue.payload with
  | .mapped _ base prior => do
      if !prior.isDirectCarrierContextLift && prior.validate && map.validate then
        let composed ← composeIndexMap prior map
        some (arena.pushValue map.destination (.mapped schema base composed))
      else some (arena.pushValue map.destination (.mapped schema source map))
  | _ => some (arena.pushValue map.destination (.mapped schema source map))

/-- Add a constant-size subject overlay without traversing or cloning the source DAG.  Nested
overlays collapse to the latest graph boundary: only that subject is observable at reduction. -/
def DirectOperationalIndexedArena.pushRebound
    (arena : DirectOperationalIndexedArena)
    (source : OperationalIndexedValueId)
    (subject : WireRef) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let value ← arena.valueAt? source
  match value.payload with
  /- Keep the externally observable map view outermost.  The rebound remains lazy beneath it,
  and reduction records the outer map before applying this subject overlay at the fixed leaf. -/
  | .mapped schema base map =>
      let baseValue ← arena.valueAt? base
      let (arena, rebound) := arena.pushValue baseValue.context
        (.rebound baseValue.payload.schema base subject)
      some (arena.pushValue value.context (.mapped schema rebound map))
  | .rebound schema base _ => some (arena.pushValue value.context (.rebound schema base subject))
  | payload => some (arena.pushValue value.context (.rebound payload.schema source subject))

/-- Annotate a direct matrix root after its fixed-assignment computation.  The source ID remains
authoritative for context, storage, schema, identity, provenance, and relations; evaluation alone
replaces the resulting total bound. -/
def DirectOperationalIndexedArena.pushMatrixResultBound
    (arena : DirectOperationalIndexedArena)
    (source : OperationalIndexedValueId)
    (totalHardBound : OperationalBoundExpr) :
    Option (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let value ← arena.valueAt? source
  match value.payload.schema with
  | .matrix _ => some (arena.pushValue value.context
      (.matrixResultBound value.payload.schema source totalHardBound))
  | .scalar _ => none

def DirectOperationalIndexedArena.pushPointwise
    (arena : DirectOperationalIndexedArena)
    (operation : OperationalIndexedPointwiseOperation)
    (inputs : Array OperationalIndexedValueId) : Option (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let values ← inputs.toList.mapM arena.valueAt?
  let (context, _) ← mergeIndexedFactShapeN values
  let schemas := values.toArray.map fun value => value.payload.schema
  let output ← match operation with
    | .matrix descriptor => some (.matrix descriptor.outputSchema)
    | .relation descriptor => some (.matrix descriptor.outputSchema)
    | .scalar { kind := .boolToInt, .. } | .scalar { kind := .intBinary _, .. } =>
        some (.scalar .integer)
    | .scalar { kind := .intCompare _, .. } | .scalar { kind := .bitExtract _, .. } =>
        some (.scalar .boolean)
    | .scalar { kind := .intToReal, .. } | .scalar { kind := .realBinary _, .. } |
        .scalar { kind := .realSqrt, .. } => some (.scalar .real)
    | .matrixToScalar { kind := .extractCoefficient _, .. }
    | .matrixToScalar { kind := .thresholdDecodeInt .., .. } => some (.scalar .integer)
    | .matrixToScalar { kind := .thresholdDecodeBool .., .. } => some (.scalar .boolean)
    | .matrixFromScalar { kind := .liftIntegerToConstantPolynomial matrixType, .. } =>
        some (.matrix matrixType)
    | .matrixFromScalar { kind := .trapdoorPublic matrixType, .. } =>
        some (.matrix matrixType)
  if pointwiseSchemasValid operation schemas output then
    some (arena.pushValue context (.pointwise output operation inputs))
  else none

/-- Rebuild one direct matrix value while transforming only the fixed matrix leaves reachable from
the requested root.  The direct carrier's contexts, index maps, delayed operations, and sharing
are retained; callers use this at graph-boundary rebinding points where subject metadata must
change without reconstructing the carrier. -/
partial def DirectOperationalIndexedArena.mapMatrixValue
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId)
    (mapFact : OperationalMatrixFact → Except OperationalError OperationalMatrixFact) :
    Except OperationalError (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let rec visit : Nat → DirectOperationalIndexedArena →
      Std.HashMap OperationalIndexedValueId OperationalIndexedValueId → OperationalIndexedValueId →
      Except OperationalError
        (DirectOperationalIndexedArena ×
          Std.HashMap OperationalIndexedValueId OperationalIndexedValueId × OperationalIndexedValueId)
    | 0, _, _, id => throw (.unsupportedOperationalExpr id)
    | fuel + 1, arena, memo, id => match memo[id]? with
      | some mapped => pure (arena, memo, mapped)
      | none => do
          let value ← match arena.valueAt? id with
            | some value => pure value
            | none => throw (.invalidOperationalExprRef id)
          if !validateContext value.context then throw (.unsupportedOperationalExpr id)
          let (arena, memo, mapped) ← match value.payload with
            | .shared (.matrix matrixType) (.matrix reference) => do
                let fact ← match arena.fixed.matrices[reference]? with
                  | some fact => pure fact
                  | none => throw (.invalidOperationalExprRef reference)
                let (fixed, replacement) := arena.fixed.pushMatrix (← mapFact fact)
                let direct := { arena with fixed }
                let (direct, mapped) ← match direct.pushShared value.context (.matrix matrixType)
                    replacement with
                  | some result => pure result
                  | none => throw (.unsupportedOperationalExpr id)
                pure (direct, memo, mapped)
            | .explicit (.matrix matrixType) binder references => do
                let (arena, references) ← references.foldlM (fun (arena, mapped) reference => do
                  let reference ← match reference with
                    | .matrix reference => pure reference
                    | .scalar _ => throw (.unsupportedOperationalExpr id)
                  let fact ← match arena.fixed.matrices[reference]? with
                    | some fact => pure fact
                    | none => throw (.invalidOperationalExprRef reference)
                  let (fixed, replacement) := arena.fixed.pushMatrix (← mapFact fact)
                  pure ({ arena with fixed }, mapped.push replacement)) (arena, #[])
                let (arena, mapped) := arena.pushValue value.context
                  (.explicit (.matrix matrixType) binder references)
                pure (arena, memo, mapped)
            | .explicitValues (.matrix matrixType) binder values => do
                let (arena, memo, values) ← values.foldlM (fun (arena, memo, mapped) value => do
                  let (arena, memo, value) ← visit fuel arena memo value
                  pure (arena, memo, mapped.push value)) (arena, memo, #[])
                let (arena, mapped) := arena.pushValue value.context
                  (.explicitValues (.matrix matrixType) binder values)
                pure (arena, memo, mapped)
            | .mapped (.matrix matrixType) source map => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) := arena.pushValue value.context
                  (.mapped (.matrix matrixType) source map)
                pure (arena, memo, mapped)
            | .rebound (.matrix _) source subject => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) ← match arena.pushRebound source subject with
                  | some result => pure result
                  | none => throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .matrixResultBound (.matrix matrixType) source totalHardBound => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) := arena.pushValue value.context
                  (.matrixResultBound (.matrix matrixType) source totalHardBound)
                pure (arena, memo, mapped)
            | .pointwise (.matrix matrixType) (.matrix operation) inputs => do
                let (arena, memo, inputs) ← inputs.foldlM (fun (arena, memo, mapped) input => do
                  let (arena, memo, input) ← visit fuel arena memo input
                  pure (arena, memo, mapped.push input)) (arena, memo, #[])
                let (arena, mapped) := arena.pushValue value.context
                  (.pointwise (.matrix matrixType) (.matrix operation) inputs)
                pure (arena, memo, mapped)
            | .pointwise (.matrix matrixType) (.relation operation) inputs => do
                let (arena, memo, inputs) ← inputs.foldlM (fun (arena, memo, mapped) input => do
                  let inputValue ← match arena.valueAt? input with
                    | some value => pure value | none => throw (.invalidOperationalExprRef input)
                  match inputValue.payload.schema with
                  | .matrix _ =>
                      let (arena, memo, input) ← visit fuel arena memo input
                      pure (arena, memo, mapped.push input)
                  | .scalar _ => pure (arena, memo, mapped.push input)) (arena, memo, #[])
                let (arena, mapped) := arena.pushValue value.context
                  (.pointwise (.matrix matrixType) (.relation operation) inputs)
                pure (arena, memo, mapped)
            /- Scalar-to-matrix lifts retain the scalar's direct identity and context.  The
            enclosing mapped matrix carrier transports the same selector; do not reject this
            executable scalar child merely because this traversal is matrix-rooted. -/
            | .pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs =>
                let (arena, mapped) := arena.pushValue value.context
                  (.pointwise (.matrix matrixType) (.matrixFromScalar operation) inputs)
                pure (arena, memo, mapped)
            | _ => throw (.unsupportedOperationalExpr id)
          pure (arena, memo.insert id mapped, mapped)
  let (arena, _, mapped) ← visit (arena.values.size + 1) arena {} root
  pure (arena, mapped)

/-- Scalar analogue of `mapMatrixValue`.  Graph-boundary rebinding must transport integer
producer subjects through direct scalar families without treating them as matrix leaves. -/
partial def DirectOperationalIndexedArena.mapScalarValue
    (environment : ParamEnvironment)
    (arena : DirectOperationalIndexedArena)
    (root : OperationalIndexedValueId)
    (mapFact : OperationalScalarFact → Except OperationalError OperationalScalarFact) :
    Except OperationalError (DirectOperationalIndexedArena × OperationalIndexedValueId) := do
  let rec visit : Nat → DirectOperationalIndexedArena →
      Std.HashMap OperationalIndexedValueId OperationalIndexedValueId → OperationalIndexedValueId →
      Except OperationalError (DirectOperationalIndexedArena ×
        Std.HashMap OperationalIndexedValueId OperationalIndexedValueId × OperationalIndexedValueId)
    | 0, _, _, id => throw (.unsupportedOperationalExpr id)
    | fuel + 1, arena, memo, id => match memo[id]? with
      | some mapped => pure (arena, memo, mapped)
      | none => do
          let value ← match arena.valueAt? id with
            | some value => pure value | none => throw (.invalidOperationalExprRef id)
          if !validateContext value.context then throw (.unsupportedOperationalExpr id)
          let (arena, memo, mapped) ← match value.payload with
            | .shared (.scalar scalarType) (.scalar reference) => do
                if !arena.fixed.refHasSchema (.scalar scalarType) (.scalar reference) then
                  throw (.unsupportedOperationalExpr id)
                let fact ← match arena.fixed.scalars[reference]? with
                  | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                let mappedFact ← mapFact fact
                let mappedType := operationalScalarSchema mappedFact
                let (fixed, replacement) := arena.fixed.pushScalar mappedFact
                let direct := { arena with fixed }
                let (direct, mapped) ← match direct.pushShared value.context (.scalar mappedType) replacement with
                  | some result => pure result | none => throw (.unsupportedOperationalExpr id)
                pure (direct, memo, mapped)
            | .explicit (.scalar scalarType) binder references => do
                if value.context != { binders := #[binder] } || !validateContext value.context ||
                    !explicitCountValid environment binder references ||
                    !references.all (arena.fixed.refHasSchema (.scalar scalarType)) then
                  throw (.unsupportedOperationalExpr id)
                let (arena, references, mappedType) ← references.foldlM
                    (fun (arena, mapped, mappedType) reference => do
                  let reference ← match reference with
                    | .scalar reference => pure reference | .matrix _ => throw (.unsupportedOperationalExpr id)
                  let fact ← match arena.fixed.scalars[reference]? with
                    | some fact => pure fact | none => throw (.invalidOperationalExprRef reference)
                  let mappedFact ← mapFact fact
                  let nextType := operationalScalarSchema mappedFact
                  if mappedType.any (· != nextType) then throw (.unsupportedOperationalExpr id)
                  let (fixed, replacement) := arena.fixed.pushScalar mappedFact
                  pure ({ arena with fixed }, mapped.push replacement, some nextType))
                    (arena, (#[] : Array FixedOperationalPayloadRef), none)
                let mappedType ← match mappedType with
                  | some mappedType => pure mappedType
                  | none => throw (.invalidCount id 0)
                let (arena, mapped) := arena.pushValue value.context
                  (.explicit (.scalar mappedType) binder references)
                pure (arena, memo, mapped)
            | .explicitValues (.scalar scalarType) binder values => do
                let originalValues ← match values.toList.mapM arena.valueAt? with
                  | some values => pure values
                  | none => throw (.invalidOperationalExprRef id)
                if values.isEmpty ||
                    !explicitCountValid environment binder (Array.replicate values.size (.matrix 0)) ||
                    !originalValues.all (fun child => child.payload.schema == .scalar scalarType) then
                  throw (.unsupportedOperationalExpr id)
                let originalContext ← match mergeIndexContextsN (originalValues.map (·.context)) with
                  | some context => pure context
                  | none => throw (.unsupportedOperationalExpr id)
                let originalContext ← match extendContext originalContext binder with
                  | some context => pure context
                  | none => throw (.unsupportedOperationalExpr id)
                if !validateContext value.context || value.context != originalContext then
                  throw (.unsupportedOperationalExpr id)
                let (arena, memo, values) ← values.foldlM (fun (arena, memo, mapped) value => do
                  let (arena, memo, value) ← visit fuel arena memo value
                  pure (arena, memo, mapped.push value)) (arena, memo, #[])
                let mappedType ← values.foldlM (fun current child => do
                  let child ← match arena.valueAt? child with
                    | some child => pure child
                    | none => throw (.invalidOperationalExprRef id)
                  let nextType ← match child.payload.schema with
                    | .scalar nextType => pure nextType
                    | .matrix _ => throw (.unsupportedOperationalExpr id)
                  if current.any (· != nextType) then throw (.unsupportedOperationalExpr id)
                  pure (some nextType)) none
                let mappedType ← match mappedType with
                  | some mappedType => pure mappedType
                  | none => throw (.invalidCount id 0)
                let (arena, mapped) := arena.pushValue value.context
                  (.explicitValues (.scalar mappedType) binder values)
                pure (arena, memo, mapped)
            | .mapped (.scalar scalarType) source map => do
                let originalSource ← match arena.valueAt? source with
                  | some value => pure value
                  | none => throw (.invalidOperationalExprRef source)
                if !map.transportValid || map.source != originalSource.context ||
                    map.destination != value.context ||
                    originalSource.payload.schema != .scalar scalarType then
                  throw (.unsupportedOperationalExpr id)
                let (arena, memo, source) ← visit fuel arena memo source
                let sourceValue ← match arena.valueAt? source with
                  | some value => pure value
                  | none => throw (.invalidOperationalExprRef source)
                match sourceValue.payload.schema with
                  | .scalar _ => pure ()
                  | .matrix _ => throw (.unsupportedOperationalExpr id)
                let (arena, mapped) ← match arena.pushMapped source map with
                  | some result => pure result
                  | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value
                  | none => throw (.invalidOperationalExprRef mapped)
                if mappedValue.context != value.context then throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .rebound (.scalar _) source subject => do
                let (arena, memo, source) ← visit fuel arena memo source
                let (arena, mapped) ← match arena.pushRebound source subject with
                  | some result => pure result
                  | none => throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | .pointwise (.scalar scalarType) operation inputs => do
                let originalInputs ← match inputs.toList.mapM arena.valueAt? with
                  | some inputs => pure inputs
                  | none => throw (.invalidOperationalExprRef id)
                let (originalContext, _) ← match mergeIndexedFactShapeN originalInputs with
                  | some shape => pure shape
                  | none => throw (.unsupportedOperationalExpr id)
                let originalSchemas := originalInputs.toArray.map fun input => input.payload.schema
                if !pointwiseSchemasValid operation originalSchemas (.scalar scalarType) ||
                    originalContext != value.context then
                  throw (.unsupportedOperationalExpr id)
                let (arena, memo, inputs) ← inputs.foldlM (fun (arena, memo, mapped) input => do
                  let inputValue ← match arena.valueAt? input with
                    | some value => pure value | none => throw (.invalidOperationalExprRef input)
                  match inputValue.payload.schema with
                  | .scalar _ =>
                      let (arena, memo, input) ← visit fuel arena memo input
                      pure (arena, memo, mapped.push input)
                  | .matrix _ => pure (arena, memo, mapped.push input)) (arena, memo, #[])
                let (arena, mapped) ← match arena.pushPointwise operation inputs with
                  | some result => pure result
                  | none => throw (.unsupportedOperationalExpr id)
                let mappedValue ← match arena.valueAt? mapped with
                  | some value => pure value
                  | none => throw (.invalidOperationalExprRef mapped)
                if mappedValue.context != value.context then throw (.unsupportedOperationalExpr id)
                pure (arena, memo, mapped)
            | _ => throw (.unsupportedOperationalExpr id)
          pure (arena, memo.insert id mapped, mapped)
  let (arena, _, mapped) ← visit (arena.values.size + 1) arena {} root
  pure (arena, mapped)

def directCarrierFixtureBinder (node : Nat) : IndexVariable := {
  owner := { stage := ⟨"direct-indexed-carrier-fixture"⟩, scope := ⟨[]⟩, node := ⟨node⟩ }
  slot := 0
  count := .constant 2
}

def directCarrierFixtureArena : DirectOperationalIndexedArena :=
  let (fixed, _) := ({} : FixedOperationalPayloadArena).pushScalar .real
  let (fixed, _) := fixed.pushScalar .real
  let (fixed, _) := fixed.pushScalar .boolean
  { fixed }

def directCarrierMatrixType
    (modulus ringDimension rows columns : Int) : MatrixTypeExpr := {
  modulus := .constant modulus
  ringDimension := .constant ringDimension
  rows := .constant rows
  columns := .constant columns
}

def directCarrierMatrixOperation
    (kind : PrimitiveOperationKind)
    (outputType : MatrixTypeExpr) : PrimitiveOperation := {
  kind := kind,
  outputType := .fromIr outputType,
  outputSchema := outputType,
  ownerScope := none,
  ownerNode := 0,
  outputPort := 0,
  parameterEnvironment := []
}

/-- Matrix addition requires both inputs to have exactly the declared output ring and shape. -/
example :
    let output := directCarrierMatrixType 17 2 3 4
    let wrongRing := directCarrierMatrixType 19 2 3 4
    let wrongShape := directCarrierMatrixType 17 2 3 5
    pointwiseSchemasValid (.matrix (directCarrierMatrixOperation (.add false) output))
      #[.matrix output, .matrix output] (.matrix output) &&
    !pointwiseSchemasValid (.matrix (directCarrierMatrixOperation (.add false) output))
      #[.matrix output, .matrix wrongRing] (.matrix output) &&
    !pointwiseSchemasValid (.matrix (directCarrierMatrixOperation (.add false) output))
      #[.matrix output, .matrix wrongShape] (.matrix output) = true := by
  native_decide

/-- Matrix multiplication requires an equal ring, compatible inner dimensions, and exactly the
inferred output dimensions. -/
example :
    let left := directCarrierMatrixType 17 2 3 4
    let right := directCarrierMatrixType 17 2 4 5
    let output := directCarrierMatrixType 17 2 3 5
    let wrongRing := directCarrierMatrixType 19 2 4 5
    let wrongInner := directCarrierMatrixType 17 2 6 5
    let wrongOutput := directCarrierMatrixType 17 2 3 6
    let operation := directCarrierMatrixOperation
      (.multiply .matrixMultiplyBound { node := 0, port := 0 }) output
    pointwiseSchemasValid (.matrix operation) #[.matrix left, .matrix right] (.matrix output) &&
    !pointwiseSchemasValid (.matrix operation) #[.matrix left, .matrix wrongRing] (.matrix output) &&
    !pointwiseSchemasValid (.matrix operation) #[.matrix left, .matrix wrongInner] (.matrix output) &&
    !pointwiseSchemasValid (.matrix operation) #[.matrix left, .matrix right] (.matrix wrongOutput) = true := by
  native_decide

/-- Fixed references name only direct fixed entries, and a missing entry is rejected before an
indexed value is allocated. -/
example : directCarrierFixtureArena.pushShared emptyContext (.scalar .real) (.scalar 99) = none := by
  native_decide

/-- The minimal explicit-table representation models exactly one binder.  An unrelated context
dimension cannot be attached without representing that additional table dimension. -/
example :
    let lane := directCarrierFixtureBinder 0
    let other := directCarrierFixtureBinder 1
    directCarrierFixtureArena.pushExplicit [] { binders := #[lane, other] } lane (.scalar .real)
      #[.scalar 0, .scalar 1] = none := by
  native_decide

/-- A valid ordered table retains its exact binder context and fixed-reference order. -/
example : (
    let lane := directCarrierFixtureBinder 0
    match directCarrierFixtureArena.pushExplicit [] { binders := #[lane] } lane (.scalar .real)
        #[.scalar 1, .scalar 0] with
    | some (arena, id) =>
        arena.values[id]?.any fun value => value.context == { binders := #[lane] } &&
          value.payload == .explicit (.scalar .real) lane #[.scalar 1, .scalar 0]
    | none => false) = true := by
  native_decide

/-- Nested mapped storage is flattened by composing maps.  The original value ID and payload
schema are preserved; no destination fixed reference can be forged. -/
example : (
    let sourceBinder := directCarrierFixtureBinder 2
    let middleBinder := directCarrierFixtureBinder 3
    let sourceContext : IndexContext := { binders := #[sourceBinder] }
    let middleContext : IndexContext := { binders := #[middleBinder] }
    let firstMap : IndexMap := {
      source := sourceContext
      destination := middleContext
      assignments := #[.variable middleBinder]
    }
    let secondMap : IndexMap := {
      source := middleContext
      destination := emptyContext
      assignments := #[.constant 1]
    }
    match directCarrierFixtureArena.pushShared sourceContext (.scalar .real) (.scalar 0) with
    | none => false
    | some (arena, source) =>
        match arena.pushMapped source firstMap with
        | none => false
        | some (arena, mapped) =>
            match arena.pushMapped mapped secondMap with
            | none => false
            | some (arena, result) =>
                arena.values[result]?.any fun value =>
                  value.context == emptyContext && value.payload ==
                    .mapped (.scalar .real) source {
                      source := sourceContext
                      destination := emptyContext
                      assignments := #[.constant 1]
                    }
    ) = true := by
  native_decide

/-- Pointwise lifting derives the merged context from its operands.  Independent binders remain
two correlated dimensions in one delayed operation; no Cartesian branch table is allocated. -/
example : (
    let leftBinder := directCarrierFixtureBinder 4
    let rightBinder := directCarrierFixtureBinder 5
    let realAdd : DirectScalarOperation := {
      kind := (OperationalScalarPrimitiveKind.realBinary Mxx.Ir.RealBinaryOp.add)
      ownerScope := none
      ownerNode := 0
      outputPort := 0 }
    match directCarrierFixtureArena.pushShared { binders := #[leftBinder] }
        (.scalar .real) (.scalar 0) with
    | none => false
    | some (arena, left) =>
        match arena.pushShared { binders := #[rightBinder] } (.scalar .real) (.scalar 1) with
        | none => false
        | some (arena, right) =>
            match arena.pushPointwise (.scalar realAdd) #[left, right] with
            | none => false
            | some (arena, result) =>
                arena.values.size == 3 && arena.values[result]?.any fun value =>
                  value.context == { binders := #[leftBinder, rightBinder] } &&
                    value.payload == .pointwise (.scalar .real)
                      (.scalar realAdd) #[left, right]
    ) = true := by
  native_decide

/-- Pointwise arity and schemas are checked by the closed operation registry. -/
example : (
    let realAdd : DirectScalarOperation := {
      kind := (OperationalScalarPrimitiveKind.realBinary Mxx.Ir.RealBinaryOp.add)
      ownerScope := none
      ownerNode := 0
      outputPort := 0 }
    let realSqrt : DirectScalarOperation := {
      kind := OperationalScalarPrimitiveKind.realSqrt
      ownerScope := none
      ownerNode := 0
      outputPort := 0 }
    match directCarrierFixtureArena.pushShared emptyContext (.scalar .real) (.scalar 0) with
    | none => false
    | some (arena, realValue) =>
        (arena.pushPointwise (.scalar realAdd) #[realValue]).isNone &&
          match arena.pushShared emptyContext (.scalar .boolean) (.scalar 2) with
          | none => false
          | some (arena, booleanValue) =>
              (arena.pushPointwise (.scalar realSqrt) #[booleanValue]).isNone
    ) = true := by
  native_decide


end Mxx.Certificate
