import Mxx.Certificate.Semantics
import Mxx.Certificate.Typing
import Mxx.Certificate.RecurrenceBasisAlignment

namespace Mxx.Certificate

def listToVector? {α : Type} (length : Nat) (values : List α) : Option (Vector α length) :=
  if h : values.length = length then some ⟨values.toArray, by simp [h]⟩ else none

private def MatrixRelation.kind : MatrixRelation → MatrixRelationKind
  | .preimage .. => .preimage
  | .gadgetDecomposition .. => .gadgetDecomposition

private def SignalTerm.schema
    (outputType : MatrixTypeExpr)
    (term : SignalTerm) : Option SignalTermSchema := do
  let coefficientType ← term.coefficient.expression.inferType
  let basisType := term.basis.inferType.getD <| match term.mode with
    | .ordinaryMatrixProduct => {
        outputType with rows := coefficientType.columns
      }
    | .leftPolynomialScalarBroadcast => outputType
    | .rightPolynomialScalarBroadcast | .swappedRowVectorScalarProduct => {
        outputType with rows := .constant 1, columns := .constant 1
      }
  return { coefficientType, basisType, mode := term.mode }

/-- Infer the fixed structural schema used to validate every lane or iteration of a template. -/
def ValueFact.schema (matrixType : Option MatrixTypeExpr) : ValueFact → Option ValueFactSchema
  | .matrix fact => do
      let type ← matrixType
      let primary ← match fact.primary with
        | .exact _ => pure .exact
        | .affine form => pure (.affine (← form.terms.mapM (SignalTerm.schema type)))
      return .matrix type primary (fact.relations.map MatrixRelation.kind)
        fact.coefficientRepresentation
  | .trapdoor _ => some .trapdoor
  | .integer _ => some .integer
  | .boolean _ => some .boolean
  | .bytes _ => some .bytes
  | .family fact => some (.family fact.count fact.elementSchema)

def ScopedWireFact.toTemplate (fact : ScopedWireFact) : Option ValueFactTemplate := do
  let schema ← fact.fact.schema fact.matrixType
  return { fact := fact.fact, schema }

private def ValueFact.ownedTemplateWire? : ValueFact → Option TemplateWireRef
  | .matrix fact =>
      match fact.subject with
      | .template wire => some wire
      | _ => none
  | .trapdoor fact =>
      match fact.privatePort with
      | .template wire => some wire
      | _ => none
  | .bytes (.template wire) => some wire
  | _ => none

/-- Recover the formal output-port mapping from a checked joint element tuple.  Scalar outputs do
not need a wire mapping because family-get reconstructs their typed runtime provenance directly. -/
def JointFamilyFact.bodyOutputTemplates (family : JointFamilyFact) :
    List (TemplateWireRef × Nat) :=
  family.elementTuple.toList.zipIdx.filterMap fun (template, slot) =>
    template.fact.ownedTemplateWire?.map (·, slot)

private def mapMatrixInstance
    (map : ValueInstanceRef → ValueInstanceRef)
    (reference : MatrixInstanceRef) : MatrixInstanceRef :=
  { reference with value := map reference.value }

private def ValueInstanceRef.appendInstancePath
    (suffix : InstancePathExpr) : ValueInstanceRef → ValueInstanceRef
  | .template wire => .instantiatedTemplate wire suffix
  | .instantiatedTemplate wire path => .instantiatedTemplate wire (path ++ suffix)
  | .familyElement aggregate index => .familyElement (aggregate.appendPath suffix) index
  | .recurrenceResult recurrence slot => .recurrenceResult (recurrence.appendPath suffix) slot
  | reference => reference

private def RuntimeExpr.appendInstancePath
    (suffix : InstancePathExpr) :
    {type : RuntimeScalarType} → RuntimeExpr type → RuntimeExpr type
  | _, .intWire wire => .intWire (wire.appendInstancePath suffix)
  | _, .boolWire wire => .boolWire (wire.appendInstancePath suffix)
  | _, .intConstant value => .intConstant value
  | _, .boolConstant value => .boolConstant value
  | _, .parameter value => .parameter value
  | _, .intBinary operation left right =>
      .intBinary operation (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | _, .compare operation left right =>
      .compare operation (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | _, .bitExtract value position => .bitExtract (value.appendInstancePath suffix) position
  | _, .boolToInt value => .boolToInt (value.appendInstancePath suffix)
  | _, .thresholdDecodeBool matrix ciphertextModulus plaintextModulus position =>
      .thresholdDecodeBool (matrix.appendInstancePath suffix) ciphertextModulus plaintextModulus
        position
  | _, .extractCoefficient matrix position => .extractCoefficient matrix position
  | _, .familyElement type aggregate indexRef index =>
      .familyElement type (aggregate.appendPath suffix) indexRef
        (index.appendInstancePath suffix)
  | _, .select type index branches => .select type (index.appendInstancePath suffix) branches
  | _, .loopIndex loop => .loopIndex loop
  | _, .carriedInput path => .carriedInput path

private def MatrixExpr.appendInstancePath
    (suffix : InstancePathExpr) : MatrixExpr → MatrixExpr
  | .wire reference => .wire {
      reference with value := reference.value.appendInstancePath suffix
    }
  | .zero type => .zero type
  | .identity type => .identity type
  | .gadget type base => .gadget type base
  | .add left right => .add (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .negate value => .negate (value.appendInstancePath suffix)
  | .multiply left right =>
      .multiply (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .scalarMultiply scalar value => .scalarMultiply scalar (value.appendInstancePath suffix)
  | .rowSlice value start stop => .rowSlice (value.appendInstancePath suffix) start stop
  | .rowConcat parts => .rowConcat (parts.map (·.appendInstancePath suffix))
  | .columnSlice value start stop => .columnSlice (value.appendInstancePath suffix) start stop
  | .columnConcat parts => .columnConcat (parts.map (·.appendInstancePath suffix))
  | .diagonalConcat parts => .diagonalConcat (parts.map (·.appendInstancePath suffix))
  | .rowCoefficientEmbed layout part value =>
      .rowCoefficientEmbed layout part (value.appendInstancePath suffix)
  | .columnBasisEmbed layout part value =>
      .columnBasisEmbed layout part (value.appendInstancePath suffix)
  | .diagonalCoefficientEmbed layout part value =>
      .diagonalCoefficientEmbed layout part (value.appendInstancePath suffix)
  | .diagonalBasisEmbed layout part value =>
      .diagonalBasisEmbed layout part (value.appendInstancePath suffix)
  | .select index branches =>
      .select (index.appendInstancePath suffix) (branches.map (·.appendInstancePath suffix))
  | .loopResult type recurrence path =>
      .loopResult type (recurrence.appendPath suffix) path
  | .carriedInput type path => .carriedInput type path

private def BoundExpr.appendInstancePath
    (suffix : InstancePathExpr) : BoundExpr → BoundExpr
  | .add left right => .add (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .multiply left right =>
      .multiply (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .maximum left right =>
      .maximum (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .floorDivide value divisor => .floorDivide (value.appendInstancePath suffix) divisor
  | .matrixProduct ring inner left right =>
      .matrixProduct ring inner (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .minimum left right =>
      .minimum (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .recurrenceResult recurrence path => .recurrenceResult (recurrence.appendPath suffix) path
  | expression => expression

private def IntBoundExpr.appendInstancePath
    (suffix : InstancePathExpr) : IntBoundExpr → IntBoundExpr
  | .negate value => .negate (value.appendInstancePath suffix)
  | .add left right => .add (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .subtract left right =>
      .subtract (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .multiply left right =>
      .multiply (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .divide left right => .divide (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .minimum left right =>
      .minimum (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .maximum left right =>
      .maximum (left.appendInstancePath suffix) (right.appendInstancePath suffix)
  | .recurrenceResult recurrence path => .recurrenceResult (recurrence.appendPath suffix) path
  | expression => expression

def RuntimeExpr.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef) :
    {type : RuntimeScalarType} → RuntimeExpr type → RuntimeExpr type
  | _, .intWire wire => .intWire (map wire)
  | _, .boolWire wire => .boolWire (map wire)
  | _, .intConstant value => .intConstant value
  | _, .boolConstant value => .boolConstant value
  | _, .parameter value => .parameter value
  | _, .intBinary operation left right =>
      .intBinary operation (left.mapInstances map) (right.mapInstances map)
  | _, .compare operation left right =>
      .compare operation (left.mapInstances map) (right.mapInstances map)
  | _, .bitExtract value position => .bitExtract (value.mapInstances map) position
  | _, .boolToInt value => .boolToInt (value.mapInstances map)
  | _, .thresholdDecodeBool matrix ciphertextModulus plaintextModulus position =>
      .thresholdDecodeBool (map matrix) ciphertextModulus plaintextModulus position
  | _, .extractCoefficient matrix position => .extractCoefficient matrix position
  | _, .familyElement type aggregate indexRef index =>
      .familyElement type aggregate indexRef (index.mapInstances map)
  | _, .select type index branches => .select type (index.mapInstances map) branches
  | _, .loopIndex loop => .loopIndex loop
  | _, .carriedInput path => .carriedInput path

def MatrixExpr.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef) : MatrixExpr → MatrixExpr
  | .wire reference => .wire (mapMatrixInstance map reference)
  | .zero type => .zero type
  | .identity type => .identity type
  | .gadget type base => .gadget type base
  | .add left right => .add (left.mapInstances map) (right.mapInstances map)
  | .negate value => .negate (value.mapInstances map)
  | .multiply left right => .multiply (left.mapInstances map) (right.mapInstances map)
  | .scalarMultiply scalar value => .scalarMultiply scalar (value.mapInstances map)
  | .rowSlice value start stop => .rowSlice (value.mapInstances map) start stop
  | .rowConcat parts => .rowConcat (parts.map (·.mapInstances map))
  | .columnSlice value start stop => .columnSlice (value.mapInstances map) start stop
  | .columnConcat parts => .columnConcat (parts.map (·.mapInstances map))
  | .diagonalConcat parts => .diagonalConcat (parts.map (·.mapInstances map))
  | .rowCoefficientEmbed layout part value =>
      .rowCoefficientEmbed layout part (value.mapInstances map)
  | .columnBasisEmbed layout part value =>
      .columnBasisEmbed layout part (value.mapInstances map)
  | .diagonalCoefficientEmbed layout part value =>
      .diagonalCoefficientEmbed layout part (value.mapInstances map)
  | .diagonalBasisEmbed layout part value =>
      .diagonalBasisEmbed layout part (value.mapInstances map)
  | .select index branches =>
      .select (index.mapInstances map) (branches.map (·.mapInstances map))
  | .loopResult type recurrence path => .loopResult type recurrence path
  | .carriedInput type path => .carriedInput type path

private def BoundedMatrixExpr.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef)
    (expression : BoundedMatrixExpr) : BoundedMatrixExpr :=
  { expression with expression := expression.expression.mapInstances map }

private def SignalTerm.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef)
    (term : SignalTerm) : SignalTerm := {
  term with
  coefficient := term.coefficient.mapInstances map
  basis := term.basis.mapInstances map
}

private def AffineForm.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef)
    (form : AffineForm) : AffineForm :=
  { form with terms := form.terms.map (·.mapInstances map) }

private def MatrixPrimaryForm.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef) : MatrixPrimaryForm → MatrixPrimaryForm
  | .exact expression => .exact (expression.mapInstances map)
  | .affine form => .affine (form.mapInstances map)

private def MatrixRelation.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef) : MatrixRelation → MatrixRelation
  | .preimage subject source target trapdoor =>
      .preimage (map subject) (mapMatrixInstance map source) (mapMatrixInstance map target)
        (map trapdoor)
  | .gadgetDecomposition subject target base digitCount =>
      .gadgetDecomposition (map subject) (mapMatrixInstance map target) base digitCount

private def BoundedMatrixExpr.appendInstancePath
    (suffix : InstancePathExpr)
    (expression : BoundedMatrixExpr) : BoundedMatrixExpr := {
  expression := expression.expression.appendInstancePath suffix
  normBound := expression.normBound.appendInstancePath suffix
}

private def SignalTerm.appendInstancePath
    (suffix : InstancePathExpr)
    (term : SignalTerm) : SignalTerm := {
  term with
  coefficient := term.coefficient.appendInstancePath suffix
  basis := term.basis.appendInstancePath suffix
}

private def MatrixPrimaryForm.appendInstancePath
    (suffix : InstancePathExpr) : MatrixPrimaryForm → MatrixPrimaryForm
  | .exact expression => .exact (expression.appendInstancePath suffix)
  | .affine form => .affine {
      terms := form.terms.map (·.appendInstancePath suffix)
      noiseBound := form.noiseBound.appendInstancePath suffix
    }

private def MatrixRelation.appendInstancePath
    (suffix : InstancePathExpr) : MatrixRelation → MatrixRelation
  | .preimage subject source target trapdoor => .preimage
      (subject.appendInstancePath suffix)
      { source with value := source.value.appendInstancePath suffix }
      { target with value := target.value.appendInstancePath suffix }
      (trapdoor.appendInstancePath suffix)
  | .gadgetDecomposition subject target base digitCount => .gadgetDecomposition
      (subject.appendInstancePath suffix)
      { target with value := target.value.appendInstancePath suffix }
      base digitCount

private def ValueFact.appendInstancePath
    (suffix : InstancePathExpr) : ValueFact → ValueFact
  | .matrix fact => .matrix {
      fact with
      subject := fact.subject.appendInstancePath suffix
      primary := fact.primary.appendInstancePath suffix
      relations := fact.relations.map (·.appendInstancePath suffix)
      totalNormBound := fact.totalNormBound.appendInstancePath suffix
    }
  | .trapdoor fact => .trapdoor {
      privatePort := fact.privatePort.appendInstancePath suffix
      publicPort := fact.publicPort.appendInstancePath suffix
      publicMatrix := fact.publicMatrix.appendInstancePath suffix
    }
  | .integer fact => .integer {
      expression := fact.expression.appendInstancePath suffix
      lower := fact.lower.appendInstancePath suffix
      upper := fact.upper.appendInstancePath suffix
    }
  | .boolean fact => .boolean {
      expression := fact.expression.appendInstancePath suffix
    }
  | .bytes wire => .bytes (wire.appendInstancePath suffix)
  | .family fact => .family {
      fact with aggregate := fact.aggregate.appendPath suffix
    }

def ValueFact.mapInstances
    (map : ValueInstanceRef → ValueInstanceRef) : ValueFact → ValueFact
  | .matrix fact => .matrix {
      fact with
      subject := map fact.subject
      primary := fact.primary.mapInstances map
      relations := fact.relations.map (·.mapInstances map)
    }
  | .trapdoor fact => .trapdoor {
      fact with
      privatePort := map fact.privatePort
      publicPort := map fact.publicPort
      publicMatrix := fact.publicMatrix.mapInstances map
    }
  | .integer fact => .integer {
      fact with expression := fact.expression.mapInstances map
    }
  | .boolean fact => .boolean {
      fact with expression := fact.expression.mapInstances map
    }
  | .bytes wire => .bytes (map wire)
  | .family fact => .family fact

/-- Retarget only the identity owned by an output alias. Provenance references inside expressions
and relation source/target/trapdoor fields remain unchanged. -/
def ValueFact.retargetOutput
    (output : ValueInstanceRef) : ValueFact → ValueFact
  | .matrix fact => .matrix {
      fact with
      subject := output
      relations := fact.relations.map (·.retargetSubject output)
    }
  | .trapdoor fact => .trapdoor { fact with privatePort := output }
  | fact => fact

private def encodeIdentityPart (value : String) : String :=
  s!"{value.length}:{value}"

/-- Canonical, injective-by-components naming convention for a parallel-loop family.  The
length-prefixed scope components prevent nested definitions with delimiter-bearing names from
colliding. -/
def parallelJointFamilyId (site : CoreNodeRef) : JointFamilyId := ⟨
  "parallel:" ++ encodeIdentityPart site.stage.name ++
    String.join (site.scope.path.map (fun part ↦ ":" ++ encodeIdentityPart part)) ++
    s!":node:{site.node.value}"
⟩

/-- Structural identity for a sequential-loop recurrence. -/
def sequentialSequentialRecurrenceRef (site : CoreNodeRef) : SequentialRecurrenceRef := ⟨site⟩

/-- Analyzer-owned substitution from a parallel body template to one exact lane. The constructor
is deliberately private: the derivation and both index identities come from frozen analysis. -/
structure ParallelTemplateSubstitution where
  derivation : ParallelFamilyDerivationSource
  actualIndex : RuntimeExprRef .integer
  actualIndexExpression : RuntimeExpr .integer
  originContext : Option MatrixOriginNormalizationContext := none

private def ParallelTemplateSubstitution.templateIndex
    (substitution : ParallelTemplateSubstitution) : RuntimeExprRef .integer :=
  substitution.derivation.indexReference

private def ParallelTemplateSubstitution.laneFrame
    (substitution : ParallelTemplateSubstitution) : InstanceFrame :=
  .parallelLane substitution.derivation.loopSite substitution.actualIndex

private def ParallelTemplateSubstitution.replaceIndex
    (substitution : ParallelTemplateSubstitution) (index : RuntimeExprRef .integer) :
    RuntimeExprRef .integer :=
  if index = substitution.templateIndex then substitution.actualIndex else index

private def ParallelTemplateSubstitution.replaceIndexExpression
    (substitution : ParallelTemplateSubstitution)
    (indexRef : RuntimeExprRef .integer)
    (index : RuntimeExpr .integer) : RuntimeExpr .integer :=
  if indexRef = substitution.templateIndex then substitution.actualIndexExpression else index

private def ParallelTemplateSubstitution.replacePath
    (substitution : ParallelTemplateSubstitution) : InstancePathExpr → InstancePathExpr
  | [] => []
  | .subgraphCall site :: tail => .subgraphCall site :: substitution.replacePath tail
  | .parallelLane site index :: tail =>
      .parallelLane site (substitution.replaceIndex index) :: substitution.replacePath tail
  | .sequentialIteration site index :: tail =>
      .sequentialIteration site (substitution.replaceIndex index) :: substitution.replacePath tail

private def findParallelOutputSlot
    (wire : TemplateWireRef) : Nat → List ValueFactTemplate → Option Nat
  | _, [] => none
  | slot, template :: tail =>
      match template.fact.ownedTemplateWire? with
      | some candidate => if candidate = wire then some slot else findParallelOutputSlot wire (slot + 1) tail
      | none => findParallelOutputSlot wire (slot + 1) tail

private def ParallelTemplateSubstitution.outputSlot?
    (substitution : ParallelTemplateSubstitution)
    (wire : TemplateWireRef) : Option Nat :=
  findParallelOutputSlot wire 0 substitution.derivation.elementTemplates

private def ParallelTemplateSubstitution.replaceAggregate
    (substitution : ParallelTemplateSubstitution) : FamilyAggregateRef → FamilyAggregateRef
  | .joint joint outputSlot path => .joint joint outputSlot (substitution.replacePath path)
  | .familyElement parent index =>
      .familyElement (substitution.replaceAggregate parent) (substitution.replaceIndex index)
  | .carriedInput slot => .carriedInput slot
  | .recurrenceResult recurrence path slot =>
      .recurrenceResult recurrence (substitution.replacePath path) slot

private def ParallelTemplateSubstitution.replaceValue
    (substitution : ParallelTemplateSubstitution) : ValueInstanceRef → ValueInstanceRef
  | .template wire =>
      match substitution.outputSlot? wire with
      | some outputSlot => .familyElement (.joint substitution.derivation.family outputSlot [])
          substitution.actualIndex
      | none => .instantiatedTemplate wire [substitution.laneFrame]
  | .instantiatedTemplate wire path =>
      .instantiatedTemplate wire (substitution.replacePath path)
  | .familyElement aggregate index =>
      .familyElement (substitution.replaceAggregate aggregate) (substitution.replaceIndex index)
  | .recurrenceResult recurrence slot => .recurrenceResult recurrence slot
  | value => value

private def ParallelTemplateSubstitution.replaceMatrixInstance
    (substitution : ParallelTemplateSubstitution) (reference : MatrixInstanceRef) : MatrixInstanceRef :=
  { reference with value := substitution.replaceValue reference.value }

/-- References inside a scalar or matrix `select` are arena entries, rather than nested syntax.
When a parallel template is instantiated, every old arena entry is rewritten once in order and
interned after the immutable source arena. This preserves the arena's older-reference invariant
and prevents a selected branch from retaining the template lane by accident. -/
private structure ParallelArenaRewriteState where
  arena : ExpressionArena
  integerReferences : List (RuntimeExprRef .integer × RuntimeExprRef .integer) := []
  booleanReferences : List (RuntimeExprRef .boolean × RuntimeExprRef .boolean) := []
  matrixReferences : List (MatrixExprRef × MatrixExprRef) := []

private inductive ParallelTemplateInstantiationError where
  | invalidExpressionReference
  deriving BEq, DecidableEq, Repr

private def ParallelArenaRewriteState.lookupInteger
    (state : ParallelArenaRewriteState) (reference : RuntimeExprRef .integer) :
    Option (RuntimeExprRef .integer) :=
  (state.integerReferences.find? fun entry => entry.1 == reference).map (·.2)

private def ParallelArenaRewriteState.lookupBoolean
    (state : ParallelArenaRewriteState) (reference : RuntimeExprRef .boolean) :
    Option (RuntimeExprRef .boolean) :=
  (state.booleanReferences.find? fun entry => entry.1 == reference).map (·.2)

private def ParallelArenaRewriteState.lookupMatrix
    (state : ParallelArenaRewriteState) (reference : MatrixExprRef) : Option MatrixExprRef :=
  (state.matrixReferences.find? fun entry => entry.1 == reference).map (·.2)

private def ParallelTemplateSubstitution.rewriteIntegerReference
    (substitution : ParallelTemplateSubstitution)
    (state : ParallelArenaRewriteState) (reference : RuntimeExprRef .integer) :
    Except ParallelTemplateInstantiationError (RuntimeExprRef .integer) :=
  if reference == substitution.templateIndex then pure substitution.actualIndex
  else match state.lookupInteger reference with
    | some rewritten => pure rewritten
    | none => throw .invalidExpressionReference

private def ParallelArenaRewriteState.rewriteBooleanReference
    (state : ParallelArenaRewriteState) (reference : RuntimeExprRef .boolean) :
    Except ParallelTemplateInstantiationError (RuntimeExprRef .boolean) :=
  match state.lookupBoolean reference with
  | some rewritten => pure rewritten
  | none => throw .invalidExpressionReference

private def ParallelArenaRewriteState.rewriteMatrixReference
    (state : ParallelArenaRewriteState) (reference : MatrixExprRef) :
    Except ParallelTemplateInstantiationError MatrixExprRef :=
  match state.lookupMatrix reference with
  | some rewritten => pure rewritten
  | none => throw .invalidExpressionReference

private def ParallelTemplateSubstitution.rewriteRuntime
    (substitution : ParallelTemplateSubstitution)
    (state : ParallelArenaRewriteState) :
    {type : RuntimeScalarType} → RuntimeExpr type → Except ParallelTemplateInstantiationError (RuntimeExpr type)
  | _, .intWire wire => pure (.intWire (substitution.replaceValue wire))
  | _, .boolWire wire => pure (.boolWire (substitution.replaceValue wire))
  | _, .intConstant value => pure (.intConstant value)
  | _, .boolConstant value => pure (.boolConstant value)
  | _, .parameter value => pure (.parameter value)
  | _, .intBinary operation left right =>
      return .intBinary operation (← substitution.rewriteRuntime state left)
        (← substitution.rewriteRuntime state right)
  | _, .compare operation left right =>
      return .compare operation (← substitution.rewriteRuntime state left)
        (← substitution.rewriteRuntime state right)
  | _, .bitExtract value position =>
      return .bitExtract (← substitution.rewriteRuntime state value) position
  | _, .boolToInt value => return .boolToInt (← substitution.rewriteRuntime state value)
  | _, .thresholdDecodeBool matrix q p position =>
      pure (.thresholdDecodeBool (substitution.replaceValue matrix) q p position)
  | _, .extractCoefficient matrix position =>
      return .extractCoefficient (← state.rewriteMatrixReference matrix) position
  | _, .familyElement type aggregate indexRef index => do
      let rewrittenIndex ← substitution.rewriteIntegerReference state indexRef
      return .familyElement type (substitution.replaceAggregate aggregate) rewrittenIndex
        (substitution.replaceIndexExpression indexRef (← substitution.rewriteRuntime state index))
  | .integer, .select .integer index branches => do
      let rewrittenBranches ← branches.mapM (substitution.rewriteIntegerReference state)
      return .select .integer (← substitution.rewriteRuntime state index) rewrittenBranches
  | .boolean, .select .boolean index branches => do
      let rewrittenBranches ← branches.mapM state.rewriteBooleanReference
      return .select .boolean (← substitution.rewriteRuntime state index) rewrittenBranches
  | _, .loopIndex loop => pure (.loopIndex loop)
  | _, .carriedInput path => pure (.carriedInput path)

private def ParallelTemplateSubstitution.rewriteMatrix
    (substitution : ParallelTemplateSubstitution)
    (state : ParallelArenaRewriteState) : MatrixExpr → Except ParallelTemplateInstantiationError MatrixExpr
  | .wire reference => pure (.wire (substitution.replaceMatrixInstance reference))
  | .zero type => pure (.zero type)
  | .identity type => pure (.identity type)
  | .gadget type base => pure (.gadget type base)
  | .add left right => do
      return .add (← substitution.rewriteMatrix state left) (← substitution.rewriteMatrix state right)
  | .negate value => do return .negate (← substitution.rewriteMatrix state value)
  | .multiply left right => do
      return .multiply (← substitution.rewriteMatrix state left) (← substitution.rewriteMatrix state right)
  | .scalarMultiply scalar value => do
      return .scalarMultiply scalar (← substitution.rewriteMatrix state value)
  | .rowSlice value start stop => do
      return .rowSlice (← substitution.rewriteMatrix state value) start stop
  | .rowConcat parts => do return .rowConcat (← parts.mapM (substitution.rewriteMatrix state))
  | .columnSlice value start stop => do
      return .columnSlice (← substitution.rewriteMatrix state value) start stop
  | .columnConcat parts => do return .columnConcat (← parts.mapM (substitution.rewriteMatrix state))
  | .diagonalConcat parts => do return .diagonalConcat (← parts.mapM (substitution.rewriteMatrix state))
  | .rowCoefficientEmbed layout part value => do
      return .rowCoefficientEmbed layout part (← substitution.rewriteMatrix state value)
  | .columnBasisEmbed layout part value => do
      return .columnBasisEmbed layout part (← substitution.rewriteMatrix state value)
  | .diagonalCoefficientEmbed layout part value => do
      return .diagonalCoefficientEmbed layout part (← substitution.rewriteMatrix state value)
  | .diagonalBasisEmbed layout part value => do
      return .diagonalBasisEmbed layout part (← substitution.rewriteMatrix state value)
  | .select index branches => do
      return .select (← substitution.rewriteRuntime state index)
        (← branches.mapM (substitution.rewriteMatrix state))
  | .loopResult type recurrence path => pure (.loopResult type recurrence path)
  | .carriedInput type path => pure (.carriedInput type path)

private def ParallelTemplateSubstitution.rewriteArenaEntries
    (substitution : ParallelTemplateSubstitution)
    (original : List SymbolicExprEntry)
    (entryIndex : Nat)
    (state : ParallelArenaRewriteState) : Except ParallelTemplateInstantiationError ParallelArenaRewriteState := do
  match original with
  | [] => pure state
  | .integer expression :: tail =>
      let expression ← substitution.rewriteRuntime state expression
      let ⟨arena, reference⟩ ← match state.arena.internInteger expression with
        | some result => pure result
        | none => throw .invalidExpressionReference
      substitution.rewriteArenaEntries tail (entryIndex + 1) {
        state with arena, integerReferences := state.integerReferences ++ [(⟨entryIndex⟩, reference)]
      }
  | .boolean expression :: tail =>
      let expression ← substitution.rewriteRuntime state expression
      let ⟨arena, reference⟩ ← match state.arena.internBoolean expression with
        | some result => pure result
        | none => throw .invalidExpressionReference
      substitution.rewriteArenaEntries tail (entryIndex + 1) {
        state with arena, booleanReferences := state.booleanReferences ++ [(⟨entryIndex⟩, reference)]
      }
  | .matrix expression :: tail =>
      let expression ← substitution.rewriteMatrix state expression
      let ⟨arena, reference⟩ ← match state.arena.internMatrix expression with
        | some result => pure result
        | none => throw .invalidExpressionReference
      substitution.rewriteArenaEntries tail (entryIndex + 1) {
        state with arena, matrixReferences := state.matrixReferences ++ [(⟨entryIndex⟩, reference)]
      }

private def ParallelTemplateSubstitution.rewriteArena
    (substitution : ParallelTemplateSubstitution)
    (arena : ExpressionArena) : Except ParallelTemplateInstantiationError ParallelArenaRewriteState :=
  substitution.rewriteArenaEntries arena.entries 0 { arena }

private def ParallelTemplateSubstitution.aliasesFor
    (substitution : ParallelTemplateSubstitution)
  (source : MatrixInstanceRef) : List MatrixAliasTemplate :=
  substitution.derivation.matrixAliasTemplates.filter fun candidate =>
    source.type == candidate.subjectType &&
      source.value == substitution.replaceValue (.template candidate.subject)

/-- Alias resolution only admits wire aliases and a scalar-one wrapper.  It therefore does not
need to reinterpret arbitrary arena-backed selections; all other forms are rejected by the
origin normalizer immediately after this structural transport. -/
private def ParallelTemplateSubstitution.replaceOriginExpression
    (substitution : ParallelTemplateSubstitution) : MatrixExpr → MatrixExpr
  | .wire reference => .wire (substitution.replaceMatrixInstance reference)
  | .scalarMultiply (.constant 1) value =>
      .scalarMultiply (.constant 1) (substitution.replaceOriginExpression value)
  | expression => expression

/-- Only a frozen producer derivation may resolve the aliases it recorded.  The body templates
are intentionally not compared here: they can contain arbitrary symbolic facts, whereas this
identity is the immutable location of the producer loop and its indexed family. -/
private def sameParallelDerivationIdentity
    (left right : ParallelFamilyDerivationSource) : Bool :=
  left.family == right.family &&
    left.loopSite == right.loopSite &&
    left.childScope == right.childScope &&
    left.definition == right.definition &&
    left.indexSlot == right.indexSlot &&
    left.indexReference == right.indexReference

private def resolveRelationSourceOriginFrom
    (derivation : ParallelFamilyDerivationSource)
    (substitution : ParallelTemplateSubstitution)
    (fuel : Nat)
    (visited : List TemplateWireRef)
    (source : MatrixInstanceRef) : Except RecurrenceBasisAlignmentError IndexedMatrixOrigin :=
  match fuel with
  | 0 => .error .unsupportedOrigin
  | fuel + 1 => do
  let aliases := substitution.aliasesFor source
  let candidate ← match aliases with
    | [candidate] => pure candidate
    | [] => throw .missingAlias
    | _ => throw .ambiguousAlias
  if source.type == candidate.subjectType then pure () else throw .aliasTypeMismatch
  if visited.contains candidate.subject then throw .unsupportedOrigin
  let target := substitution.replaceOriginExpression candidate.exactTarget
  let nextVisited := candidate.subject :: visited
  match target with
  | .wire next =>
      if (substitution.aliasesFor next).isEmpty then
        match substitution.originContext with
        | some context => normalizeMatrixOrigin context.arena context.selectedLoop context.indexSlot target
        | none => throw .unsupportedOrigin
      else resolveRelationSourceOriginFrom derivation substitution fuel nextVisited next
  | .scalarMultiply (.constant 1) _ =>
      match substitution.originContext with
      | some context => normalizeMatrixOrigin context.arena context.selectedLoop context.indexSlot target
      | none => throw .unsupportedOrigin
  | _ => throw .unsupportedOrigin

/-- Resolve an instantiated body-local preimage source through only the producer-local exact
alias table retained by its own parallel derivation. The result is a structural origin used for
recurrence comparison; the raw relation endpoint remains untouched for execution semantics. -/
def resolveRelationSourceOrigin
    (derivation : ParallelFamilyDerivationSource)
    (substitution : ParallelTemplateSubstitution)
    (rawSource : MatrixInstanceRef) : Except RecurrenceBasisAlignmentError IndexedMatrixOrigin :=
  if sameParallelDerivationIdentity derivation substitution.derivation then
    resolveRelationSourceOriginFrom derivation substitution
      (derivation.matrixAliasTemplates.length + 1) [] rawSource
  else .error .unsupportedOrigin

private def ParallelTemplateSubstitution.replaceBound
    (substitution : ParallelTemplateSubstitution) : BoundExpr → BoundExpr
  | .add left right => .add (substitution.replaceBound left) (substitution.replaceBound right)
  | .multiply left right =>
      .multiply (substitution.replaceBound left) (substitution.replaceBound right)
  | .maximum left right =>
      .maximum (substitution.replaceBound left) (substitution.replaceBound right)
  | .floorDivide value divisor => .floorDivide (substitution.replaceBound value) divisor
  | .matrixProduct ring inner left right =>
      .matrixProduct ring inner (substitution.replaceBound left) (substitution.replaceBound right)
  | .minimum left right =>
      .minimum (substitution.replaceBound left) (substitution.replaceBound right)
  | bound => bound

private def ParallelTemplateSubstitution.replaceIntBound
    (substitution : ParallelTemplateSubstitution) : IntBoundExpr → IntBoundExpr
  | .integer value => .integer value
  | .natural value => .natural (substitution.replaceBound value)
  | .negate value => .negate (substitution.replaceIntBound value)
  | .add left right =>
      .add (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | .subtract left right =>
      .subtract (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | .multiply left right =>
      .multiply (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | .divide left right =>
      .divide (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | .minimum left right =>
      .minimum (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | .maximum left right =>
      .maximum (substitution.replaceIntBound left) (substitution.replaceIntBound right)
  | expression => expression

private def ParallelTemplateSubstitution.replaceRelation
    (substitution : ParallelTemplateSubstitution) : MatrixRelation → Except ParallelTemplateInstantiationError MatrixRelation
  | .preimage subject source target trapdoor => pure (.preimage
      (substitution.replaceValue subject)
      (substitution.replaceMatrixInstance source)
      (substitution.replaceMatrixInstance target)
      (substitution.replaceValue trapdoor))
  | .gadgetDecomposition subject target base digitCount => pure (.gadgetDecomposition
      (substitution.replaceValue subject)
      (substitution.replaceMatrixInstance target)
      base digitCount)

private def ParallelTemplateSubstitution.replaceFact
    (substitution : ParallelTemplateSubstitution)
    (state : ParallelArenaRewriteState) : ValueFact → Except ParallelTemplateInstantiationError ValueFact
  | .matrix fact => do
      let primary ← match fact.primary with
        | .exact expression => do
            let expression ← substitution.rewriteMatrix state expression
            pure (.exact expression)
        | .affine form => do
            let terms ← form.terms.mapM fun term => do
              let coefficient ← substitution.rewriteMatrix state term.coefficient.expression
              let basis ← substitution.rewriteMatrix state term.basis
              pure {
                term with
                coefficient := {
                  expression := coefficient
                  normBound := substitution.replaceBound term.coefficient.normBound
                }
                basis
              }
            pure (.affine {
              terms
              noiseBound := substitution.replaceBound form.noiseBound
            })
      let relations ← fact.relations.mapM substitution.replaceRelation
      pure (.matrix {
        fact with
        subject := substitution.replaceValue fact.subject
        primary
        relations
        totalNormBound := substitution.replaceBound fact.totalNormBound
      })
  | .trapdoor fact => do
      let publicMatrix ← substitution.rewriteMatrix state fact.publicMatrix
      pure (.trapdoor {
        privatePort := substitution.replaceValue fact.privatePort
        publicPort := substitution.replaceValue fact.publicPort
        publicMatrix
      })
  | .integer fact => do
      let expression ← substitution.rewriteRuntime state fact.expression
      pure (.integer {
        fact with
        expression
        lower := substitution.replaceIntBound fact.lower
        upper := substitution.replaceIntBound fact.upper
      })
  | .boolean fact => do
      let expression ← substitution.rewriteRuntime state fact.expression
      pure (.boolean { fact with expression })
  | .bytes wire => pure (.bytes (substitution.replaceValue wire))
  | .family fact => pure (.family { fact with aggregate := substitution.replaceAggregate fact.aggregate })

def instantiateParallelTemplate
    (substitution : ParallelTemplateSubstitution)
    (arena : ExpressionArena)
    (output : ValueInstanceRef)
    (template : ValueFactTemplate) :
    Except ParallelTemplateInstantiationError (ExpressionArena × ValueFact) := do
  let state ← substitution.rewriteArena arena
  let substitution := match substitution.originContext with
    | some context => { substitution with originContext := some { context with arena := state.arena } }
    | none => substitution
  return (state.arena, (← substitution.replaceFact state template.fact).retargetOutput output)

private def familyTestType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def familyTestTemplate : TemplateWireRef where
  definition := { stage := ⟨"family-test"⟩, name := "body" }
  bodyScope := ⟨[]⟩
  node := ⟨0⟩
  port := 0

private def familyTestFact : ValueFactTemplate := {
  fact := .matrix {
    subject := .template familyTestTemplate
    primary := .exact (.wire { value := .template familyTestTemplate, type := familyTestType })
    relations := []
    totalNormBound := .constant 8
  }
  schema := .matrix familyTestType .exact [] .unknown
}

private def familyTestLoopSite : CoreNodeRef :=
  { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨1⟩ }

private def familyTestIndex : RuntimeExprRef .integer := ⟨0⟩

private def familyTestSource : ParallelFamilyDerivationSource := {
  family := ⟨"joint"⟩
  loopSite := familyTestLoopSite
  childScope := ⟨["body"]⟩
  definition := "body"
  count := .constant 5
  indexSlot := 0
  indexReference := familyTestIndex
  indexExpression := .loopIndex { site := familyTestLoopSite }
  bindings := []
  modes := []
  argumentRefs := []
  outputCount := 1
  outputTypes := [.matrix familyTestType]
  body := { nodes := [], outputs := [], inputNames := [] }
  seededFacts := []
  analyzedFacts := []
  outputFacts := []
  elementTemplates := [familyTestFact]
}

private def familyTestSubstitution : ParallelTemplateSubstitution := {
  derivation := familyTestSource
  actualIndex := ⟨3⟩
  actualIndexExpression := .intConstant 3
}

/-- A scalar select owns branch references in the expression arena.  Instantiation rewrites both
branches and allocates fresh references after the immutable source entries. -/
private def familySelectArena : ExpressionArena := { entries := [
  .integer (.intConstant 0),
  .integer (.intWire (.template familyTestTemplate)),
  .integer (.select .integer (.intConstant 0) [⟨0⟩, ⟨1⟩])
] }

example :
    (familyTestSubstitution.rewriteArena familySelectArena).map
      (fun state => state.arena.lookupInteger ⟨5⟩) =
      .ok (some (.select .integer (.intConstant 0) [⟨3⟩, ⟨4⟩])) := by
  rfl

private def familyTestLocalTemplate : TemplateWireRef where
  definition := { stage := ⟨"family-test"⟩, name := "body" }
  bodyScope := ⟨[]⟩
  node := ⟨2⟩
  port := 0

private def familyTestExternalWire : CoreWireRef :=
  { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨9⟩, port := 0 }

private def familyTestAliasSource : ParallelFamilyDerivationSource := {
  familyTestSource with
  outputCount := 0
  outputTypes := []
  elementTemplates := []
  matrixAliasTemplates := [{
    subject := familyTestLocalTemplate
    subjectType := familyTestType
    exactTarget := .wire { value := .concrete familyTestExternalWire, type := familyTestType }
  }]
}

private def familyTestAliasSubstitution : ParallelTemplateSubstitution := {
  derivation := familyTestAliasSource
  actualIndex := ⟨3⟩
  actualIndexExpression := .intConstant 3
  originContext := some {
    arena := { entries := [] }
    selectedLoop := ⟨familyTestLoopSite⟩
    indexSlot := 0
  }
}

private def familyTestRawAliasSource : MatrixInstanceRef := {
  value := .instantiatedTemplate familyTestLocalTemplate
    [.parallelLane familyTestLoopSite ⟨3⟩]
  type := familyTestType
}

example :
    resolveRelationSourceOrigin familyTestSource familyTestSubstitution familyTestRawAliasSource =
      .error .missingAlias := by
  rfl

example :
    instantiateParallelTemplate
      familyTestSubstitution
      { entries := [] }
      (ValueInstanceRef.familyElement (.joint ⟨"joint"⟩ 0 []) ⟨3⟩)
      familyTestFact =
    .ok ({ entries := [] }, .matrix {
      subject := ValueInstanceRef.familyElement (.joint ⟨"joint"⟩ 0 []) ⟨3⟩
      primary := .exact (.wire {
        value := .familyElement (.joint ⟨"joint"⟩ 0 []) ⟨3⟩
        type := familyTestType
      })
      relations := []
      totalNormBound := .constant 8
    }) := by
  simp [instantiateParallelTemplate, ParallelTemplateSubstitution.rewriteArena,
    ParallelTemplateSubstitution.rewriteArenaEntries, ParallelTemplateSubstitution.replaceFact,
    ParallelTemplateSubstitution.rewriteMatrix, ParallelTemplateSubstitution.replaceMatrixInstance,
    ParallelTemplateSubstitution.replaceValue, ParallelTemplateSubstitution.outputSlot?,
    findParallelOutputSlot, ValueFact.ownedTemplateWire?, ParallelTemplateSubstitution.replaceBound,
    ValueFact.retargetOutput, familyTestSubstitution, familyTestSource, familyTestFact]
  rfl

private def familyTestPublicTemplate : TemplateWireRef :=
  { familyTestTemplate with port := 0 }

private def familyTestPrivateTemplate : TemplateWireRef :=
  { familyTestTemplate with port := 1 }

private def familyTestTrapdoorTemplate : ValueFactTemplate := {
  fact := .trapdoor {
    privatePort := .template familyTestPrivateTemplate
    publicPort := .template familyTestPublicTemplate
    publicMatrix := .wire { value := .template familyTestPublicTemplate, type := familyTestType }
  }
  schema := .trapdoor
}

private def familyTestPublicFact : ValueFactTemplate := {
  fact := .matrix {
    subject := .template familyTestPublicTemplate
    primary := .exact (.wire { value := .template familyTestPublicTemplate, type := familyTestType })
    relations := []
    totalNormBound := .constant 8
  }
  schema := .matrix familyTestType .exact [] .unknown
}

private def familyTestTrapdoorSource : ParallelFamilyDerivationSource := {
  family := ⟨"joint"⟩
  loopSite := familyTestLoopSite
  childScope := ⟨["body"]⟩
  definition := "body"
  count := .constant 5
  indexSlot := 0
  indexReference := familyTestIndex
  indexExpression := .loopIndex { site := familyTestLoopSite }
  bindings := []
  modes := []
  argumentRefs := []
  outputCount := 2
  outputTypes := [.matrix familyTestType, .trapdoor familyTestType (.rational 1) (.constant 2)
    (.constant 1) (.constant 1)]
  body := { nodes := [], outputs := [], inputNames := [] }
  seededFacts := []
  analyzedFacts := []
  outputFacts := []
  elementTemplates := [familyTestPublicFact, familyTestTrapdoorTemplate]
}

private def familyTestTrapdoorSubstitution : ParallelTemplateSubstitution := {
  derivation := familyTestTrapdoorSource
  actualIndex := ⟨4⟩
  actualIndexExpression := .intConstant 4
}

/-- Retrieving the private trapdoor lane aliases only that lane.  Its paired public matrix keeps
the same joint-family identity and index instead of being collapsed into the private output. -/
example :
    instantiateParallelTemplate
      familyTestTrapdoorSubstitution
      { entries := [] }
      (.concrete { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨2⟩, port := 0 })
      familyTestTrapdoorTemplate =
    .ok ({ entries := [] }, .trapdoor {
      privatePort := .concrete {
        stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨2⟩, port := 0
      }
      publicPort := .familyElement (.joint ⟨"joint"⟩ 0 []) ⟨4⟩
      publicMatrix := .wire {
        value := .familyElement (.joint ⟨"joint"⟩ 0 []) ⟨4⟩
        type := familyTestType
      }
    }) := by
  simp [instantiateParallelTemplate, ParallelTemplateSubstitution.rewriteArena,
    ParallelTemplateSubstitution.rewriteArenaEntries, ParallelTemplateSubstitution.replaceFact,
    ParallelTemplateSubstitution.rewriteMatrix, ParallelTemplateSubstitution.replaceMatrixInstance,
    ParallelTemplateSubstitution.replaceValue, ParallelTemplateSubstitution.outputSlot?,
    findParallelOutputSlot, ValueFact.ownedTemplateWire?, ValueFact.retargetOutput,
    familyTestTrapdoorSubstitution, familyTestTrapdoorSource, familyTestTrapdoorTemplate,
    familyTestPublicFact, familyTestPublicTemplate, familyTestPrivateTemplate, familyTestTemplate]
  rfl

end Mxx.Certificate
