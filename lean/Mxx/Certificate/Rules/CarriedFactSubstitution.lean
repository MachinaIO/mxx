import Mxx.Certificate.Rules.LoopRecurrence

namespace Mxx.Certificate

/-!
# Analyzer-owned carried-fact substitution

Sequential bodies are analyzed once with `carriedInput` placeholders.  They cannot be interpreted
by the ordinary semantic relations.  This module is the only structural operation that replaces
those placeholders: it resolves a typed path against the complete immutable fact table from the
previous iteration.  It is deliberately an `Option`-valued, fail-closed transformation.  In
particular, a family path is accepted only when the analyzer retained the exact element template
for that aggregate; there is no caller-provided value or fallback schema reconstruction.
-/

/-- Complete analyzer facts available at one sequential iteration boundary.  `facts` and
`schemas` are kept together so path resolution verifies both the concrete fact shape and the
declared carried schema.  Family templates are analyzer products of the exact body/family
analysis, indexed by the frozen aggregate identity. -/
structure CarriedFactSubstitution where
  facts : List ValueFactTemplate
  values : List Mxx.Ir.Value
  familyElementTemplates : List (FamilyAggregateRef × ValueFactTemplate) := []

/-- Construct the immutable substitution only when every complete fact has one corresponding
actual carried value.  This is the sole structural constructor used by trace integration. -/
def CarriedFactSubstitution.build
    (facts : List ValueFactTemplate)
    (values : List Mxx.Ir.Value)
    (familyElementTemplates : List (FamilyAggregateRef × ValueFactTemplate) := []) :
    Option CarriedFactSubstitution :=
  if facts.length = values.length then some { facts, values, familyElementTemplates } else none

theorem CarriedFactSubstitution.build_values
    {facts : List ValueFactTemplate}
    {values : List Mxx.Ir.Value}
    {familyElementTemplates : List (FamilyAggregateRef × ValueFactTemplate)}
    {substitution : CarriedFactSubstitution}
    (built : CarriedFactSubstitution.build facts values familyElementTemplates = some substitution) :
    substitution.values = values := by
  simp only [CarriedFactSubstitution.build] at built
  split at built
  · cases built
    rfl
  · contradiction

private def CarriedFactSubstitution.root? (substitution : CarriedFactSubstitution)
    (slot : Nat) : Option ValueFactTemplate :=
  substitution.facts[slot]?

private def CarriedFactSubstitution.familyElement? (substitution : CarriedFactSubstitution)
    (aggregate : FamilyAggregateRef) : Option ValueFactTemplate :=
  substitution.familyElementTemplates.find? (fun entry ↦ entry.1 = aggregate) |>.map Prod.snd

/-! ## Structural fuel

The substitution engine must terminate even when an invalid template refers to itself.  Fuel is
therefore derived exclusively from the frozen complete template table, rather than being chosen
by a caller.  An acyclic placeholder path consumes one constructor at a time; a cycle exhausts
this finite budget and returns `none`.
-/

mutual
  private def runtimeWeight {type : RuntimeScalarType} : RuntimeExpr type → Nat
    | .intWire _ | .boolWire _ | .intConstant _ | .boolConstant _ | .parameter _ | .loopIndex _ |
        .carriedInput _ => 1
    | .intBinary _ left right | .compare _ left right => 1 + runtimeWeight left + runtimeWeight right
    | .bitExtract value _ | .boolToInt value => 1 + runtimeWeight value
    | .thresholdDecodeBool .. | .extractCoefficient .. => 1
    | .familyElement _ _ _ index => 1 + runtimeWeight index
    | .select _ index _ => 1 + runtimeWeight index

  private def matrixWeight : MatrixExpr → Nat
    | .wire _ | .zero _ | .identity _ | .gadget _ _ | .loopResult .. | .carriedInput .. => 1
    | .add left right | .multiply left right => 1 + matrixWeight left + matrixWeight right
    | .negate value | .scalarMultiply _ value | .rowSlice value _ _ | .columnSlice value _ _ |
        .rowCoefficientEmbed _ _ value | .columnBasisEmbed _ _ value |
        .diagonalCoefficientEmbed _ _ value | .diagonalBasisEmbed _ _ value => 1 + matrixWeight value
    | .rowConcat parts | .columnConcat parts | .diagonalConcat parts =>
        1 + parts.foldl (fun total part ↦ total + matrixWeight part) 0
    | .select index branches =>
        1 + runtimeWeight index + branches.foldl (fun total branch ↦ total + matrixWeight branch) 0
end

private def boundWeight : BoundExpr → Nat
  | .constant _ | .parameter _ | .absolute _ | .recurrenceResult .. | .carriedInput _ => 1
  | .add left right | .multiply left right | .maximum left right | .minimum left right =>
      1 + boundWeight left + boundWeight right
  | .floorDivide value _ => 1 + boundWeight value
  | .matrixProduct _ _ left right => 1 + boundWeight left + boundWeight right

private def intBoundWeight : IntBoundExpr → Nat
  | .integer _ | .carriedInput _ | .recurrenceResult .. => 1
  | .natural value => 1 + boundWeight value
  | .negate value => 1 + intBoundWeight value
  | .add left right | .subtract left right | .multiply left right | .divide left right |
      .minimum left right | .maximum left right => 1 + intBoundWeight left + intBoundWeight right

private def primaryWeight : MatrixPrimaryForm → Nat
  | .exact expression => matrixWeight expression
  | .affine form => 1 + boundWeight form.noiseBound + form.terms.foldl (fun total term ↦
      total + matrixWeight term.coefficient.expression + boundWeight term.coefficient.normBound +
        matrixWeight term.basis) 0

private def factWeight : ValueFact → Nat
  | .matrix fact => 1 + primaryWeight fact.primary + boundWeight fact.totalNormBound
  | .trapdoor _ | .bytes _ | .family _ => 1
  | .integer fact => 1 + runtimeWeight fact.expression + intBoundWeight fact.lower +
      intBoundWeight fact.upper
  | .boolean fact => 1 + runtimeWeight fact.expression

private def templateWeight (template : ValueFactTemplate) : Nat := factWeight template.fact

private def CarriedFactSubstitution.fuel (substitution : CarriedFactSubstitution) : Nat :=
  1 + substitution.facts.foldl (fun total template ↦ total + templateWeight template) 0 +
    substitution.familyElementTemplates.foldl (fun total entry ↦ total + templateWeight entry.2) 0

private def resolveMatrixPathIn
    (substitution : CarriedFactSubstitution) (rootSlot : Nat) :
    MatrixFactPath → ValueFactTemplate → Option MatrixExpr
  | .exactExpression slot, { fact := .matrix fact, schema := .matrix _ .exact _ _ } =>
      if slot = rootSlot then
        match fact.primary with
        | .exact expression => some expression
        | .affine _ => none
      else none
  | .affineCoefficient slot term, { fact := .matrix fact, schema := .matrix _ (.affine _) _ _ } =>
      if slot = rootSlot then
        match fact.primary with
        | .affine form => form.terms[term]?.map (·.coefficient.expression)
        | .exact _ => none
      else none
  | .affineBasis slot term, { fact := .matrix fact, schema := .matrix _ (.affine _) _ _ } =>
      if slot = rootSlot then
        match fact.primary with
        | .affine form => form.terms[term]?.map (·.basis)
        | .exact _ => none
      else none
  | .familyElement slot _ nested, { fact := .family family, schema := .family _ _ } =>
      if slot = rootSlot then
        substitution.familyElement? family.aggregate |>.bind
          (resolveMatrixPathIn substitution rootSlot nested)
      else none
  | _, _ => none

/-- Resolve a matrix component of the previous immutable carried state.  Paths store their root
slot, while the local resolver uses slot zero after selecting that root; this makes a nested
family element unable to switch to another carried slot. -/
def CarriedFactSubstitution.resolveMatrix
    (substitution : CarriedFactSubstitution) (path : MatrixFactPath) : Option MatrixExpr :=
  match path with
  | .exactExpression slot =>
      substitution.root? slot |>.bind (resolveMatrixPathIn substitution slot (.exactExpression slot))
  | .affineCoefficient slot term =>
      substitution.root? slot |>.bind
        (resolveMatrixPathIn substitution slot (.affineCoefficient slot term))
  | .affineBasis slot term =>
      substitution.root? slot |>.bind (resolveMatrixPathIn substitution slot (.affineBasis slot term))
  | .familyElement slot index nested =>
      substitution.root? slot |>.bind
        (resolveMatrixPathIn substitution slot (.familyElement slot index nested))

private def resolveBoundPathIn
    (substitution : CarriedFactSubstitution) (rootSlot : Nat) :
    BoundFactPath → ValueFactTemplate → Option BoundExpr
  | .affineCoefficientBound slot term, { fact := .matrix fact, schema := .matrix _ (.affine _) _ _ } =>
      if slot = rootSlot then
        match fact.primary with
        | .affine form => form.terms[term]?.map (·.coefficient.normBound)
        | .exact _ => none
      else none
  | .affineNoiseBound slot, { fact := .matrix fact, schema := .matrix _ (.affine _) _ _ } =>
      if slot = rootSlot then
        match fact.primary with
        | .affine form => some form.noiseBound
        | .exact _ => none
      else none
  | .matrixTotalBound slot, { fact := .matrix fact, schema := .matrix .. } =>
      if slot = rootSlot then some fact.totalNormBound else none
  | .familyElement slot _ nested, { fact := .family family, schema := .family _ _ } =>
      if slot = rootSlot then
        substitution.familyElement? family.aggregate |>.bind
          (resolveBoundPathIn substitution rootSlot nested)
      else none
  | _, _ => none

def CarriedFactSubstitution.resolveBound
    (substitution : CarriedFactSubstitution) (path : BoundFactPath) : Option BoundExpr :=
  match path with
  | .affineCoefficientBound slot term => substitution.root? slot |>.bind
      (resolveBoundPathIn substitution slot (.affineCoefficientBound slot term))
  | .affineNoiseBound slot => substitution.root? slot |>.bind
      (resolveBoundPathIn substitution slot (.affineNoiseBound slot))
  | .matrixTotalBound slot => substitution.root? slot |>.bind
      (resolveBoundPathIn substitution slot (.matrixTotalBound slot))
  | .familyElement slot index nested => substitution.root? slot |>.bind
      (resolveBoundPathIn substitution slot (.familyElement slot index nested))

private def resolveIntBoundPathIn
    (substitution : CarriedFactSubstitution) (rootSlot : Nat) :
    IntBoundFactPath → ValueFactTemplate → Option IntBoundExpr
  | .lower slot, { fact := .integer fact, schema := .integer } =>
      if slot = rootSlot then some fact.lower else none
  | .upper slot, { fact := .integer fact, schema := .integer } =>
      if slot = rootSlot then some fact.upper else none
  | .familyElement slot _ nested, { fact := .family family, schema := .family _ _ } =>
      if slot = rootSlot then
        substitution.familyElement? family.aggregate |>.bind
          (resolveIntBoundPathIn substitution rootSlot nested)
      else none
  | _, _ => none

def CarriedFactSubstitution.resolveIntBound
    (substitution : CarriedFactSubstitution) (path : IntBoundFactPath) : Option IntBoundExpr :=
  match path with
  | .lower slot => substitution.root? slot |>.bind
      (resolveIntBoundPathIn substitution slot (.lower slot))
  | .upper slot => substitution.root? slot |>.bind
      (resolveIntBoundPathIn substitution slot (.upper slot))
  | .familyElement slot index nested => substitution.root? slot |>.bind
      (resolveIntBoundPathIn substitution slot (.familyElement slot index nested))

private def resolveRuntimePathIn {type : RuntimeScalarType}
    (substitution : CarriedFactSubstitution) (rootSlot : Nat) : RuntimeFactPath type → ValueFactTemplate →
      Option (RuntimeExpr type)
  | .integerValue slot, { fact := .integer fact, schema := .integer } =>
      if slot = rootSlot then some fact.expression else none
  | .booleanValue slot, { fact := .boolean fact, schema := .boolean } =>
      if slot = rootSlot then some fact.expression else none
  | .familyElement slot _ nested, { fact := .family family, schema := .family _ _ } =>
      if slot = rootSlot then
        substitution.familyElement? family.aggregate |>.bind
          (resolveRuntimePathIn substitution rootSlot nested)
      else none
  | _, _ => none

def CarriedFactSubstitution.resolveRuntime {type : RuntimeScalarType}
    (substitution : CarriedFactSubstitution) (path : RuntimeFactPath type) : Option (RuntimeExpr type) :=
  match path with
  | .integerValue slot => substitution.root? slot |>.bind
      (resolveRuntimePathIn substitution slot (.integerValue slot))
  | .booleanValue slot => substitution.root? slot |>.bind
      (resolveRuntimePathIn substitution slot (.booleanValue slot))
  | .familyElement slot index nested => substitution.root? slot |>.bind
      (resolveRuntimePathIn substitution slot (.familyElement slot index nested))

mutual
  def substituteRuntime {type : RuntimeScalarType} (substitution : CarriedFactSubstitution) :
      Nat → RuntimeExpr type → Option (RuntimeExpr type)
    | 0, _ => none
    | fuel + 1, expression => match expression with
    | .intWire wire => some (.intWire wire)
    | .boolWire wire => some (.boolWire wire)
    | .intConstant value => some (.intConstant value)
    | .boolConstant value => some (.boolConstant value)
    | .parameter value => some (.parameter value)
    | .intBinary operation left right =>
        return .intBinary operation (← substituteRuntime substitution fuel left)
          (← substituteRuntime substitution fuel right)
    | .compare operation left right =>
        return .compare operation (← substituteRuntime substitution fuel left)
          (← substituteRuntime substitution fuel right)
    | .bitExtract value position => return .bitExtract (← substituteRuntime substitution fuel value) position
    | .boolToInt value => return .boolToInt (← substituteRuntime substitution fuel value)
    | .thresholdDecodeBool matrix ciphertextModulus plaintextModulus position =>
        some (.thresholdDecodeBool matrix ciphertextModulus plaintextModulus position)
    | .extractCoefficient matrix position => some (.extractCoefficient matrix position)
    | .familyElement elementType aggregate indexRef index =>
        return .familyElement elementType aggregate indexRef (← substituteRuntime substitution fuel index)
    | .select resultType index branches =>
        return .select resultType (← substituteRuntime substitution fuel index) branches
    | .loopIndex loop => some (.loopIndex loop)
    | .carriedInput path => substitution.resolveRuntime path >>= substituteRuntime substitution fuel

  def substituteMatrix (substitution : CarriedFactSubstitution) : Nat → MatrixExpr → Option MatrixExpr
    | 0, _ => none
    | fuel + 1, expression => match expression with
    | .wire reference => some (.wire reference)
    | .zero type => some (.zero type)
    | .identity type => some (.identity type)
    | .gadget type base => some (.gadget type base)
    | .add left right => do
        return .add (← substituteMatrix substitution fuel left) (← substituteMatrix substitution fuel right)
    | .negate value => return .negate (← substituteMatrix substitution fuel value)
    | .multiply left right => do
        return .multiply (← substituteMatrix substitution fuel left) (← substituteMatrix substitution fuel right)
    | .scalarMultiply scalar value => return .scalarMultiply scalar (← substituteMatrix substitution fuel value)
    | .rowSlice value start stop => return .rowSlice (← substituteMatrix substitution fuel value) start stop
    | .rowConcat parts => return .rowConcat (← parts.mapM (substituteMatrix substitution fuel))
    | .columnSlice value start stop => return .columnSlice (← substituteMatrix substitution fuel value) start stop
    | .columnConcat parts => return .columnConcat (← parts.mapM (substituteMatrix substitution fuel))
    | .diagonalConcat parts => return .diagonalConcat (← parts.mapM (substituteMatrix substitution fuel))
    | .rowCoefficientEmbed layout part value => do
        return .rowCoefficientEmbed layout part (← substituteMatrix substitution fuel value)
    | .columnBasisEmbed layout part value => do
        return .columnBasisEmbed layout part (← substituteMatrix substitution fuel value)
    | .diagonalCoefficientEmbed layout part value => do
        return .diagonalCoefficientEmbed layout part (← substituteMatrix substitution fuel value)
    | .diagonalBasisEmbed layout part value => do
        return .diagonalBasisEmbed layout part (← substituteMatrix substitution fuel value)
    | .select index branches => do
        return .select (← substituteRuntime substitution fuel index)
          (← branches.mapM (substituteMatrix substitution fuel))
    | .loopResult type recurrence path => some (.loopResult type recurrence path)
    | .carriedInput _ path => substitution.resolveMatrix path >>= substituteMatrix substitution fuel
end

def substituteBound (substitution : CarriedFactSubstitution) : Nat → BoundExpr → Option BoundExpr
  | 0, _ => none
  | fuel + 1, expression => match expression with
  | .constant value => some (.constant value)
  | .parameter value => some (.parameter value)
  | .add left right => return .add (← substituteBound substitution fuel left) (← substituteBound substitution fuel right)
  | .multiply left right => do
      return .multiply (← substituteBound substitution fuel left) (← substituteBound substitution fuel right)
  | .maximum left right => do
      return .maximum (← substituteBound substitution fuel left) (← substituteBound substitution fuel right)
  | .absolute value => some (.absolute value)
  | .floorDivide value divisor => return .floorDivide (← substituteBound substitution fuel value) divisor
  | .matrixProduct ring inner left right => do
      return .matrixProduct ring inner (← substituteBound substitution fuel left)
        (← substituteBound substitution fuel right)
  | .minimum left right => do
      return .minimum (← substituteBound substitution fuel left) (← substituteBound substitution fuel right)
  | .recurrenceResult recurrence path => some (.recurrenceResult recurrence path)
  | .carriedInput path => substitution.resolveBound path >>= substituteBound substitution fuel

def substituteIntBound (substitution : CarriedFactSubstitution) : Nat → IntBoundExpr → Option IntBoundExpr
  | 0, _ => none
  | fuel + 1, expression => match expression with
  | .integer value => some (.integer value)
  | .natural value => return .natural (← substituteBound substitution fuel value)
  | .negate value => return .negate (← substituteIntBound substitution fuel value)
  | .add left right => do
      return .add (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .subtract left right => do
      return .subtract (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .multiply left right => do
      return .multiply (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .divide left right => do
      return .divide (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .minimum left right => do
      return .minimum (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .maximum left right => do
      return .maximum (← substituteIntBound substitution fuel left) (← substituteIntBound substitution fuel right)
  | .carriedInput path => substitution.resolveIntBound path >>= substituteIntBound substitution fuel
  | .recurrenceResult recurrence path => some (.recurrenceResult recurrence path)

/-- A substitution succeeds only if it removes every analyzer-only placeholder reachable in the
finite budget mechanically derived from its frozen template graph.  Callers do not choose a
budget, and a failed substitution has no ordinary environment interpretation. -/
def CarriedFactSubstitution.instantiateMatrix
    (substitution : CarriedFactSubstitution) (expression : MatrixExpr) : Option MatrixExpr :=
  substituteMatrix substitution substitution.fuel expression

def CarriedFactSubstitution.instantiateBound
    (substitution : CarriedFactSubstitution) (expression : BoundExpr) : Option BoundExpr :=
  substituteBound substitution substitution.fuel expression

def CarriedFactSubstitution.instantiateIntBound
    (substitution : CarriedFactSubstitution) (expression : IntBoundExpr) : Option IntBoundExpr :=
  substituteIntBound substitution substitution.fuel expression

def CarriedFactSubstitution.instantiateRuntime {type : RuntimeScalarType}
    (substitution : CarriedFactSubstitution) (expression : RuntimeExpr type) :
    Option (RuntimeExpr type) :=
  substituteRuntime substitution substitution.fuel expression

/-- Resolve a carried family aggregate from the same immutable fact table used for matrix and
bound paths. A missing family slot or aggregate cycle fails rather than retaining an escaped
placeholder in an otherwise instantiated body fact. -/
private def CarriedFactSubstitution.instantiateAggregate
    (substitution : CarriedFactSubstitution) : Nat → FamilyAggregateRef → Option FamilyAggregateRef
  | 0, _ => none
  | fuel + 1, .carriedInput slot => do
      let template ← substitution.root? slot
      let .family family := template.fact | none
      substitution.instantiateAggregate fuel family.aggregate
  | fuel + 1, .familyElement parent index =>
      return .familyElement (← substitution.instantiateAggregate fuel parent) index
  | _, aggregate => some aggregate

private def CarriedFactSubstitution.instantiateValue
    (substitution : CarriedFactSubstitution) : ValueInstanceRef → Option ValueInstanceRef
  | .familyElement aggregate index =>
      return .familyElement (← substitution.instantiateAggregate substitution.fuel aggregate) index
  | value => some value

private def CarriedFactSubstitution.instantiateMatrixInstance
    (substitution : CarriedFactSubstitution) (reference : MatrixInstanceRef) :
    Option MatrixInstanceRef := do
  return { reference with value := ← substitution.instantiateValue reference.value }

private def CarriedFactSubstitution.instantiateRelation
    (substitution : CarriedFactSubstitution) : MatrixRelation → Option MatrixRelation
  | .preimage subject source target trapdoor => do
      let subject ← substitution.instantiateValue subject
      let source ← substitution.instantiateMatrixInstance source
      let target ← substitution.instantiateMatrixInstance target
      let trapdoor ← substitution.instantiateValue trapdoor
      return .preimage subject source target trapdoor
  | .gadgetDecomposition subject target base digitCount => do
      let subject ← substitution.instantiateValue subject
      let target ← substitution.instantiateMatrixInstance target
      return .gadgetDecomposition subject target base digitCount

private def instantiateBoundedMatrixExpr
    (fuel : Nat) (substitution : CarriedFactSubstitution) (value : BoundedMatrixExpr) :
    Option BoundedMatrixExpr := do
  return {
    expression := ← substituteMatrix substitution fuel value.expression
    normBound := ← substituteBound substitution fuel value.normBound
  }

private def instantiateSignalTerm
    (fuel : Nat) (substitution : CarriedFactSubstitution) (term : SignalTerm) : Option SignalTerm := do
  return {
    coefficient := ← instantiateBoundedMatrixExpr fuel substitution term.coefficient
    basis := ← substituteMatrix substitution fuel term.basis
    mode := term.mode
  }

private def instantiatePrimary
    (fuel : Nat) (substitution : CarriedFactSubstitution) : MatrixPrimaryForm →
    Option MatrixPrimaryForm
  | .exact expression => return .exact (← substituteMatrix substitution fuel expression)
  | .affine form => return .affine {
      terms := ← form.terms.mapM (instantiateSignalTerm fuel substitution)
      noiseBound := ← substituteBound substitution fuel form.noiseBound
    }

/-- Instantiate every placeholder in a complete fact at once, including family aggregates and
the identities embedded in retained relations. -/
def CarriedFactSubstitution.instantiateFact
    (substitution : CarriedFactSubstitution) : ValueFact → Option ValueFact
  | .matrix fact => return .matrix {
      fact with
      subject := ← substitution.instantiateValue fact.subject
      primary := ← instantiatePrimary substitution.fuel substitution fact.primary
      relations := ← fact.relations.mapM substitution.instantiateRelation
      totalNormBound := ← substitution.instantiateBound fact.totalNormBound
    }
  | .trapdoor fact => return .trapdoor {
      fact with
      privatePort := ← substitution.instantiateValue fact.privatePort
      publicPort := ← substitution.instantiateValue fact.publicPort
      publicMatrix := ← substitution.instantiateMatrix fact.publicMatrix
    }
  | .integer fact => return .integer {
      expression := ← substitution.instantiateRuntime fact.expression
      lower := ← substitution.instantiateIntBound fact.lower
      upper := ← substitution.instantiateIntBound fact.upper
    }
  | .boolean fact => return .boolean {
      expression := ← substitution.instantiateRuntime fact.expression
    }
  | .bytes wire => return .bytes (← substitution.instantiateValue wire)
  | .family fact => return .family {
      fact with aggregate := ← substitution.instantiateAggregate substitution.fuel fact.aggregate
    }

/-- Replace every analyzer-only carried placeholder in one scoped body fact.  The executable
wire remains unchanged: only its symbolic explanation is instantiated from the immutable prior
carried state. -/
def CarriedFactSubstitution.instantiateScopedFact
    (substitution : CarriedFactSubstitution) (fact : ScopedWireFact) : Option ScopedWireFact := do
  return { fact with fact := ← substitution.instantiateFact fact.fact }

/-- Semantic boundary between the abstract loop-body analyzer and ordinary fact soundness.
`carriedInput` receives no constructor in the ordinary denotation relations.  Instead, this
judgment requires a successful structural instantiation and then uses the existing
`ScopedWireFact.Holds` relation for the same executable output wire and actual value. -/
def InstantiatedScopedFact.Holds
    (substitution : CarriedFactSubstitution)
    (environment : FactEnvironment)
    (raw : ScopedWireFact)
    (actual : Mxx.Ir.Value) : Prop :=
  ∃ instantiated : ScopedWireFact,
    substitution.instantiateScopedFact raw = some instantiated ∧
    environment.values (.ofCoreWire raw.wire) = some actual ∧
    instantiated.Holds environment

theorem InstantiatedScopedFact.Holds.noCarriedInput
    {substitution : CarriedFactSubstitution}
    {environment : FactEnvironment}
    {raw : ScopedWireFact}
    {actual : Mxx.Ir.Value}
    (_holds : InstantiatedScopedFact.Holds substitution environment raw actual) :
    raw.fact.hasCarriedInput = false ∨
      ∃ instantiated, substitution.instantiateScopedFact raw = some instantiated := by
  right
  obtain ⟨instantiated, instantiates, _, _⟩ := _holds
  exact ⟨instantiated, instantiates⟩

def CarriedFactSubstitution.instantiateTemplate
    (substitution : CarriedFactSubstitution) (template : ValueFactTemplate) :
    Option ValueFactTemplate := do
  return { template with fact := ← substitution.instantiateFact template.fact }

/-- Instantiate the full output tuple of one abstract sequential body from one immutable prior
state.  Every output reads the same `substitution`; this is the structural simultaneous-update
rule and prevents an output later in the tuple from observing an earlier updated output. -/
def CarriedFactSubstitution.instantiateTemplates
    (substitution : CarriedFactSubstitution) (templates : List ValueFactTemplate) :
    Option (List ValueFactTemplate) :=
  templates.mapM substitution.instantiateTemplate

def CarriedFactSubstitution.instantiateBodyOutputs
    (substitution : CarriedFactSubstitution)
    (templates : List ValueFactTemplate)
    (values : List Mxx.Ir.Value)
    (familyElementTemplates : List (FamilyAggregateRef × ValueFactTemplate) := []) :
    Option CarriedFactSubstitution := do
  let facts ← substitution.instantiateTemplates templates
  CarriedFactSubstitution.build facts values familyElementTemplates

private def substitutionFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 1
  columns := .constant 1

private def substitutionFixture : CarriedFactSubstitution where
  facts := [{
    fact := .matrix {
      subject := .protocolInput ⟨"carried-substitution-fixture"⟩
      primary := .exact (.zero substitutionFixtureType)
      relations := []
      totalNormBound := .constant 9
    }
    schema := .matrix substitutionFixtureType .exact [] .unknown
  }]
  values := [.matrix {
    modulus := 17
    ringDimension := 4
    rows := 1
    columns := 1
    coefficients := [0]
  }]

/-- A typed exact path eliminates the placeholder without consulting a value environment. -/
example : substituteMatrix substitutionFixture 2
    (.carriedInput substitutionFixtureType (.exactExpression 0)) =
    some (.zero substitutionFixtureType) := by
  rfl

/-- Nested family paths retain their selected root slot; the exact element template is recovered
from the aggregate identity rather than from a same-shaped fact in another carried slot. -/
private def familySubstitutionFixture : CarriedFactSubstitution where
  facts := [{
    fact := .family {
      aggregate := .carriedInput 0
      count := .constant 1
      elementSchema := .matrix substitutionFixtureType .exact [] .unknown
    }
    schema := .family (.constant 1) (.matrix substitutionFixtureType .exact [] .unknown)
  }]
  values := [.family []]
  familyElementTemplates := [(.carriedInput 0, {
    fact := .matrix {
      subject := .protocolInput ⟨"family-carried-substitution-fixture"⟩
      primary := .exact (.zero substitutionFixtureType)
      relations := []
      totalNormBound := .constant 0
    }
    schema := .matrix substitutionFixtureType .exact [] .unknown
  })]

private def resolvedFamilyAggregate : FamilyAggregateRef := .joint ⟨"resolved-family"⟩ 0 []

private def resolvedFamilySubstitutionFixture : CarriedFactSubstitution where
  facts := [{
    fact := .family {
      aggregate := resolvedFamilyAggregate
      count := .constant 1
      elementSchema := .matrix substitutionFixtureType .exact [] .unknown
    }
    schema := .family (.constant 1) (.matrix substitutionFixtureType .exact [] .unknown)
  }]
  values := [.family []]

example : CarriedFactSubstitution.build substitutionFixture.facts [] = none := by
  rfl

example : substituteMatrix familySubstitutionFixture 3
    (.carriedInput substitutionFixtureType (.familyElement 0 ⟨0⟩ (.exactExpression 0))) =
    some (.zero substitutionFixtureType) := by
  rfl

/-- A family carried placeholder is eliminated from the full fact, not only from an element path. -/
example : resolvedFamilySubstitutionFixture.instantiateFact (.family {
    aggregate := .carriedInput 0
    count := .constant 1
    elementSchema := .matrix substitutionFixtureType .exact [] .unknown
  }) = some (.family {
    aggregate := resolvedFamilyAggregate
    count := .constant 1
    elementSchema := .matrix substitutionFixtureType .exact [] .unknown
  }) := by
  rfl

/-- A path cannot cross from a family element back to another carried root. -/
example : substituteMatrix substitutionFixture 2
    (.carriedInput substitutionFixtureType (.familyElement 0 ⟨0⟩ (.exactExpression 1))) = none := by
  rfl

/-- Replacing a placeholder by itself consumes fuel and fails closed rather than creating a
meaning for `carriedInput` in the ordinary semantic environment. -/
private def selfReferentialFixture : CarriedFactSubstitution where
  facts := [{
    fact := .matrix {
      subject := .protocolInput ⟨"self-referential-carried-fixture"⟩
      primary := .exact (.carriedInput substitutionFixtureType (.exactExpression 0))
      relations := []
      totalNormBound := .constant 0
    }
    schema := .matrix substitutionFixtureType .exact [] .unknown
  }]
  values := [.matrix {
    modulus := 17
    ringDimension := 4
    rows := 1
    columns := 1
    coefficients := [0]
  }]

example : substituteMatrix selfReferentialFixture 3
    (.carriedInput substitutionFixtureType (.exactExpression 0)) = none := by
  rfl

end Mxx.Certificate
