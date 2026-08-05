import Mxx.Certificate.Semantics
import Mxx.Certificate.Typing

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

private def MatrixRelation.retargetSubject
    (subject : ValueInstanceRef) : MatrixRelation → MatrixRelation
  | .preimage _ source target trapdoor => .preimage subject source target trapdoor
  | .gadgetDecomposition _ target base digitCount =>
      .gadgetDecomposition subject target base digitCount

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

/-- Canonical naming convention for a sequential-loop recurrence. -/
def sequentialFactRecurrenceRef (site : CoreNodeRef) : FactRecurrenceRef := ⟨
  "sequential:" ++ encodeIdentityPart site.stage.name ++
    String.join (site.scope.path.map (fun part ↦ ":" ++ encodeIdentityPart part)) ++
    s!":node:{site.node.value}"
⟩

def instantiateParallelTemplate
    (loopSite : CoreNodeRef)
    (index : RuntimeExprRef .integer)
    (joint : JointFamilyId)
    (bodyOutputs : List (TemplateWireRef × Nat))
    (output : ValueInstanceRef)
    (template : ValueFactTemplate) : ValueFact :=
  let frame := InstanceFrame.parallelLane loopSite index
  let aliasOutput : ValueInstanceRef → ValueInstanceRef
    | .template wire =>
        match bodyOutputs.find? (fun entry ↦ entry.1 = wire) with
        | some (_, slot) => .familyElement (.joint joint slot []) index
        | none => .instantiatedTemplate wire []
    | reference => reference
  ((template.fact.mapInstances aliasOutput).appendInstancePath [frame]).retargetOutput output

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

example :
    instantiateParallelTemplate
      { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨1⟩ }
      ⟨3⟩ ⟨"joint"⟩ [] (ValueInstanceRef.familyElement (.joint ⟨"joint"⟩ 0 []) ⟨3⟩)
      familyTestFact =
    .matrix {
      subject := ValueInstanceRef.familyElement (.joint ⟨"joint"⟩ 0 []) ⟨3⟩
      primary := .exact (.wire {
        value := .instantiatedTemplate familyTestTemplate
          [.parallelLane { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨1⟩ } ⟨3⟩]
        type := familyTestType
      })
      relations := []
      totalNormBound := .constant 8
    } := by
  simp [instantiateParallelTemplate, familyTestFact, ValueFact.mapInstances,
    ValueFact.appendInstancePath, ValueInstanceRef.appendInstancePath,
    MatrixPrimaryForm.appendInstancePath, MatrixExpr.appendInstancePath,
    BoundExpr.appendInstancePath, ValueFact.retargetOutput,
    MatrixPrimaryForm.mapInstances, MatrixExpr.mapInstances, mapMatrixInstance]

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

/-- Retrieving the private trapdoor lane aliases only that lane.  Its paired public matrix keeps
the same joint-family identity and index instead of being collapsed into the private output. -/
example :
    instantiateParallelTemplate
      { stage := ⟨"family-test"⟩, scope := ⟨["nested"]⟩, node := ⟨1⟩ }
      ⟨4⟩ ⟨"joint"⟩ [(familyTestPublicTemplate, 0), (familyTestPrivateTemplate, 1)]
      (.concrete { stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨2⟩, port := 0 })
      familyTestTrapdoorTemplate =
    .trapdoor {
      privatePort := .concrete {
        stage := ⟨"family-test"⟩, scope := ⟨[]⟩, node := ⟨2⟩, port := 0
      }
      publicPort := .familyElement (.joint ⟨"joint"⟩ 0
        [.parallelLane { stage := ⟨"family-test"⟩, scope := ⟨["nested"]⟩, node := ⟨1⟩ } ⟨4⟩]) ⟨4⟩
      publicMatrix := .wire {
        value := .familyElement (.joint ⟨"joint"⟩ 0
          [.parallelLane { stage := ⟨"family-test"⟩, scope := ⟨["nested"]⟩, node := ⟨1⟩ } ⟨4⟩]) ⟨4⟩
        type := familyTestType
      }
    } := by
  simp [instantiateParallelTemplate, familyTestTrapdoorTemplate, ValueFact.mapInstances,
    ValueFact.appendInstancePath, ValueInstanceRef.appendInstancePath,
    MatrixExpr.appendInstancePath, ValueFact.retargetOutput,
    MatrixExpr.mapInstances, mapMatrixInstance, FamilyAggregateRef.appendPath,
    familyTestPublicTemplate, familyTestPrivateTemplate, familyTestTemplate]

end Mxx.Certificate
