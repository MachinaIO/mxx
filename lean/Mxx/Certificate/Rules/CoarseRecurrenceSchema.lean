import Mxx.Certificate.RecurrenceSchema

namespace Mxx.Certificate

inductive CoarseRecurrenceSchemaError where
  | carriedTrapdoor
  | carriedMatrixRelation
  | factSchemaMismatch
  deriving BEq, DecidableEq, Repr

/-- Forget exact/affine syntax, DAG shape, and affine term positions while retaining every piece
of type information required to carry a value through a sequential loop. Matrix relations are
rejected recursively because the initial recurrence registry has no soundness rule for carrying
them across iterations. -/
def ValueFactSchema.toCarriedValueSchema :
    ValueFactSchema → Except CoarseRecurrenceSchemaError CarriedValueSchema
  | .matrix matrixType _ relations representation =>
      if relations.isEmpty then .ok (.matrix matrixType representation)
      else .error .carriedMatrixRelation
  | .trapdoor => .error .carriedTrapdoor
  | .integer => .ok .integer
  | .boolean => .ok .boolean
  | .bytes => .ok .bytes
  | .family count element => do
      return .family count (← element.toCarriedValueSchema)

/-- Check an analyzer-produced fact against its declared template only at the stable carried-value
schema boundary. In particular, matrix primary syntax and affine term count are intentionally not
compared. -/
def ValueFact.coarseSchemaAgainst
    (fact : ValueFact)
    (declared : ValueFactSchema) :
    Except CoarseRecurrenceSchemaError CarriedValueSchema := do
  let carried ← declared.toCarriedValueSchema
  match fact with
  /- A bare `ValueFact` does not carry its owning matrix type. Accepting the declared type here
  would make that type self-reported. Top-level matrix carry stays fail-closed until conversion
  is performed from `ScopedWireFact`; family elements retain their typed recursive schema. -/
  | .matrix _ => .error .factSchemaMismatch
  | .trapdoor _ =>
      match declared with
      | .trapdoor => .error .carriedTrapdoor
      | _ => .error .factSchemaMismatch
  | .integer _ => if declared == .integer then .ok carried else .error .factSchemaMismatch
  | .boolean _ => if declared == .boolean then .ok carried else .error .factSchemaMismatch
  | .bytes _ => if declared == .bytes then .ok carried else .error .factSchemaMismatch
  | .family familyFact =>
      match declared with
      | .family count element => do
          let actual ←
            (ValueFactSchema.family familyFact.count
              familyFact.elementSchema).toCarriedValueSchema
          let expected ← (ValueFactSchema.family count element).toCarriedValueSchema
          if actual == expected then .ok carried else .error .factSchemaMismatch
      | _ => .error .factSchemaMismatch

def ValueFactTemplate.toCarriedValueSchema
    (template : ValueFactTemplate) :
    Except CoarseRecurrenceSchemaError CarriedValueSchema :=
  template.fact.coarseSchemaAgainst template.schema

/-- Coarse loop validation deliberately ignores exact-vs-affine representation and affine term
count. Any retained type, count, coefficient-representation, or value-kind difference still
rejects. -/
def ValueFactTemplate.sameCarriedValueSchema
    (left right : ValueFactTemplate) : Except CoarseRecurrenceSchemaError Bool := do
  return (← left.toCarriedValueSchema) == (← right.toCarriedValueSchema)

private def coarseFixtureType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 3

private def coarseFixtureTerm : SignalTermSchema where
  coefficientType := {
    coarseFixtureType with columns := .constant 2
  }
  basisType := coarseFixtureType
  mode := .ordinaryMatrixProduct

private def exactFamilyElementSchema : ValueFactSchema :=
  .matrix coarseFixtureType .exact [] .unknown

private def fiveTermAffineFamilyElementSchema : ValueFactSchema :=
  .matrix coarseFixtureType (.affine (List.replicate 5 coarseFixtureTerm)) [] .unknown

private def exactFamilyTemplate : ValueFactTemplate := {
  fact := .family {
    aggregate := .joint ⟨"coarse-exact-family"⟩ 0 []
    count := .constant 8
    elementSchema := exactFamilyElementSchema
  }
  schema := .family (.constant 8) exactFamilyElementSchema
}

private def fiveTermAffineFamilyTemplate : ValueFactTemplate := {
  fact := .family {
    aggregate := .joint ⟨"coarse-affine-family"⟩ 0 []
    count := .constant 8
    elementSchema := fiveTermAffineFamilyElementSchema
  }
  schema := .family (.constant 8) fiveTermAffineFamilyElementSchema
}

private def relatedMatrixTemplate : ValueFactTemplate := {
  fact := .matrix {
    subject := .protocolInput ⟨"related-matrix"⟩
    primary := .exact (.zero coarseFixtureType)
    relations := [.preimage
      (.protocolInput ⟨"preimage"⟩)
      { value := .protocolInput ⟨"source"⟩, type := coarseFixtureType }
      { value := .protocolInput ⟨"target"⟩, type := coarseFixtureType }
      (.protocolInput ⟨"trapdoor"⟩)]
    totalNormBound := .constant 1
  }
  schema := .matrix coarseFixtureType .exact [] .unknown
}

example : exactFamilyTemplate.toCarriedValueSchema = .ok
    (.family (.constant 8) (.matrix coarseFixtureType .unknown)) := by
  rfl

example : fiveTermAffineFamilyTemplate.toCarriedValueSchema = .ok
    (.family (.constant 8) (.matrix coarseFixtureType .unknown)) := by
  rfl

example : exactFamilyTemplate.sameCarriedValueSchema fiveTermAffineFamilyTemplate = .ok true := by
  rfl

example : relatedMatrixTemplate.toCarriedValueSchema = .error .factSchemaMismatch := by
  rfl

example :
    ((ValueFactSchema.matrix coarseFixtureType .exact [.preimage] .unknown)
      |>.toCarriedValueSchema) = .error .carriedMatrixRelation := by
  rfl

example :
    ((ValueFactSchema.family (.constant 8)
      (.matrix coarseFixtureType (.affine []) [.gadgetDecomposition] .unknown))
      |>.toCarriedValueSchema) = .error .carriedMatrixRelation := by
  rfl

end Mxx.Certificate
