import Mxx.Certificate.Rules.CoarseRecurrenceSchema

namespace Mxx.Certificate

/-- A symbolic body output whose type is fixed by the coarse carried-value schema. -/
inductive SymbolicCarriedOutput : CarriedValueSchema → Type where
  | matrix {matrixType representation}
      (form : SymbolicMatrixFormRef) :
      SymbolicCarriedOutput (.matrix matrixType representation)
  | integer (expression : RuntimeExprRef .integer) : SymbolicCarriedOutput .integer
  | boolean (expression : RuntimeExprRef .boolean) : SymbolicCarriedOutput .boolean
  | bytes (value : ValueInstanceRef) : SymbolicCarriedOutput .bytes
  | family {count element}
      (aggregate : FamilyAggregateRef) : SymbolicCarriedOutput (.family count element)

/-- Schema-indexed output vector. It is impossible to put a matrix reference into an integer
slot or silently change a family's coarse element schema. -/
inductive SymbolicCarriedOutputVector : List CarriedValueSchema → Type where
  | nil : SymbolicCarriedOutputVector []
  | cons {schema schemas}
      (head : SymbolicCarriedOutput schema)
      (tail : SymbolicCarriedOutputVector schemas) :
      SymbolicCarriedOutputVector (schema :: schemas)

inductive SymbolicRecurrenceError where
  | initialSchema (slot : Nat) (error : CoarseRecurrenceSchemaError)
  | bodySchema (slot : Nat) (error : CoarseRecurrenceSchemaError)
  | initialBodyMismatch (slot : Nat)
  | arityMismatch
  | suppliedSchemaMismatch
  deriving BEq, DecidableEq, Repr

private def validateRecurrenceSchemaLists :
    Nat → List ValueFact → List ValueFactTemplate →
      Except SymbolicRecurrenceError (List CarriedValueSchema)
  | _, [], [] => .ok []
  | slot, initial :: initials, body :: bodies => do
      let initialSchema ← initial.coarseSchemaAgainst body.schema
        |>.mapError (.initialSchema slot)
      let bodySchema ← body.toCarriedValueSchema |>.mapError (.bodySchema slot)
      if initialSchema != bodySchema then throw (.initialBodyMismatch slot)
      return initialSchema :: (← validateRecurrenceSchemaLists (slot + 1) initials bodies)
  | _, _, _ => .error .arityMismatch

/-- Derive the stable, term-count-independent carried schema from the actual initial facts and the
analyzer-produced one-step body templates. Both sides pass through the relation-rejecting coarse
conversion. -/
def FactRecurrence.validateCoarseCarriedSchemas
    (recurrence : FactRecurrence) :
    Except SymbolicRecurrenceError (List CarriedValueSchema) :=
  validateRecurrenceSchemaLists 0 recurrence.initial.toList recurrence.bodyOutputs.toList

/-- Analyzer-owned closed recurrence transfer. The symbolic output vector and the numeric bound
transition are indexed by the same coarse schema, so every simultaneous update has one fixed
type. The validation equality records that this schema was derived from the actual recurrence,
not supplied by a protocol certificate. -/
structure SymbolicRecurrenceTransfer where
  identity : FactRecurrenceInstanceRef
  source : FactRecurrence
  carriedSchemas : List CarriedValueSchema
  bodyOutputs : SymbolicCarriedOutputVector carriedSchemas
  boundTransition : CarriedBoundTransitionVector
    (carriedSchemas.map CarriedValueSchema.boundSchema)
    (carriedSchemas.map CarriedValueSchema.boundSchema)
  schemaValidation : source.validateCoarseCarriedSchemas = .ok carriedSchemas

/-- The only public constructor checks initial/body coarse equality and relation-freedom before
packaging already type-indexed symbolic outputs and bound transitions. -/
def SymbolicRecurrenceTransfer.build
    {schemas : List CarriedValueSchema}
    (identity : FactRecurrenceInstanceRef)
    (source : FactRecurrence)
    (bodyOutputs : SymbolicCarriedOutputVector schemas)
    (boundTransition : CarriedBoundTransitionVector
      (schemas.map CarriedValueSchema.boundSchema)
      (schemas.map CarriedValueSchema.boundSchema)) :
    Except SymbolicRecurrenceError SymbolicRecurrenceTransfer :=
  match validation : source.validateCoarseCarriedSchemas with
  | .error error => .error error
  | .ok actual =>
      if same : actual = schemas then
        .ok {
          identity
          source
          carriedSchemas := schemas
          bodyOutputs
          boundTransition
          schemaValidation := by simpa [same] using validation
        }
      else .error .suppliedSchemaMismatch

end Mxx.Certificate
