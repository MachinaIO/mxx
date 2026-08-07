import Mxx.Certificate.Facts

namespace Mxx.Certificate

/-! Dependent numeric state produced only by checked symbolic recurrence evaluation. -/

inductive CarriedBoundSchema where
  | matrixSummary
  | integerInterval
  | boolean
  | bytes
  | family (count : IntExpr) (element : CarriedBoundSchema)
  deriving BEq, DecidableEq

inductive EvaluatedCarriedBoundState : CarriedBoundSchema → Type where
  | matrix (signal : Bool) (coefficientL1 noise total : Nat) :
      EvaluatedCarriedBoundState .matrixSummary
  | integer (lower upper : Int) : EvaluatedCarriedBoundState .integerInterval
  | boolean : EvaluatedCarriedBoundState .boolean
  | bytes : EvaluatedCarriedBoundState .bytes
  | familyEnvelope {count : IntExpr} {elementSchema : CarriedBoundSchema}
      (element : EvaluatedCarriedBoundState elementSchema) :
      EvaluatedCarriedBoundState (.family count elementSchema)

inductive CarriedBoundStateVector : List CarriedBoundSchema → Type where
  | nil : CarriedBoundStateVector []
  | cons {schema : CarriedBoundSchema} {schemas : List CarriedBoundSchema}
      (head : EvaluatedCarriedBoundState schema)
      (tail : CarriedBoundStateVector schemas) :
      CarriedBoundStateVector (schema :: schemas)

/-- Existential package retaining the exact dependent schema of one checked recurrence result. -/
structure CheckedSymbolicRecurrenceState where
  identity : SequentialRecurrenceInstanceRef
  schemas : List CarriedBoundSchema
  values : CarriedBoundStateVector schemas

structure CheckedSymbolicRecurrenceStateTable where
  entries : List CheckedSymbolicRecurrenceState := []

end Mxx.Certificate
