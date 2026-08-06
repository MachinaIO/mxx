import Mxx.Certificate.Rules.BggThreeTraceEndpoint
import Mxx.Certificate.Rules.RequirementAcceptance

namespace Mxx.Certificate

/-!
# Diamond BGG endpoint normalization

The first endpoint transport is the accepted requirement's exact selection from the final
Boolean recurrence family.  This module extracts that selection from the kernel-checked scalar
denotation.  It deliberately stops before identifying the recurrence aggregate with a
`BggThreeTraceQuotientDerivation` final list: that identification must be derived from the actual
recurrence execution, not supplied as an equality by an endpoint certificate.
-/

/-- The exact runtime lookup selected by one matcher-owned recurrence Boolean.  Its aggregate,
index expression, and arena reference are fixed by `selected`; only their runtime denotations are
existential witnesses. -/
def CheckedSelectedRecurrenceBoolean.TrueLookup
    (environment : FactEnvironment)
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression) : Prop :=
  match selected with
  | .familyElement recurrence path slot indexReference index =>
      ∃ indexValue,
        environment.expressionArena.lookupInteger indexReference = some index ∧
        RuntimeIntExpr.Denotes environment index indexValue ∧
        environment.values
          (.familyElement (.recurrenceResult recurrence path slot) indexReference) =
            some (.boolean true)

/-- Invert exact Boolean denotation at the matcher-owned recurrence selection. -/
theorem CheckedSelectedRecurrenceBoolean.trueLookup_of_denotes
    {environment : FactEnvironment}
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression)
    (denotes : RuntimeBoolExpr.Denotes environment expression true) :
    selected.TrueLookup environment := by
  cases selected with
  | familyElement recurrence path slot indexReference index =>
      cases denotes with
      | familyElement arenaLookup indexDenotes lookup =>
          exact ⟨_, arenaLookup, indexDenotes, lookup⟩

/-- Bind one exact element of the actual final pure-recurrence family at the identity selected by
the closed requirement matcher.  The resulting environment is a derived semantic view of the
trace; the family value itself is not supplied independently. -/
def CheckedSelectedRecurrenceBoolean.bindActualFinalElement
    {analysis : AnalysisResult}
    {stage : StageId}
    {execution : PureProgramExecution}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundPureSequentialRecurrence analysis stage execution recurrenceInstance)
    (environment : FactEnvironment)
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression)
    (index : Nat)
    (element : evidence.FinalFamilyElementAt selected.slot index) : FactEnvironment :=
  match selected with
  | .familyElement recurrence path slot indexReference _ =>
      environment.bind
        (.familyElement (.recurrenceResult recurrence path slot) indexReference) element.value

/-- Exact denotation of the matcher-owned selection forces the corresponding element of the
actual final recurrence family to be Boolean `true`.  The conclusion is derived from
`FactEnvironment.bind_same`; there is no `booleanFound` premise. -/
theorem CheckedSelectedRecurrenceBoolean.actualFinalElement_true
    {analysis : AnalysisResult}
    {stage : StageId}
    {execution : PureProgramExecution}
    {recurrenceInstance : SequentialRecurrenceInstanceRef}
    (evidence : TraceBoundPureSequentialRecurrence analysis stage execution recurrenceInstance)
    (environment : FactEnvironment)
    {expression : RuntimeExpr .boolean}
    (selected : CheckedSelectedRecurrenceBoolean expression)
    (identityMatches : selected.instance = recurrenceInstance)
    (indexValue : Int)
    (element : evidence.FinalFamilyElementAt selected.slot indexValue.toNat)
    (outputDenotes : RuntimeBoolExpr.Denotes
      (selected.bindActualFinalElement evidence environment indexValue.toNat element)
      expression true) :
    element.value = .boolean true := by
  cases identityMatches
  cases selected with
  | familyElement recurrence path slot indexReference index =>
      cases outputDenotes with
      | familyElement arenaLookup indexDenotes lookup =>
          have actualLookup := FactEnvironment.bind_same environment
            (.familyElement (.recurrenceResult recurrence path slot) indexReference)
            element.value
          exact Option.some.inj (actualLookup.symm.trans lookup)

end Mxx.Certificate
