import Mxx.Certificate.SymbolicFormSemantics

namespace Mxx.Certificate

/-! # Per-term symbolic matrix evaluation

This module records a proved signal/noise interpretation without collapsing all signal terms into
one matrix.  Carrier identity, coefficient identity, and coefficient norm therefore remain
available to endpoint checks.  The records are proof-only semantic objects; they are not part of
the executable IR or the Rust certificate format.
-/

/-- The exact selector used to turn a protocol Boolean input into a zero-or-identity matrix. -/
def protocolBooleanSelectorExpression
    (input : ProtocolInputId)
    (selectorType : MatrixTypeExpr) : MatrixExpr :=
  .select
    (.boolToInt (.boolWire (.protocolInput input)))
    [.zero selectorType, .identity selectorType]

/-- Closed coefficient provenance retained by semantic evaluation.  The Boolean-selector case is
deliberately restricted to a stable protocol-input identity and the exact zero/identity syntax.
Arbitrary names, Boolean expressions, and caller-provided provenance are not accepted. -/
inductive SignalCoefficientIdentity where
  | matrix (expression : MatrixExpr)
  | protocolBooleanSelector (input : ProtocolInputId) (selectorType : MatrixTypeExpr)

def SignalCoefficientIdentity.Matches
    (identity : SignalCoefficientIdentity)
    (expression : MatrixExpr) : Prop :=
  match identity with
  | .matrix expected => expression = expected
  | .protocolBooleanSelector input selectorType =>
      expression = protocolBooleanSelectorExpression input selectorType

/-- One evaluated signal term.  Its carrier remains the exact `SignalTerm.basis`; its coefficient
is not merged with coefficients of other carriers. -/
structure EvaluatedSignalTerm (environment : FactEnvironment) where
  symbolic : SignalTerm
  coefficientIdentity : SignalCoefficientIdentity
  coefficientBound : Nat
  carrierValue : Mxx.Matrix
  termValue : Mxx.Matrix
  identityMatches : coefficientIdentity.Matches symbolic.coefficient.expression
  coefficientBoundEvaluates : symbolic.coefficient.normBound.evaluate environment.parameters =
    .ok coefficientBound
  carrierDenotes : MatrixExpr.Denotes environment symbolic.basis carrierValue
  termDenotes : SignalTerm.Denotes environment symbolic termValue

/-- Ordinary term denotations determine proof-only evaluated terms.  This is an existence theorem
rather than a data-producing function because Lean does not eliminate semantic proofs into data. -/
theorem evaluatedSignalTerms_exists
    {environment : FactEnvironment}
    {terms : List SignalTerm}
    {values : List Mxx.Matrix}
    (denotes : List.Forall₂ (SignalTerm.Denotes environment) terms values) :
    ∃ evaluated : List (EvaluatedSignalTerm environment),
      evaluated.map (fun term => term.symbolic) = terms ∧
      evaluated.map (fun term => term.termValue) = values := by
  induction denotes with
  | nil => exact ⟨[], rfl, rfl⟩
  | cons head tail induction =>
      obtain ⟨evaluatedTail, symbolicTail, valueTail⟩ := induction
      cases head with
      | identityCoefficient basisDenotes =>
          rename_i identityType basis mode
          let symbolic : SignalTerm := {
            coefficient := { expression := .identity identityType, normBound := .constant 1 }
            basis
            mode
          }
          let evaluatedHead : EvaluatedSignalTerm environment := {
            symbolic
            coefficientIdentity := .matrix symbolic.coefficient.expression
            coefficientBound := 1
            carrierValue := _
            termValue := _
            identityMatches := rfl
            coefficientBoundEvaluates := rfl
            carrierDenotes := basisDenotes
            termDenotes := .identityCoefficient basisDenotes
          }
          exact ⟨evaluatedHead :: evaluatedTail, by simp [evaluatedHead, symbolic, symbolicTail],
            by simp [evaluatedHead, valueTail]⟩
      | product coefficientHolds basisDenotes =>
          rename_i coefficient basis mode coefficientValue basisValue
          obtain ⟨coefficientDenotes, bound, boundEvaluates, coefficientNorm⟩ := coefficientHolds
          let symbolic : SignalTerm := { coefficient, basis, mode }
          let evaluatedHead : EvaluatedSignalTerm environment := {
            symbolic
            coefficientIdentity := .matrix symbolic.coefficient.expression
            coefficientBound := bound
            carrierValue := _
            termValue := _
            identityMatches := rfl
            coefficientBoundEvaluates := boundEvaluates
            carrierDenotes := basisDenotes
            termDenotes := .product
              ⟨coefficientDenotes, bound, boundEvaluates, coefficientNorm⟩ basisDenotes
          }
          exact ⟨evaluatedHead :: evaluatedTail, by simp [evaluatedHead, symbolic, symbolicTail],
            by simp [evaluatedHead, valueTail]⟩

/-- A proved interpretation of one symbolic-form reference.  `terms` is intentionally retained as
a list of independently identified carrier terms.  Only their modular sum participates in the
value equation; no aggregate signal matrix is stored or used for coefficient bounds. -/
structure SymbolicMatrixEvaluation
    (environment : FactEnvironment)
    (arena : SymbolicMatrixFormArena)
    (reference : SymbolicMatrixFormRef) where
  value : Mxx.Matrix
  terms : List (EvaluatedSignalTerm environment)
  noise : Mxx.Matrix
  denotes : SymbolicMatrixFormArena.Denotes environment arena reference value
  valueEquation : Mxx.MatrixModEq value
    (terms.foldr (fun term rest ↦ Mxx.matrixAdd term.termValue rest) noise)

/-- Signal presence belongs to the symbolic evaluation, not to the numeric bound witness. -/
def SymbolicMatrixEvaluation.signalPresence
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference) : SignalPresence :=
  if evaluation.terms.isEmpty then .none else .present

/-- Certified coefficient mass is summed term by term before any carrier cancellation.  Each
summand is the analyzer-preserved hard bound of that exact coefficient expression. -/
def SymbolicMatrixEvaluation.coefficientL1
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
  (evaluation : SymbolicMatrixEvaluation environment arena reference) : Nat :=
  evaluation.terms.foldl (fun total term ↦ total + term.coefficientBound) 0

/-- The retained term/noise decomposition must be the decomposition of the referenced form, not
an unrelated equation for the same value.  This closes the gap between form denotation and the
per-term data consumed by endpoint and bound proofs. -/
inductive SymbolicMatrixEvaluation.MatchesForm
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference) : Prop where
  | signalAtom {matrixType expression term}
      (lookup : arena.lookup reference = some { matrixType, form := .signalAtom expression })
      (terms : evaluation.terms = [term])
      (basis : ∃ carrier, environment.expressionArena.lookupMatrix expression = some carrier ∧
        term.symbolic.basis = carrier)
      (noise : Mxx.maxCenteredCoefficientNorm evaluation.noise = 0) :
      evaluation.MatchesForm
  | boundedAtom {matrixType expression bound}
      (lookup : arena.lookup reference = some {
        matrixType, form := .boundedAtom expression bound
      })
      (terms : evaluation.terms = [])
      (noise : evaluation.noise = evaluation.value) :
      evaluation.MatchesForm
  | boundedAffineLeaf {matrixType form totalBound}
      (lookup : arena.lookup reference = some {
        matrixType, form := .boundedAffineLeaf form totalBound
      })
      (terms : evaluation.terms.map (fun term => term.symbolic) = form.terms)
      (noiseBound : ∃ bound,
        form.noiseBound.evaluate environment.parameters = .ok bound ∧
        Mxx.maxCenteredCoefficientNorm evaluation.noise ≤ bound) :
      evaluation.MatchesForm

/-- Numeric meaning of a hard-bound summary. Every bound is evaluated through the checked
schema-indexed recurrence states. A signal-free summary must have no signal terms; signal
presence itself stays
in the symbolic evaluation because a signal-capable dynamic selection may choose a bounded
branch at runtime. -/
def MatrixBoundSummary.Holds
    (environment : FactEnvironment)
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (summary : MatrixBoundSummary) : Prop :=
  ∃ coefficientL1 noise total,
    summary.coefficientL1Bound.evaluateWithSymbolicRecurrences
        environment.parameters environment.recurrenceStates = .ok coefficientL1 ∧
    summary.noiseBound.evaluateWithSymbolicRecurrences
        environment.parameters environment.recurrenceStates = .ok noise ∧
    summary.totalBound.evaluateWithSymbolicRecurrences
        environment.parameters environment.recurrenceStates = .ok total ∧
    evaluation.coefficientL1 ≤ coefficientL1 ∧
    Mxx.maxCenteredCoefficientNorm evaluation.noise ≤ noise ∧
    Mxx.maxCenteredCoefficientNorm evaluation.value ≤ total ∧
    (summary.signal = .none → evaluation.terms = [])

/-- A symbolic matrix fact is meaningful only together with a per-term evaluation and sound hard
bounds for that same evaluation. -/
def MatrixSymbolicFact.Holds
    (environment : FactEnvironment)
    (arena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    (fact : MatrixSymbolicFact)
    (value : Mxx.Matrix) : Prop :=
  environment.values fact.subject = some (.matrix value) ∧
    (∃ exactExpression,
      environment.expressionArena.lookupMatrix fact.exactValue = some exactExpression ∧
      MatrixExpr.Denotes environment exactExpression value) ∧
    (∃ evaluation : SymbolicMatrixEvaluation environment arena fact.decomposition,
      evaluation.value = value ∧ fact.bounds.Holds environment evaluation) ∧
    fact.boundWitnesses.MatchRoles witnessArena ∧
    (∀ relation ∈ fact.relations, relation.Holds environment) ∧
    fact.coefficientRepresentation.Holds environment.parameters value

/-- The registered carrier occurs exactly once.  This proposition avoids introducing an
independent Boolean equality procedure for matrix expressions. -/
def SymbolicMatrixEvaluation.HasExactlyOneCarrier
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (carrier : MatrixExpr) : Prop :=
  ∃ before selected after,
    evaluation.terms = before ++ selected :: after ∧
    selected.symbolic.basis = carrier ∧
    (∀ term ∈ before, term.symbolic.basis ≠ carrier) ∧
    ∀ term ∈ after, term.symbolic.basis ≠ carrier

end Mxx.Certificate
