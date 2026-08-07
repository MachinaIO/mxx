import Mxx.Certificate.SymbolicEvaluation
import Mxx.Certificate.SymbolicEvaluationConstruction
import Mxx.Certificate.LocalSoundness
import Mxx.Certificate.Rules.Family
import Mxx.Toolkit.Norms

namespace Mxx.Certificate

private def zeroLike (value : Mxx.Matrix) : Mxx.Matrix :=
  { value with coefficients := value.coefficients.map (fun _ => 0) }

private theorem addCoefficients_map_zero (values : List Int) :
    Mxx.addCoefficients values (values.map (fun _ => 0)) = values := by
  have replicated : Mxx.addCoefficients values (List.replicate values.length 0) = values := by
    induction values with
    | nil => rfl
    | cons head tail induction =>
        rw [List.length_cons, show tail.length + 1 = Nat.succ tail.length by omega,
          List.replicate_succ, Mxx.addCoefficients, induction]
        simp
  simpa using replicated

private theorem reduceCoefficient_idempotent (modulus value : Int) :
    Mxx.reduceCoefficient modulus (Mxx.reduceCoefficient modulus value) =
      Mxx.reduceCoefficient modulus value := by
  simp [Mxx.reduceCoefficient]
  split <;> simp_all

private theorem matrixAdd_zeroLike_modEq (value : Mxx.Matrix) :
    Mxx.MatrixModEq value (Mxx.matrixAdd value (zeroLike value)) := by
  refine ⟨rfl, rfl, rfl, rfl, ?_⟩
  intro row column coefficient rowLt columnLt coefficientLt
  simp only [Mxx.matrixAdd, zeroLike, addCoefficients_map_zero]
  simp only [Mxx.Matrix.coefficient]
  by_cases inBounds :
      ((row * value.columns + column) * value.ringDimension + coefficient) <
        value.coefficients.length
  · rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
    simp [inBounds, reduceCoefficient_idempotent]
  · rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
    simp [inBounds, List.length_map, Mxx.reduceCoefficient]

private theorem zeroLike_norm (value : Mxx.Matrix) :
    Mxx.maxCenteredCoefficientNorm (zeroLike value) = 0 := by
  have zeroCentered : Mxx.centeredCoefficient value.modulus 0 = 0 := by
    by_cases nonpositive : value.modulus ≤ 0
    · simp [Mxx.centeredCoefficient, nonpositive]
    · unfold Mxx.centeredCoefficient Mxx.reduceCoefficient
      simp [nonpositive]
      omega
  have normReplicate : ∀ count, Mxx.coefficientNorm (List.replicate count 0) = 0 := by
    intro count
    induction count with
    | zero => rfl
    | succ count induction => simp [List.replicate_succ, Mxx.coefficientNorm, induction]
  unfold Mxx.maxCenteredCoefficientNorm zeroLike
  simp [zeroCentered, normReplicate]

/-! # Sound hard-bound witnesses for per-term symbolic evaluations

This module relates one symbolic-form reference, one bound-witness reference, and one proof-only
`SymbolicMatrixEvaluation`.  Runtime values are never recomputed: all actual norms are taken from
the matrices already carried by the evaluation, and all symbolic values use the existing
denotation relations.
-/

/-- The three public summary roles.  Relation-local roles remain fail-closed until their checked
relation tables are threaded into this judgment. -/
inductive SummaryBoundRole : BoundRole → Prop where
  | coefficient : SummaryBoundRole .coefficient
  | noise : SummaryBoundRole .noise
  | total : SummaryBoundRole .total

/-- Actual quantity constrained by a public summary role. -/
def SymbolicMatrixEvaluation.RoleBounded
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (role : BoundRole)
    (bound : Nat) : Prop :=
  match role with
  | .coefficient => evaluation.coefficientL1 ≤ bound
  | .noise => Mxx.maxCenteredCoefficientNorm evaluation.noise ≤ bound
  | .total => Mxx.maxCenteredCoefficientNorm evaluation.value ≤ bound
  | .matchedCoefficient | .unmatchedCoefficient | .relationError => False

/-- Closed atom rules.  The bound expression is fixed by the form and role rather than supplied
by a certificate. -/
inductive BoundAtomRule : SymbolicMatrixFormEntry → BoundRole → BoundExpr → Prop where
  | signalCoefficient {matrixType expression} :
      BoundAtomRule { matrixType, form := .signalAtom expression }
        .coefficient (.constant 1)
  | signalNoise {matrixType expression} :
      BoundAtomRule { matrixType, form := .signalAtom expression }
        .noise (.constant 0)
  | signalTotal {matrixType expression} :
      BoundAtomRule { matrixType, form := .signalAtom expression }
        .total (.floorDivide (.absolute matrixType.modulus) 2)
  | boundedCoefficient {matrixType expression bound} :
      BoundAtomRule { matrixType, form := .boundedAtom expression bound }
        .coefficient (.constant 0)
  | boundedNoise {matrixType expression bound} :
      BoundAtomRule { matrixType, form := .boundedAtom expression bound } .noise bound
  | boundedTotal {matrixType expression bound} :
      BoundAtomRule { matrixType, form := .boundedAtom expression bound } .total bound
  | affineCoefficient {matrixType form totalBound} :
      BoundAtomRule { matrixType, form := .boundedAffineLeaf form totalBound }
        .coefficient form.coefficientL1Bound
  | affineNoise {matrixType form totalBound} :
      BoundAtomRule { matrixType, form := .boundedAffineLeaf form totalBound }
        .noise form.noiseBound
  | affineTotal {matrixType form totalBound} :
      BoundAtomRule { matrixType, form := .boundedAffineLeaf form totalBound }
        .total totalBound

private def addBoundValue (role : BoundRole) (modulus : Int) (left right : Nat) : Nat :=
  match role with
  | .total => min (modulus.natAbs / 2) (left + right)
  | _ => left + right

mutual

/-- A witness derives one numeric bound only by matching the exact symbolic-form DAG. -/
inductive BoundWitnessArena.DerivesBound
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena) :
    SymbolicMatrixFormRef → BoundWitnessRef → BoundRole → Nat → Prop where
  | atom {formReference : SymbolicMatrixFormRef}
      {witnessReference : BoundWitnessRef}
      {formEntry : SymbolicMatrixFormEntry}
      {role : BoundRole}
      {bound : BoundExpr}
      {value : Nat}
      (formLookup : formArena.lookup formReference = some formEntry)
      (witnessLookup : witnessArena.lookup witnessReference = some {
        role
        witness := .atom role bound
      })
      (rule : BoundAtomRule formEntry role bound)
      (evaluates : bound.evaluateWithSymbolicRecurrences
        environment.parameters environment.recurrenceStates = .ok value) :
      BoundWitnessArena.DerivesBound environment formArena witnessArena
        formReference witnessReference role value
  | add {formReference witnessReference leftForm rightForm leftWitness rightWitness
      role leftBound rightBound modulus matrixType}
      (formLookup : formArena.lookup formReference = some {
        matrixType
        form := .add leftForm rightForm
      })
      (witnessLookup : witnessArena.lookup witnessReference = some {
        role
        witness := .add leftWitness rightWitness
      })
      (modulusEvaluates : evaluateIntExpr environment.parameters matrixType.modulus =
        .ok modulus)
      (left : BoundWitnessArena.DerivesBound environment formArena witnessArena
        leftForm leftWitness role leftBound)
      (right : BoundWitnessArena.DerivesBound environment formArena witnessArena
        rightForm rightWitness role rightBound) :
      BoundWitnessArena.DerivesBound environment formArena witnessArena
        formReference witnessReference role (addBoundValue role modulus leftBound rightBound)
  | select {formReference witnessReference index formBranches witnessBranches role bounds matrixType}
      (formLookup : formArena.lookup formReference = some {
        matrixType
        form := .select index formBranches
      })
      (witnessLookup : witnessArena.lookup witnessReference = some {
        role
        witness := .selectMax index witnessBranches
      })
      (branches : BoundWitnessBranchDerivations environment formArena witnessArena role
        formBranches.toList witnessBranches.toList bounds) :
      BoundWitnessArena.DerivesBound environment formArena witnessArena
        formReference witnessReference role (Mxx.Toolkit.selectBound bounds)

/-- Pointwise derivations for a whole-form selection. -/
inductive BoundWitnessBranchDerivations
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena) :
    BoundRole → List SymbolicMatrixFormRef → List BoundWitnessRef → List Nat → Prop where
  | nil (role : BoundRole) :
      BoundWitnessBranchDerivations environment formArena witnessArena role [] [] []
  | cons {role : BoundRole}
      {formReference witnessReference bound formReferences witnessReferences bounds}
      (head : BoundWitnessArena.DerivesBound environment formArena witnessArena
        formReference witnessReference role bound)
      (tail : BoundWitnessBranchDerivations environment formArena witnessArena role
        formReferences witnessReferences bounds) :
      BoundWitnessBranchDerivations environment formArena witnessArena role
        (formReference :: formReferences) (witnessReference :: witnessReferences) (bound :: bounds)

end

theorem BoundWitnessArena.DerivesBound.witness_role
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference role bound}
    (derivation : witnessArena.DerivesBound environment formArena
      formReference witnessReference role bound) :
    ∃ entry, witnessArena.lookup witnessReference = some entry ∧ entry.role = role := by
  cases derivation with
  | atom formLookup witnessLookup rule evaluates => exact ⟨_, witnessLookup, rfl⟩
  | add formLookup witnessLookup modulusEvaluates left right => exact ⟨_, witnessLookup, rfl⟩
  | select formLookup witnessLookup branches => exact ⟨_, witnessLookup, rfl⟩

/-- A form/witness/bound triple occurs at the same branch position. -/
inductive BoundWitnessBranchDerivations.Contains
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {role : BoundRole} :
    {formReferences : List SymbolicMatrixFormRef} →
    {witnessReferences : List BoundWitnessRef} →
    {bounds : List Nat} →
    BoundWitnessBranchDerivations environment formArena witnessArena role
      formReferences witnessReferences bounds →
    SymbolicMatrixFormRef → BoundWitnessRef → Nat → Prop where
  | head {formReference witnessReference bound formReferences witnessReferences bounds}
      {headDerivation : BoundWitnessArena.DerivesBound environment formArena witnessArena
        formReference witnessReference role bound}
      {tailDerivations : BoundWitnessBranchDerivations environment formArena witnessArena role
        formReferences witnessReferences bounds} :
      BoundWitnessBranchDerivations.Contains (.cons headDerivation tailDerivations)
        formReference witnessReference bound
  | tail {headForm headWitness headBound formReferences witnessReferences bounds}
      {headDerivation : BoundWitnessArena.DerivesBound environment formArena witnessArena
        headForm headWitness role headBound}
      {tailDerivations : BoundWitnessBranchDerivations environment formArena witnessArena role
        formReferences witnessReferences bounds}
      {formReference witnessReference bound}
      (member : BoundWitnessBranchDerivations.Contains tailDerivations
        formReference witnessReference bound) :
      BoundWitnessBranchDerivations.Contains (.cons headDerivation tailDerivations)
        formReference witnessReference bound

theorem BoundWitnessBranchDerivations.Contains.bound_mem
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {role : BoundRole}
    {formReferences witnessReferences bounds}
    {derivations : BoundWitnessBranchDerivations environment formArena witnessArena role
      formReferences witnessReferences bounds}
    {formReference witnessReference bound}
    (member : derivations.Contains formReference witnessReference bound) :
    bound ∈ bounds := by
  induction member with
  | head => simp
  | tail _ induction => exact List.mem_cons_of_mem _ induction

/-- Semantic soundness of one witness root for one evaluation of the same form reference. -/
structure BoundWitnessArena.Holds
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    {formReference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (witnessReference : BoundWitnessRef)
    (role : BoundRole)
    (bound : Nat) : Prop where
  derives : witnessArena.DerivesBound environment formArena
    formReference witnessReference role bound
  actual : evaluation.RoleBounded role bound

/-- Full semantic meaning of one analyzer-owned symbolic matrix fact.  Each of the three public
bound roots must both derive from the exact symbolic-form DAG and bound the corresponding runtime
quantity of one common evaluation.  Thus `MatchRoles` alone cannot make a symbolic fact true. -/
def MatrixSymbolicFact.SemanticallyHolds
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    (fact : MatrixSymbolicFact) : Prop :=
  ∃ value,
    ∃ evaluation : SymbolicMatrixEvaluation environment formArena fact.decomposition,
    ∃ coefficientBound noiseBound totalBound,
    fact.Holds environment formArena witnessArena value ∧
    evaluation.value = value ∧
    evaluation.MatchesForm ∧
    fact.bounds.Holds environment evaluation ∧
    fact.bounds.coefficientL1Bound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok coefficientBound ∧
    fact.bounds.noiseBound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok noiseBound ∧
    fact.bounds.totalBound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok totalBound ∧
    witnessArena.Holds environment formArena evaluation
      fact.boundWitnesses.coefficientL1 .coefficient coefficientBound ∧
    witnessArena.Holds environment formArena evaluation
      fact.boundWitnesses.noise .noise noiseBound ∧
    witnessArena.Holds environment formArena evaluation
      fact.boundWitnesses.total .total totalBound

/-- Soundness of an analyzer result includes every ordinary fact and every analyzer-constructed
symbolic matrix evaluation.  The latter is therefore theorem-relevant data rather than an
unchecked report attached to `AnalysisResult`. -/
def ParallelFamilyAnalysesOwned (analysis : AnalysisResult) : Prop :=
  ∀ joint family, (joint, family) ∈ analysis.families →
    ∃ source,
      analysis.parallelFamilyDerivations.filter (fun candidate => candidate.family = joint) =
        [source] ∧
      source.MatchesFamily family ∧
      source.indexExpression = .loopIndex { site := source.loopSite } ∧
      analysis.expressionArena.lookupInteger source.indexReference = some source.indexExpression ∧
      source.OutputFactsMatchBody ∧
      source.outputFacts.mapM ScopedWireFact.toTemplate = some source.elementTemplates

def AnalysisHolds (environment : FactEnvironment) (analysis : AnalysisResult) : Prop :=
  BaseAnalysisHolds environment analysis ∧
    (∀ fact ∈ analysis.symbolicMatrixFacts,
      fact.SemanticallyHolds environment analysis.symbolicFormArena analysis.boundWitnessArena) ∧
    ParallelFamilyAnalysesOwned analysis

/-- The evaluation is exactly the per-term sum of two child evaluations.  This is proof evidence,
not a new executable operation. -/
structure SymbolicMatrixEvaluation.IsAddOf
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference leftReference rightReference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (left : SymbolicMatrixEvaluation environment arena leftReference)
    (right : SymbolicMatrixEvaluation environment arena rightReference) : Prop where
  value : evaluation.value = Mxx.matrixAdd left.value right.value
  terms : evaluation.terms = left.terms ++ right.terms
  noise : evaluation.noise = Mxx.matrixAdd left.noise right.noise

/-- A whole-form selection reuses exactly the selected branch's per-term interpretation. -/
structure SymbolicMatrixEvaluation.IsSelectionOf
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference selectedReference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (selected : SymbolicMatrixEvaluation environment arena selectedReference) : Prop where
  value : evaluation.value = selected.value
  terms : evaluation.terms = selected.terms
  noise : evaluation.noise = selected.noise

/-- A signal atom contributes exactly one independently identified carrier term.  The term itself
retains the checked coefficient identity and its match to the coefficient expression; this
structure additionally ties the carrier to the exact matrix expression stored by the form. -/
def SymbolicMatrixEvaluation.IsSignalAtomOf
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference : SymbolicMatrixFormRef}
    (evaluation : SymbolicMatrixEvaluation environment arena reference)
    (expression : MatrixExprRef)
    (carrier : MatrixExpr) : Prop :=
  environment.expressionArena.lookupMatrix expression = some carrier ∧
  ∃ term : EvaluatedSignalTerm environment,
    evaluation.terms = [term] ∧
    term.symbolic.basis = carrier ∧
    term.coefficientBound ≤ 1 ∧
    Mxx.maxCenteredCoefficientNorm evaluation.noise = 0

private theorem coefficientFoldl_from
    {environment : FactEnvironment}
    (terms : List (EvaluatedSignalTerm environment))
    (initial : Nat) :
    terms.foldl (fun total term ↦ total + term.coefficientBound) initial =
      initial + terms.foldl (fun total term ↦ total + term.coefficientBound) 0 := by
  induction terms generalizing initial with
  | nil => simp
  | cons head tail induction =>
      simp only [List.foldl_cons]
      rw [induction (initial + head.coefficientBound)]
      have tailFromHead :
          tail.foldl (fun total term ↦ total + term.coefficientBound)
              (0 + head.coefficientBound) =
            head.coefficientBound +
              tail.foldl (fun total term ↦ total + term.coefficientBound) 0 := by
        simpa using induction head.coefficientBound
      rw [tailFromHead]
      omega

private theorem coefficientBoundFold_evaluates
    (environment : FactEnvironment)
    (terms : List (EvaluatedSignalTerm environment))
    (initialExpression : BoundExpr)
    (initialValue : Nat)
    (initialEvaluates : initialExpression.evaluate environment.parameters = .ok initialValue) :
    (terms.foldl (fun bound term => .add bound term.symbolic.coefficient.normBound)
      initialExpression).evaluate environment.parameters =
    .ok (terms.foldl (fun total term => total + term.coefficientBound) initialValue) := by
  induction terms generalizing initialExpression initialValue with
  | nil => exact initialEvaluates
  | cons head tail induction =>
      simp only [List.foldl_cons]
      apply induction
      rw [show BoundExpr.evaluate environment.parameters
        (.add initialExpression head.symbolic.coefficient.normBound) =
          (do
            let left ← initialExpression.evaluate environment.parameters
            let right ← head.symbolic.coefficient.normBound.evaluate environment.parameters
            pure (left + right)) by rfl,
        initialEvaluates, head.coefficientBoundEvaluates]
      change Except.ok (initialValue + head.coefficientBound) =
        Except.ok (initialValue + head.coefficientBound)
      rfl

/-- The affine coefficient expression evaluates to the exact sum of the independently certified
coefficient bounds retained by the corresponding evaluated terms. -/
theorem AffineForm.coefficientL1Bound_evaluates
    (environment : FactEnvironment)
    (form : AffineForm)
    (terms : List (EvaluatedSignalTerm environment))
    (alignment : terms.map (fun term => term.symbolic) = form.terms) :
    form.coefficientL1Bound.evaluate environment.parameters =
      .ok (terms.foldl (fun total term => total + term.coefficientBound) 0) := by
  cases terms with
  | nil =>
      simp at alignment
      simp [AffineForm.coefficientL1Bound, alignment, BoundExpr.evaluate]
  | cons head tail =>
      have formTerms : form.terms = head.symbolic :: tail.map (fun term => term.symbolic) :=
        alignment.symm
      rw [AffineForm.coefficientL1Bound, formTerms]
      simp only [List.foldl_cons, List.foldl_map]
      have evaluated := coefficientBoundFold_evaluates environment tail
        head.symbolic.coefficient.normBound head.coefficientBound
        head.coefficientBoundEvaluates
      rw [evaluated]
      simp

private theorem coefficientL1_append
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference leftReference rightReference : SymbolicMatrixFormRef}
    {evaluation : SymbolicMatrixEvaluation environment arena reference}
    {left : SymbolicMatrixEvaluation environment arena leftReference}
    {right : SymbolicMatrixEvaluation environment arena rightReference}
    (composition : evaluation.IsAddOf left right) :
    evaluation.coefficientL1 = left.coefficientL1 + right.coefficientL1 := by
  simp only [SymbolicMatrixEvaluation.coefficientL1, composition.terms, List.foldl_append]
  rw [coefficientFoldl_from]

/-- A signal atom retains one exact carrier identity and has coefficient mass at most one. -/
theorem signalAtom_coefficient_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression carrier}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .signalAtom expression
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .coefficient
      witness := .atom .coefficient (.constant 1)
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (atomEvaluation : evaluation.IsSignalAtomOf expression carrier) :
    witnessArena.Holds environment formArena evaluation witnessReference .coefficient 1 := by
  obtain ⟨_, term, terms, _, coefficientNorm, _⟩ := atomEvaluation
  constructor
  · exact .atom formLookup witnessLookup .signalCoefficient rfl
  · simpa [SymbolicMatrixEvaluation.RoleBounded,
      SymbolicMatrixEvaluation.coefficientL1, terms] using coefficientNorm

/-- A signal atom contributes no noise independently of its carrier value. -/
theorem signalAtom_noise_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression carrier}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .signalAtom expression
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .noise
      witness := .atom .noise (.constant 0)
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (atomEvaluation : evaluation.IsSignalAtomOf expression carrier) :
    witnessArena.Holds environment formArena evaluation witnessReference .noise 0 := by
  obtain ⟨_, _, _, _, _, noiseNorm⟩ := atomEvaluation
  constructor
  · exact .atom formLookup witnessLookup .signalNoise rfl
  · simpa [SymbolicMatrixEvaluation.RoleBounded] using noiseNorm.le

/-- A signal atom's stored value is bounded only by the centered modulus radius. -/
theorem signalAtom_total_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression carrier boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .signalAtom expression
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .total
      witness := .atom .total
        (.floorDivide (.absolute formEntry.matrixType.modulus) 2)
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (atomEvaluation : evaluation.IsSignalAtomOf expression carrier)
    (boundEvaluates :
      BoundExpr.evaluateWithSymbolicRecurrences environment.parameters environment.recurrenceStates
        (BoundExpr.floorDivide (.absolute formEntry.matrixType.modulus) 2) =
        Except.ok boundValue)
    (actualNorm : Mxx.maxCenteredCoefficientNorm evaluation.value ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference .total boundValue := by
  obtain ⟨_, _⟩ := atomEvaluation
  constructor
  · exact .atom formLookup witnessLookup .signalTotal boundEvaluates
  · exact actualNorm

/-- The exact signal leaf emitted by construction has one identity-weighted carrier and zero
noise. Its semantic evaluation and all three bound roots are determined by the analyzer-owned
arena entries. -/
theorem constructedSignalAtom_semanticallyHolds
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    (subject : ValueInstanceRef)
    (matrixType : MatrixTypeExpr)
    (expression : MatrixExpr)
    (exactReference : MatrixExprRef)
    (formReference : SymbolicMatrixFormRef)
    (coefficientWitness noiseWitness totalWitness : BoundWitnessRef)
    (relations : List MatrixRelation)
    (representation : CoefficientRepresentation)
    (value : Mxx.Matrix)
    (totalBoundValue : Nat)
    (subjectLookup : environment.values subject = some (.matrix value))
    (expressionDenotes : MatrixExpr.Denotes environment expression value)
    (expressionLookup : environment.expressionArena.lookupMatrix exactReference = some expression)
    (formLookup : formArena.lookup formReference = some {
      matrixType, form := .signalAtom exactReference
    })
    (coefficientLookup : witnessArena.lookup coefficientWitness = some {
      role := .coefficient, witness := .atom .coefficient (.constant 1)
    })
    (noiseLookup : witnessArena.lookup noiseWitness = some {
      role := .noise, witness := .atom .noise (.constant 0)
    })
    (totalLookup : witnessArena.lookup totalWitness = some {
      role := .total
      witness := .atom .total
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2)
    })
    (totalEvaluates :
      (BoundExpr.floorDivide (.absolute matrixType.modulus) 2).evaluateWithSymbolicRecurrences
        environment.parameters environment.recurrenceStates = Except.ok totalBoundValue)
    (totalNorm : Mxx.maxCenteredCoefficientNorm value ≤ totalBoundValue)
    (relationsHold : ∀ relation ∈ relations, relation.Holds environment)
    (representationHolds : representation.Holds environment.parameters value) :
    MatrixSymbolicFact.SemanticallyHolds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := MatrixBoundSummary.exactLarge
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2)
      boundWitnesses := {
        coefficientL1 := coefficientWitness
        noise := noiseWitness
        total := totalWitness
      }
      relations
      coefficientRepresentation := representation
    } := by
  let symbolicTerm : SignalTerm := {
    coefficient := { expression := .identity matrixType, normBound := .constant 1 }
    basis := expression
    mode := .ordinaryMatrixProduct
  }
  let evaluatedTerm : EvaluatedSignalTerm environment := {
    symbolic := symbolicTerm
    coefficientIdentity := .matrix (.identity matrixType)
    coefficientBound := 1
    carrierValue := value
    termValue := value
    identityMatches := rfl
    coefficientBoundEvaluates := rfl
    carrierDenotes := expressionDenotes
    termDenotes := .identityCoefficient expressionDenotes
  }
  let evaluation : SymbolicMatrixEvaluation environment formArena formReference := {
    value
    terms := [evaluatedTerm]
    noise := zeroLike value
    denotes := .entry formLookup (.signalAtom expressionLookup expressionDenotes)
    valueEquation := by simpa [evaluatedTerm] using matrixAdd_zeroLike_modEq value
  }
  have atomEvaluation : evaluation.IsSignalAtomOf exactReference expression := by
    exact ⟨expressionLookup, evaluatedTerm, rfl, rfl, by simp [evaluatedTerm],
      by simp [evaluation, zeroLike_norm]⟩
  have summaryHolds :
      (MatrixBoundSummary.exactLarge
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2)).Holds environment evaluation := by
    refine ⟨1, 0, totalBoundValue, rfl, rfl, totalEvaluates, ?_, ?_, totalNorm, ?_⟩
    · simp [evaluation, evaluatedTerm, SymbolicMatrixEvaluation.coefficientL1]
    · simp [evaluation, zeroLike_norm]
    · simp [MatrixBoundSummary.exactLarge]
  let roots : MatrixBoundWitnessRefs := {
    coefficientL1 := coefficientWitness
    noise := noiseWitness
    total := totalWitness
  }
  have roles : roots.MatchRoles witnessArena := by
    exact ⟨⟨_, coefficientLookup, rfl⟩, ⟨⟨_, noiseLookup, rfl⟩,
      ⟨_, totalLookup, rfl⟩⟩⟩
  have factHolds : MatrixSymbolicFact.Holds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := MatrixBoundSummary.exactLarge
        (BoundExpr.floorDivide (.absolute matrixType.modulus) 2)
      boundWitnesses := roots
      relations
      coefficientRepresentation := representation
    } value := by
    exact ⟨subjectLookup, ⟨expression, expressionLookup, expressionDenotes⟩,
      ⟨evaluation, rfl, summaryHolds⟩, roles, relationsHold, representationHolds⟩
  refine ⟨value, evaluation, 1, 0, totalBoundValue, factHolds, rfl, ?_, summaryHolds,
    rfl, rfl, totalEvaluates, ?_, ?_, ?_⟩
  · exact .signalAtom formLookup rfl ⟨expression, expressionLookup, rfl⟩
      (by simp [evaluation, zeroLike_norm])
  · exact signalAtom_coefficient_witness_sound
      (formEntry := { matrixType, form := .signalAtom exactReference })
      formLookup coefficientLookup evaluation atomEvaluation
  · exact signalAtom_noise_witness_sound
      (formEntry := { matrixType, form := .signalAtom exactReference })
      formLookup noiseLookup evaluation atomEvaluation
  · exact signalAtom_total_witness_sound
      (formEntry := { matrixType, form := .signalAtom exactReference })
      formLookup totalLookup evaluation atomEvaluation totalEvaluates totalNorm

/-- A bounded atom has zero coefficient mass. -/
theorem boundedAtom_coefficient_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression bound}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAtom expression bound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .coefficient
      witness := .atom .coefficient (.constant 0)
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (noTerms : evaluation.terms = []) :
    witnessArena.Holds environment formArena evaluation witnessReference .coefficient 0 := by
  constructor
  · exact .atom formLookup witnessLookup .boundedCoefficient rfl
  · simp [SymbolicMatrixEvaluation.RoleBounded, SymbolicMatrixEvaluation.coefficientL1, noTerms]

/-- A bounded atom's declared bound controls its complete noise value. -/
theorem boundedAtom_noise_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression bound boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAtom expression bound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .noise
      witness := .atom .noise bound
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (noiseIsValue : evaluation.noise = evaluation.value)
    (boundEvaluates : bound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (actualNorm : Mxx.maxCenteredCoefficientNorm evaluation.value ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference .noise boundValue := by
  constructor
  · exact .atom formLookup witnessLookup .boundedNoise boundEvaluates
  · rw [SymbolicMatrixEvaluation.RoleBounded, noiseIsValue]
    exact actualNorm

/-- A bounded atom's declared bound also controls its stored value. -/
theorem boundedAtom_total_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {expression bound boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAtom expression bound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .total
      witness := .atom .total bound
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (boundEvaluates : bound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (actualNorm : Mxx.maxCenteredCoefficientNorm evaluation.value ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference .total boundValue := by
  constructor
  · exact .atom formLookup witnessLookup .boundedTotal boundEvaluates
  · rw [SymbolicMatrixEvaluation.RoleBounded]
    exact actualNorm

/-- The bounded leaf emitted by construction carries no signal terms and uses its complete value
as noise. The same analyzer bound controls its form denotation, summary, and noise/total roots. -/
theorem constructedBoundedAtom_semanticallyHolds
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    (subject : ValueInstanceRef)
    (matrixType : MatrixTypeExpr)
    (expression : MatrixExpr)
    (exactReference : MatrixExprRef)
    (formReference : SymbolicMatrixFormRef)
    (coefficientWitness noiseWitness totalWitness : BoundWitnessRef)
    (relations : List MatrixRelation)
    (representation : CoefficientRepresentation)
    (bound : BoundExpr)
    (value : Mxx.Matrix)
    (boundValue : Nat)
    (subjectLookup : environment.values subject = some (.matrix value))
    (expressionDenotes : MatrixExpr.Denotes environment expression value)
    (expressionLookup : environment.expressionArena.lookupMatrix exactReference = some expression)
    (formLookup : formArena.lookup formReference = some {
      matrixType, form := .boundedAtom exactReference bound
    })
    (coefficientLookup : witnessArena.lookup coefficientWitness = some {
      role := .coefficient, witness := .atom .coefficient (.constant 0)
    })
    (noiseLookup : witnessArena.lookup noiseWitness = some {
      role := .noise, witness := .atom .noise bound
    })
    (totalLookup : witnessArena.lookup totalWitness = some {
      role := .total, witness := .atom .total bound
    })
    (boundEvaluates : bound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (boundEvaluatesClosed : bound.evaluate environment.parameters = .ok boundValue)
    (actualNorm : Mxx.maxCenteredCoefficientNorm value ≤ boundValue)
    (relationsHold : ∀ relation ∈ relations, relation.Holds environment)
    (representationHolds : representation.Holds environment.parameters value) :
    MatrixSymbolicFact.SemanticallyHolds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := MatrixBoundSummary.bounded bound
      boundWitnesses := {
        coefficientL1 := coefficientWitness
        noise := noiseWitness
        total := totalWitness
      }
      relations
      coefficientRepresentation := representation
    } := by
  let evaluation : SymbolicMatrixEvaluation environment formArena formReference := {
    value
    terms := []
    noise := value
    denotes := .entry formLookup (.boundedAtom expressionLookup
      ⟨expressionDenotes, boundValue, boundEvaluatesClosed, actualNorm⟩)
    valueEquation := Mxx.MatrixModEq.refl value
  }
  have summaryHolds : (MatrixBoundSummary.bounded bound).Holds environment evaluation := by
    refine ⟨0, boundValue, boundValue, rfl, boundEvaluates, boundEvaluates,
      ?_, ?_, actualNorm, ?_⟩
    · simp [evaluation, SymbolicMatrixEvaluation.coefficientL1]
    · simpa [evaluation] using actualNorm
    · simp [evaluation]
  let roots : MatrixBoundWitnessRefs := {
    coefficientL1 := coefficientWitness
    noise := noiseWitness
    total := totalWitness
  }
  have roles : roots.MatchRoles witnessArena := by
    exact ⟨⟨_, coefficientLookup, rfl⟩, ⟨⟨_, noiseLookup, rfl⟩,
      ⟨_, totalLookup, rfl⟩⟩⟩
  have factHolds : MatrixSymbolicFact.Holds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := MatrixBoundSummary.bounded bound
      boundWitnesses := roots
      relations
      coefficientRepresentation := representation
    } value := by
    exact ⟨subjectLookup, ⟨expression, expressionLookup, expressionDenotes⟩,
      ⟨evaluation, rfl, summaryHolds⟩, roles, relationsHold, representationHolds⟩
  refine ⟨value, evaluation, 0, boundValue, boundValue, factHolds, rfl, ?_, summaryHolds,
    rfl, boundEvaluates, boundEvaluates, ?_, ?_, ?_⟩
  · exact .boundedAtom formLookup rfl rfl
  · exact boundedAtom_coefficient_witness_sound
      (formEntry := { matrixType, form := .boundedAtom exactReference bound })
      formLookup coefficientLookup evaluation rfl
  · exact boundedAtom_noise_witness_sound
      (formEntry := { matrixType, form := .boundedAtom exactReference bound })
      formLookup noiseLookup evaluation rfl boundEvaluates actualNorm
  · exact boundedAtom_total_witness_sound
      (formEntry := { matrixType, form := .boundedAtom exactReference bound })
      formLookup totalLookup evaluation boundEvaluates actualNorm

/-- A normalized affine leaf derives its coefficient witness from the preserved term list. -/
theorem boundedAffineLeaf_coefficient_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {form totalBound boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAffineLeaf form totalBound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .coefficient
      witness := .atom .coefficient form.coefficientL1Bound
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (boundEvaluates : form.coefficientL1Bound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (actualCoefficient : evaluation.coefficientL1 ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference
      .coefficient boundValue := by
  constructor
  · exact .atom formLookup witnessLookup .affineCoefficient boundEvaluates
  · exact actualCoefficient

/-- A normalized affine leaf derives its noise witness from the preserved affine noise bound. -/
theorem boundedAffineLeaf_noise_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {form totalBound boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAffineLeaf form totalBound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .noise
      witness := .atom .noise form.noiseBound
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (boundEvaluates : form.noiseBound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (actualNoise : Mxx.maxCenteredCoefficientNorm evaluation.noise ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference .noise boundValue := by
  constructor
  · exact .atom formLookup witnessLookup .affineNoise boundEvaluates
  · exact actualNoise

/-- A normalized affine leaf derives its total witness from the bound stored in the form node. -/
theorem boundedAffineLeaf_total_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference : _}
    {formEntry : SymbolicMatrixFormEntry}
    {form totalBound boundValue}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .boundedAffineLeaf form totalBound
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role := .total
      witness := .atom .total totalBound
    })
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (boundEvaluates : totalBound.evaluateWithSymbolicRecurrences
      environment.parameters environment.recurrenceStates = .ok boundValue)
    (actualTotal : Mxx.maxCenteredCoefficientNorm evaluation.value ≤ boundValue) :
    witnessArena.Holds environment formArena evaluation witnessReference .total boundValue := by
  constructor
  · exact .atom formLookup witnessLookup .affineTotal boundEvaluates
  · exact actualTotal

/-- The normalized affine leaf emitted by construction retains the analyzer's exact term list,
noise value, and hard bounds. Its semantic evaluation is reconstructed solely from the ordinary
affine fact and analyzer-owned arena entries. -/
theorem constructedBoundedAffineLeaf_semanticallyHolds
    (environment : FactEnvironment)
    (formArena : SymbolicMatrixFormArena)
    (witnessArena : BoundWitnessArena)
    (subject : ValueInstanceRef)
    (matrixType : MatrixTypeExpr)
    (exactExpression : MatrixExpr)
    (exactReference : MatrixExprRef)
    (formReference : SymbolicMatrixFormRef)
    (coefficientWitness noiseWitness totalWitness : BoundWitnessRef)
    (relations : List MatrixRelation)
    (representation : CoefficientRepresentation)
    (form : AffineForm)
    (totalBound : BoundExpr)
    (value : Mxx.Matrix)
    (totalBoundValue : Nat)
    (subjectLookup : environment.values subject = some (.matrix value))
    (exactDenotes : MatrixExpr.Denotes environment exactExpression value)
    (exactLookup : environment.expressionArena.lookupMatrix exactReference = some exactExpression)
    (formLookup : formArena.lookup formReference = some {
      matrixType, form := .boundedAffineLeaf form totalBound
    })
    (coefficientLookup : witnessArena.lookup coefficientWitness = some {
      role := .coefficient, witness := .atom .coefficient form.coefficientL1Bound
    })
    (noiseLookup : witnessArena.lookup noiseWitness = some {
      role := .noise, witness := .atom .noise form.noiseBound
    })
    (totalLookup : witnessArena.lookup totalWitness = some {
      role := .total, witness := .atom .total totalBound
    })
    (formHolds : form.Holds environment value)
    (totalEvaluatesClosed : totalBound.evaluate environment.parameters = .ok totalBoundValue)
    (totalNorm : Mxx.maxCenteredCoefficientNorm value ≤ totalBoundValue)
    (relationsHold : ∀ relation ∈ relations, relation.Holds environment)
    (representationHolds : representation.Holds environment.parameters value) :
    MatrixSymbolicFact.SemanticallyHolds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := {
        signal := if form.terms.isEmpty then .none else .present
        coefficientL1Bound := form.coefficientL1Bound
        noiseBound := form.noiseBound
        totalBound
      }
      boundWitnesses := {
        coefficientL1 := coefficientWitness
        noise := noiseWitness
        total := totalWitness
      }
      relations
      coefficientRepresentation := representation
    } := by
  obtain ⟨termValues, noise, noiseBoundValue, termsDenote, noiseEvaluatesClosed,
    noiseNorm, reconstruction⟩ := formHolds
  obtain ⟨evaluatedTerms, symbolicAlignment, valueAlignment⟩ :=
    evaluatedSignalTerms_exists termsDenote
  have coefficientEvaluatesClosed :=
    form.coefficientL1Bound_evaluates environment evaluatedTerms symbolicAlignment
  have coefficientEvaluates :=
    BoundExpr.evaluateWithSymbolicRecurrences_of_evaluate_eq_ok
      environment.parameters environment.recurrenceStates coefficientEvaluatesClosed
  have noiseEvaluates :=
    BoundExpr.evaluateWithSymbolicRecurrences_of_evaluate_eq_ok
      environment.parameters environment.recurrenceStates noiseEvaluatesClosed
  have totalEvaluates :=
    BoundExpr.evaluateWithSymbolicRecurrences_of_evaluate_eq_ok
      environment.parameters environment.recurrenceStates totalEvaluatesClosed
  let evaluation : SymbolicMatrixEvaluation environment formArena formReference := {
    value
    terms := evaluatedTerms
    noise
    denotes := .entry formLookup (.boundedAffineLeaf
      ⟨termValues, noise, noiseBoundValue, termsDenote, noiseEvaluatesClosed, noiseNorm,
        reconstruction⟩)
    valueEquation := by
      have reconstructed := reconstruction
      rw [← valueAlignment] at reconstructed
      simpa [List.foldr_map] using reconstructed
  }
  let summary : MatrixBoundSummary := {
    signal := if form.terms.isEmpty then .none else .present
    coefficientL1Bound := form.coefficientL1Bound
    noiseBound := form.noiseBound
    totalBound
  }
  have noTermsWhenNone : summary.signal = .none → evaluation.terms = [] := by
    intro noSignal
    simp only [summary] at noSignal
    split at noSignal
    · rename_i empty
      have formEmpty : form.terms = [] := by
        cases termsEq : form.terms with
        | nil => rfl
        | cons head tail => simp [termsEq] at empty
      have : evaluatedTerms.map (fun term => term.symbolic) = [] := by
        simpa [formEmpty] using symbolicAlignment
      simpa using this
    · contradiction
  have summaryHolds : summary.Holds environment evaluation := by
    refine ⟨evaluation.coefficientL1, noiseBoundValue, totalBoundValue,
      coefficientEvaluates, noiseEvaluates, totalEvaluates, le_rfl, ?_, totalNorm,
      noTermsWhenNone⟩
    simpa [evaluation] using noiseNorm
  let roots : MatrixBoundWitnessRefs := {
    coefficientL1 := coefficientWitness
    noise := noiseWitness
    total := totalWitness
  }
  have roles : roots.MatchRoles witnessArena := by
    exact ⟨⟨_, coefficientLookup, rfl⟩, ⟨⟨_, noiseLookup, rfl⟩,
      ⟨_, totalLookup, rfl⟩⟩⟩
  have factHolds : MatrixSymbolicFact.Holds environment formArena witnessArena {
      subject
      matrixType
      exactValue := exactReference
      decomposition := formReference
      bounds := summary
      boundWitnesses := roots
      relations
      coefficientRepresentation := representation
    } value := by
    exact ⟨subjectLookup, ⟨exactExpression, exactLookup, exactDenotes⟩,
      ⟨evaluation, rfl, summaryHolds⟩, roles, relationsHold, representationHolds⟩
  refine ⟨value, evaluation, evaluation.coefficientL1, noiseBoundValue, totalBoundValue,
    factHolds, rfl, ?_, summaryHolds, coefficientEvaluates, noiseEvaluates, totalEvaluates,
    ?_, ?_, ?_⟩
  · exact .boundedAffineLeaf formLookup symbolicAlignment
      ⟨noiseBoundValue, noiseEvaluatesClosed, by simpa [evaluation] using noiseNorm⟩
  · exact boundedAffineLeaf_coefficient_witness_sound
      (formEntry := { matrixType, form := .boundedAffineLeaf form totalBound })
      formLookup coefficientLookup evaluation coefficientEvaluates le_rfl
  · exact boundedAffineLeaf_noise_witness_sound
      (formEntry := { matrixType, form := .boundedAffineLeaf form totalBound })
      formLookup noiseLookup evaluation noiseEvaluates (by simpa [evaluation] using noiseNorm)
  · exact boundedAffineLeaf_total_witness_sound
      (formEntry := { matrixType, form := .boundedAffineLeaf form totalBound })
      formLookup totalLookup evaluation totalEvaluates totalNorm

/-- Closed addition preserves every public summary role using actual matrix norms. -/
theorem add_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference leftForm rightForm witnessReference leftWitness rightWitness : _}
    {formEntry : SymbolicMatrixFormEntry}
    {role : BoundRole}
    {leftBound rightBound : Nat}
    {q : Nat} [NeZero q]
    (publicRole : SummaryBoundRole role)
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .add leftForm rightForm
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role
      witness := .add leftWitness rightWitness
    })
    (modulusEvaluates : evaluateIntExpr environment.parameters formEntry.matrixType.modulus =
      .ok (q : Int))
    (left : SymbolicMatrixEvaluation environment formArena leftForm)
    (right : SymbolicMatrixEvaluation environment formArena rightForm)
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (composition : evaluation.IsAddOf left right)
    (leftHolds : witnessArena.Holds environment formArena left leftWitness role leftBound)
    (rightHolds : witnessArena.Holds environment formArena right rightWitness role rightBound)
    (leftNoiseModulus : left.noise.modulus = q)
    (rightNoiseModulus : right.noise.modulus = q)
    (leftValueModulus : left.value.modulus = q)
    (rightValueModulus : right.value.modulus = q) :
    witnessArena.Holds environment formArena evaluation witnessReference role
      (addBoundValue role (q : Int) leftBound rightBound) := by
  constructor
  · exact .add formLookup witnessLookup modulusEvaluates leftHolds.derives rightHolds.derives
  · cases publicRole with
    | coefficient =>
        simp only [SymbolicMatrixEvaluation.RoleBounded, addBoundValue]
        rw [coefficientL1_append composition]
        exact Nat.add_le_add leftHolds.actual rightHolds.actual
    | noise =>
        simp only [SymbolicMatrixEvaluation.RoleBounded, addBoundValue]
        rw [composition.noise]
        exact le_trans
          (Mxx.Toolkit.matrixAdd_norm_le q left.noise right.noise
            leftNoiseModulus rightNoiseModulus)
          (Nat.add_le_add leftHolds.actual rightHolds.actual)
    | total =>
        simp only [SymbolicMatrixEvaluation.RoleBounded, addBoundValue]
        rw [composition.value]
        apply le_min
        · have qPositive : (0 : Int) < q := by
            exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne q)
          have resultPositive : 0 < (Mxx.matrixAdd left.value right.value).modulus := by
            simpa [Mxx.matrixAdd, leftValueModulus] using qPositive
          have radius := matrix_norm_le_centered_radius
            (Mxx.matrixAdd left.value right.value) resultPositive
          simpa [Mxx.matrixAdd, leftValueModulus] using radius
        · exact le_trans
            (Mxx.Toolkit.matrixAdd_norm_le q left.value right.value
              leftValueModulus rightValueModulus)
            (Nat.add_le_add leftHolds.actual rightHolds.actual)

private theorem selection_role_bounded
    {environment : FactEnvironment}
    {arena : SymbolicMatrixFormArena}
    {reference selectedReference : SymbolicMatrixFormRef}
    {evaluation : SymbolicMatrixEvaluation environment arena reference}
    {selected : SymbolicMatrixEvaluation environment arena selectedReference}
    {role : BoundRole}
    {bound : Nat}
    (composition : evaluation.IsSelectionOf selected)
    (selectedBounded : selected.RoleBounded role bound) :
    evaluation.RoleBounded role bound := by
  cases role <;> simp only [SymbolicMatrixEvaluation.RoleBounded] at selectedBounded ⊢
  · simpa [SymbolicMatrixEvaluation.coefficientL1, composition.terms] using selectedBounded
  · simpa [composition.noise] using selectedBounded
  · simpa [composition.value] using selectedBounded

/-- A selected branch is bounded by the statically checked maximum over every witness branch. -/
theorem select_witness_sound
    {environment : FactEnvironment}
    {formArena : SymbolicMatrixFormArena}
    {witnessArena : BoundWitnessArena}
    {formReference witnessReference selectedForm selectedWitness : _}
    {formEntry : SymbolicMatrixFormEntry}
    {index formBranches witnessBranches role branchBound}
    {bounds : List Nat}
    (formLookup : formArena.lookup formReference = some {
      formEntry with form := .select index formBranches
    })
    (witnessLookup : witnessArena.lookup witnessReference = some {
      role
      witness := .selectMax index witnessBranches
    })
    (branchDerivations : BoundWitnessBranchDerivations environment formArena witnessArena role
      formBranches.toList witnessBranches.toList bounds)
    (selectedMember : branchDerivations.Contains selectedForm selectedWitness branchBound)
    (selected : SymbolicMatrixEvaluation environment formArena selectedForm)
    (evaluation : SymbolicMatrixEvaluation environment formArena formReference)
    (composition : evaluation.IsSelectionOf selected)
    (selectedActual : selected.RoleBounded role branchBound) :
    witnessArena.Holds environment formArena evaluation witnessReference role
      (Mxx.Toolkit.selectBound bounds) := by
  constructor
  · exact .select formLookup witnessLookup branchDerivations
  · exact selection_role_bounded composition <|
      match role with
      | .coefficient => selectedActual.trans
          (Mxx.Toolkit.selectBound_contains bounds _ selectedMember.bound_mem)
      | .noise => selectedActual.trans
          (Mxx.Toolkit.selectBound_contains bounds _ selectedMember.bound_mem)
      | .total => selectedActual.trans
          (Mxx.Toolkit.selectBound_contains bounds _ selectedMember.bound_mem)
      | .matchedCoefficient | .unmatchedCoefficient | .relationError => False.elim selectedActual

namespace SymbolicEvaluationSoundnessFixtures

private def matrixType : MatrixTypeExpr where
  modulus := .constant 17
  ringDimension := .constant 4
  rows := .constant 2
  columns := .constant 2

private def leftForm : SymbolicMatrixFormRef := ⟨0⟩
private def rightForm : SymbolicMatrixFormRef := ⟨1⟩
private def addForm : SymbolicMatrixFormRef := ⟨2⟩
private def selectForm : SymbolicMatrixFormRef := ⟨3⟩
private def leftWitness : BoundWitnessRef := ⟨0⟩
private def rightWitness : BoundWitnessRef := ⟨1⟩
private def addWitness : BoundWitnessRef := ⟨2⟩
private def selectWitness : BoundWitnessRef := ⟨3⟩
private def index : RuntimeExprRef .integer := ⟨0⟩

private def formArena : SymbolicMatrixFormArena := ⟨#[
  { matrixType, form := .boundedAtom ⟨0⟩ (.constant 3) },
  { matrixType, form := .boundedAtom ⟨1⟩ (.constant 5) },
  { matrixType, form := .add leftForm rightForm },
  { matrixType, form := .select index ⟨leftForm, [rightForm]⟩ }
]⟩

private def witnessArena : BoundWitnessArena := ⟨#[
  { role := .total, witness := .atom .total (.constant 3) },
  { role := .total, witness := .atom .total (.constant 5) },
  { role := .total, witness := .add leftWitness rightWitness },
  { role := .total, witness := .selectMax index ⟨leftWitness, [rightWitness]⟩ }
]⟩

private theorem leftDerives (environment : FactEnvironment) :
    witnessArena.DerivesBound environment formArena leftForm leftWitness .total 3 := by
  exact .atom rfl rfl .boundedTotal rfl

private theorem rightDerives (environment : FactEnvironment) :
    witnessArena.DerivesBound environment formArena rightForm rightWitness .total 5 := by
  exact .atom rfl rfl .boundedTotal rfl

/-- Fixture: an atom witness must resolve the exact atom form and exact witness entry. -/
example (environment : FactEnvironment) :
    witnessArena.DerivesBound environment formArena leftForm leftWitness .total 3 :=
  leftDerives environment

/-- Fixture: addition pairs the two exact child references and applies the centered modulus cap. -/
example (environment : FactEnvironment) :
    witnessArena.DerivesBound environment formArena addForm addWitness .total 8 := by
  exact .add (matrixType := matrixType) (modulus := 17) rfl rfl rfl
    (leftDerives environment) (rightDerives environment)

private theorem branchDerivations (environment : FactEnvironment) :
    BoundWitnessBranchDerivations environment formArena witnessArena .total
      [leftForm, rightForm] [leftWitness, rightWitness] [3, 5] := by
  exact .cons (leftDerives environment) (.cons (rightDerives environment) (.nil .total))

/-- Fixture: selection aligns every form and witness by branch position and takes a maximum. -/
example (environment : FactEnvironment) :
    witnessArena.DerivesBound environment formArena selectForm selectWitness .total 5 := by
  exact .select
    (formBranches := ⟨leftForm, [rightForm]⟩)
    (witnessBranches := ⟨leftWitness, [rightWitness]⟩)
    rfl rfl (branchDerivations environment)

/-- Fixture: the second selected form and witness are certified at the same branch position. -/
example (environment : FactEnvironment) :
    (branchDerivations environment).Contains rightForm rightWitness 5 := by
  exact .tail
    (headDerivation := leftDerives environment)
    (tailDerivations := .cons (rightDerives environment) (.nil .total))
    (.head
      (headDerivation := rightDerives environment)
      (tailDerivations := .nil .total))

private def constructionWire : CoreWireRef := {
  stage := ⟨"symbolic-soundness-construction"⟩
  scope := ⟨[]⟩
  node := ⟨0⟩
  port := 0
}

private def constructedAffineState : SymbolicEvaluationConstructionState :=
  match ({} : SymbolicEvaluationConstructionState).appendScopedMatrixFact {
    wire := constructionWire
    matrixType := some matrixType
    fact := .matrix {
      subject := .ofCoreWire constructionWire
      primary := .affine { terms := [], noiseBound := .constant 3 }
      relations := []
      totalNormBound := .constant 3
    }
  } with
  | .ok state => state
  | .error _ => {}

/-- Fixture: all three roots produced by affine construction derive their exact closed bounds. -/
example (environment : FactEnvironment) :
    constructedAffineState.boundWitnessArena.DerivesBound environment
        constructedAffineState.symbolicFormArena ⟨0⟩ ⟨0⟩ .coefficient 0 ∧
      constructedAffineState.boundWitnessArena.DerivesBound environment
        constructedAffineState.symbolicFormArena ⟨0⟩ ⟨1⟩ .noise 3 ∧
      constructedAffineState.boundWitnessArena.DerivesBound environment
        constructedAffineState.symbolicFormArena ⟨0⟩ ⟨2⟩ .total 3 := by
  exact ⟨.atom rfl rfl .affineCoefficient rfl,
    .atom rfl rfl .affineNoise rfl,
    .atom rfl rfl .affineTotal rfl⟩

/-- Fixture: changing the total root to the coefficient witness cannot derive a total bound. -/
example (environment : FactEnvironment) :
    ¬ constructedAffineState.boundWitnessArena.DerivesBound environment
      constructedAffineState.symbolicFormArena ⟨0⟩ ⟨0⟩ .total 3 := by
  intro derivation
  obtain ⟨entry, lookup, role⟩ := derivation.witness_role
  have expected : constructedAffineState.boundWitnessArena.lookup ⟨0⟩ = some {
      role := .coefficient, witness := .atom .coefficient (.constant 0)
    } := by rfl
  rw [expected] at lookup
  cases lookup
  cases role

/-- Fixture: the same constructed roots yield semantic `Holds` judgments for one evaluation. -/
example
    (environment : FactEnvironment)
    (evaluation : SymbolicMatrixEvaluation environment
      constructedAffineState.symbolicFormArena ⟨0⟩)
    (noTerms : evaluation.terms = [])
    (noiseNorm : Mxx.maxCenteredCoefficientNorm evaluation.noise ≤ 3)
    (totalNorm : Mxx.maxCenteredCoefficientNorm evaluation.value ≤ 3) :
    constructedAffineState.boundWitnessArena.Holds environment
        constructedAffineState.symbolicFormArena evaluation ⟨0⟩ .coefficient 0 ∧
      constructedAffineState.boundWitnessArena.Holds environment
        constructedAffineState.symbolicFormArena evaluation ⟨1⟩ .noise 3 ∧
      constructedAffineState.boundWitnessArena.Holds environment
        constructedAffineState.symbolicFormArena evaluation ⟨2⟩ .total 3 := by
  refine ⟨?_, ?_, ?_⟩
  · constructor
    · exact .atom rfl rfl .affineCoefficient rfl
    · simp [SymbolicMatrixEvaluation.RoleBounded,
        SymbolicMatrixEvaluation.coefficientL1, noTerms]
  · exact ⟨.atom rfl rfl .affineNoise rfl, noiseNorm⟩
  · exact ⟨.atom rfl rfl .affineTotal rfl, totalNorm⟩

end SymbolicEvaluationSoundnessFixtures

end Mxx.Certificate
