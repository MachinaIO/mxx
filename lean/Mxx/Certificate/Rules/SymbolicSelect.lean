import Mxx.Certificate.SymbolicForm
import Mxx.Certificate.Bounds
import Mxx.Toolkit.Norms

namespace Mxx.Certificate

/-- Numeric interpretation of the unified fixed-schema symbolic bound summary. -/
structure EvaluatedMatrixBoundSummary where
  signal : SignalPresence
  coefficientL1Bound : Nat
  noiseBound : Nat
  totalBound : Nat
  deriving Repr

def MatrixBoundSummary.Evaluates
    (environment : Mxx.Ir.ParamEnvironment)
    (summary : MatrixBoundSummary)
    (evaluated : EvaluatedMatrixBoundSummary) : Prop :=
  evaluated.signal = summary.signal ∧
    summary.coefficientL1Bound.evaluate environment = .ok evaluated.coefficientL1Bound ∧
    summary.noiseBound.evaluate environment = .ok evaluated.noiseBound ∧
    summary.totalBound.evaluate environment = .ok evaluated.totalBound

/-- Maximum of a finite list of closed hard-bound expressions. -/
def selectMaximumBounds : List BoundExpr → BoundExpr
  | [] => .constant 0
  | bound :: bounds => .maximum bound (selectMaximumBounds bounds)

theorem selectMaximumBounds_evaluate
    (environment : Mxx.Ir.ParamEnvironment) :
    ∀ {bounds values},
      List.Forall₂ (fun bound value => bound.evaluate environment = .ok value) bounds values →
      (selectMaximumBounds bounds).evaluate environment =
        .ok (Mxx.Toolkit.selectBound values)
  | [], [], .nil => rfl
  | bound :: bounds, value :: values, .cons head tail => by
      have head' :
          BoundExpr.evaluateWithRecurrences environment {} bound = .ok value := by
        simpa [BoundExpr.evaluate] using head
      have tail' :
          BoundExpr.evaluateWithRecurrences environment {} (selectMaximumBounds bounds) =
            .ok (Mxx.Toolkit.selectBound values) := by
        simpa [BoundExpr.evaluate] using selectMaximumBounds_evaluate environment tail
      change
        (do
          let left ← BoundExpr.evaluateWithRecurrences environment {} bound
          let right ←
            BoundExpr.evaluateWithRecurrences environment {} (selectMaximumBounds bounds)
          pure (Nat.max left right)) =
          .ok (Nat.max value (Mxx.Toolkit.selectBound values))
      rw [head', tail']
      rfl

/-- A branch maximum dominates every successfully evaluated branch bound. -/
theorem selectMaximumBounds_contains
    (environment : Mxx.Ir.ParamEnvironment)
    {bounds values : List _}
    (evaluates :
      List.Forall₂ (fun bound value => bound.evaluate environment = .ok value) bounds values)
    {bound : BoundExpr}
    {value : Nat}
    (member : bound ∈ bounds)
    (boundEvaluates : bound.evaluate environment = .ok value) :
    value ≤ Mxx.Toolkit.selectBound values := by
  induction evaluates generalizing bound value with
  | nil => simp at member
  | @cons head headValue tail tailValues headEvaluates tailEvaluates induction =>
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · have : value = headValue := by
          rw [headEvaluates] at boundEvaluates
          exact Except.ok.inj boundEvaluates.symm
        subst value
        exact le_max_left _ _
      · exact le_trans (induction member boundEvaluates) (le_max_right _ _)

private def combineSignal
    (result : SignalPresence)
    (branch : MatrixBoundSummary) : SignalPresence :=
  result.combine branch.signal

theorem foldSignals_from_present (branches : List MatrixBoundSummary) :
    branches.foldl combineSignal .present = .present := by
  induction branches with
  | nil => rfl
  | cons head tail induction =>
      simp only [List.foldl_cons, combineSignal, SignalPresence.combine]
      exact induction

theorem foldSignals_present_of_mem
    (initial : SignalPresence)
    {branches : List MatrixBoundSummary}
    {branch : MatrixBoundSummary}
    (member : branch ∈ branches)
    (present : branch.signal = .present) :
    branches.foldl combineSignal initial = .present := by
  induction branches generalizing initial with
  | nil => simp at member
  | cons head tail induction =>
      simp only [List.mem_cons] at member
      rcases member with rfl | member
      · simp only [present, List.foldl_cons, combineSignal]
        cases initial <;> exact foldSignals_from_present tail
      · simpa only [List.foldl_cons] using
          induction (combineSignal initial head) member

theorem symbolicSelect_signal_present_of_head
    (head : MatrixBoundSummary)
    (tail : List MatrixBoundSummary)
    (present : head.signal = .present) :
    (MatrixBoundSummary.select head tail).signal = .present := by
  simp only [MatrixBoundSummary.select]
  rw [present]
  exact foldSignals_from_present tail

theorem symbolicSelect_signal_present_of_tail
    (head : MatrixBoundSummary)
    {tail : List MatrixBoundSummary}
    {branch : MatrixBoundSummary}
    (member : branch ∈ tail)
    (present : branch.signal = .present) :
    (MatrixBoundSummary.select head tail).signal = .present := by
  simp only [MatrixBoundSummary.select]
  exact foldSignals_present_of_mem head.signal member present

/-- Public semantic spelling of the branch charge used by closed whole-form selection. A
noise-only branch selected alongside any signal branch becomes `zero signal + branchValue`, so
its complete stored-value bound is charged as noise. -/
def symbolicSelectBranchNoise
    (resultSignal : SignalPresence)
    (branch : MatrixBoundSummary) : BoundExpr :=
  match resultSignal, branch.signal with
  | .present, .none => branch.totalBound
  | _, _ => branch.noiseBound

@[simp] theorem symbolicSelectBranchNoise_lifts_noise_only
    (branch : MatrixBoundSummary)
    (noiseOnly : branch.signal = .none) :
    symbolicSelectBranchNoise .present branch = branch.totalBound := by
  simp [symbolicSelectBranchNoise, noiseOnly]

@[simp] theorem symbolicSelectBranchNoise_preserves_signal
    (branch : MatrixBoundSummary)
    (present : branch.signal = .present) :
    symbolicSelectBranchNoise .present branch = branch.noiseBound := by
  simp [symbolicSelectBranchNoise, present]

theorem symbolicSelect_lifts_noise_only_with_signal_head
    {head branch : MatrixBoundSummary}
    (headPresent : head.signal = .present)
    (branchNoiseOnly : branch.signal = .none) :
    symbolicSelectBranchNoise (MatrixBoundSummary.select head []).signal branch =
      branch.totalBound := by
  rw [symbolicSelect_signal_present_of_head head [] headPresent]
  exact symbolicSelectBranchNoise_lifts_noise_only branch branchNoiseOnly

end Mxx.Certificate
