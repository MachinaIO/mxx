import ExponentAutomorphism
import ExponentTransfers
import ExponentRefreshRoutes

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- One balanced level preserves each segment's sum. The executed odd-tail
branch carries its left member unchanged, so the wrapped right read contributes
nothing. Flat graph families instantiate these functions with their segment offsets. -/
theorem segmented_pairwise_sum {A : Type} [AddCommMonoid A] {segments width : Nat}
    (values : Fin segments → Fin width → A)
    (pairs : Fin segments → Fin ((width + 1) / 2) → A)
    (equation : ∀ segment i, pairs segment i =
      if hright : 2 * i.val + 1 < width then
        values segment ⟨2 * i.val, by omega⟩ +
          values segment ⟨2 * i.val + 1, hright⟩
      else values segment ⟨2 * i.val, by omega⟩) :
    ∀ segment, ∑ i, pairs segment i = ∑ j, values segment j := by
  intro segment
  apply pairwise_sum (values segment) (pairs segment)
  intro i
  have hleft : 2 * i.val < width := by omega
  rw [equation]
  by_cases hright : 2 * i.val + 1 < width
  · simp [hleft, hright]
  · simp [hleft, hright]

/-- Deterministic aggregation retains the actual modular error witnesses. -/
theorem approximate_sum {q n rows columns : Nat} {α : Type} [DecidableEq α]
    (members : Finset α) (actual ideal : α → ExactMatrix q n rows columns)
    (bound : α → Nat) (h : ∀ i ∈ members, Approx (actual i) (ideal i) (bound i)) :
    Approx (∑ i ∈ members, actual i) (∑ i ∈ members, ideal i) (∑ i ∈ members, bound i) := by
  induction members using Finset.induction_on with
  | empty =>
    refine ⟨0, ?_, coeffBound_zero⟩
    simp only [Finset.sum_empty, zero_add]
    ext row column
    exact (map_zero (reducePoly q n)).symm
  | @insert i members hi ih =>
    obtain ⟨ei, hei, hbi⟩ := h i (Finset.mem_insert_self _ _)
    obtain ⟨es, hes, hbs⟩ := ih (fun j hj => h j (Finset.mem_insert_of_mem hj))
    refine ⟨ei + es, ?_, ?_⟩
    · simp only [Finset.sum_insert hi, hei, hes, reduceMatrix_add]
      abel
    · simpa only [Finset.sum_insert hi] using coeffBound_add hbi hbs

/-- One nonidentity readout branch, matching `aligned + switch * raw` in
`noise_refresh/readout.rs`. The setup equation identifies the key-switch
residual explicitly; no branch noise bound is an input premise. -/
theorem readout_branch_transfer
    {q log rows columns bound cutoff actionGain helperGain targetPlainGain : Nat}
    (hq : 1 < q) {sigma : Int}
    {actual ideal raw mask maskIdeal branchIdeal : ExactMatrix q (2 ^ log) rows columns}
    {action helperAction : ErrorMatrix (2 ^ log) columns columns}
    {switchError maskError : ErrorMatrix (2 ^ log) rows columns}
    {message : ErrorPoly (2 ^ log)}
    (hinput : Approx actual ideal bound)
    (hrun : MxxRuntime.ringAutomorphismRuns sigma actual raw)
    (hmask : mask = maskIdeal + reduceMatrix q (2 ^ log) rows columns maskError)
    (hsetup : MxxRuntime.ringAutomorphism sigma.toNat ideal *
        reduceMatrix q (2 ^ log) columns columns action +
        maskIdeal * reduceMatrix q (2 ^ log) columns columns helperAction =
      branchIdeal + reduceMatrix q (2 ^ log) rows columns (message • switchError))
    (hswitch : CoeffBound switchError cutoff) (hmaskError : CoeffBound maskError cutoff)
    (hmessage : polynomialL1 message ≤ targetPlainGain)
    (haction : ActionGain action actionGain) (hhelper : ActionGain helperAction helperGain) :
    Approx (mask * reduceMatrix q (2 ^ log) columns columns helperAction +
      raw * reduceMatrix q (2 ^ log) columns columns action) branchIdeal
      (actionGain * bound + (targetPlainGain + helperGain) * cutoff) := by
  have hraw := approx_ringAutomorphismRuns hq hinput hrun
  obtain ⟨error, heq, hb⟩ := approximate_right_action (by positivity) hraw haction
  refine ⟨error + (message • switchError + maskError * helperAction), ?_, ?_⟩
  · rw [heq, hmask, Matrix.add_mul, ← reduceMatrix_mul, reduceMatrix_add, reduceMatrix_add]
    have := hsetup
    abel_nf at this ⊢
    simpa only [add_assoc] using congrArg (fun x => x +
      (reduceMatrix q (2 ^ log) rows columns (maskError * helperAction) +
        reduceMatrix q (2 ^ log) rows columns error)) this
  · have hm := left_polynomial_bound (by positivity) hswitch hmessage
    have hh := right_action_bound (by positivity) hmaskError hhelper
    convert coeffBound_add hb (coeffBound_add hm hh) using 1
    ring

/-- Exact natural-number recurrence used by the worst readout simulator. -/
def normalizedTraceBound {α : Type} [DecidableEq α] (members : Finset α)
    (inputBound cutoff targetGain identityGain normalizationGain targetPlainGain : Nat)
    (actionGain helperGain : α → Nat) : Nat :=
  normalizationGain * (identityGain * (targetGain * inputBound) +
    ∑ i ∈ members, (actionGain i * (targetGain * inputBound) +
      (targetPlainGain + helperGain i) * cutoff))

/-- The complete target/trace/normalization sequence. Every branch consumes its
actual automorphism execution, online mask equation, and setup error equation.
The identity branch is the actual identity action (`identityGain = 1` in Rust). -/
theorem normalized_trace_transfer
    {q log rows columns inputBound cutoff targetGain identityGain normalizationGain
      targetPlainGain : Nat} {α : Type} [DecidableEq α]
    (hq : 1 < q) (members : Finset α)
    {source ideal : ExactMatrix q (2 ^ log) rows columns}
    (target identity normalization : ErrorMatrix (2 ^ log) columns columns)
    (sigma : α → Int)
    (raw mask maskIdeal branchIdeal : α → ExactMatrix q (2 ^ log) rows columns)
    (action helperAction : α → ErrorMatrix (2 ^ log) columns columns)
    (switchError maskError : α → ErrorMatrix (2 ^ log) rows columns)
    (message : α → ErrorPoly (2 ^ log)) (actionGain helperGain : α → Nat)
    (hsource : Approx source ideal inputBound)
    (htarget : ActionGain target targetGain) (hidentity : ActionGain identity identityGain)
    (hnormalization : ActionGain normalization normalizationGain)
    (hrun : ∀ i ∈ members, MxxRuntime.ringAutomorphismRuns (sigma i)
      (source * reduceMatrix q (2 ^ log) columns columns target) (raw i))
    (hmask : ∀ i ∈ members, mask i = maskIdeal i +
      reduceMatrix q (2 ^ log) rows columns (maskError i))
    (hsetup : ∀ i ∈ members,
      MxxRuntime.ringAutomorphism (sigma i).toNat
        (ideal * reduceMatrix q (2 ^ log) columns columns target) *
          reduceMatrix q (2 ^ log) columns columns (action i) +
        maskIdeal i * reduceMatrix q (2 ^ log) columns columns (helperAction i) =
      branchIdeal i + reduceMatrix q (2 ^ log) rows columns (message i • switchError i))
    (hswitch : ∀ i ∈ members, CoeffBound (switchError i) cutoff)
    (hmaskError : ∀ i ∈ members, CoeffBound (maskError i) cutoff)
    (hmessage : ∀ i ∈ members, polynomialL1 (message i) ≤ targetPlainGain)
    (haction : ∀ i ∈ members, ActionGain (action i) (actionGain i))
    (hhelper : ∀ i ∈ members, ActionGain (helperAction i) (helperGain i)) :
    Approx
      ((source * reduceMatrix q (2 ^ log) columns columns target *
          reduceMatrix q (2 ^ log) columns columns identity +
        ∑ i ∈ members, (mask i * reduceMatrix q (2 ^ log) columns columns (helperAction i) +
          raw i * reduceMatrix q (2 ^ log) columns columns (action i))) *
        reduceMatrix q (2 ^ log) columns columns normalization)
      ((ideal * reduceMatrix q (2 ^ log) columns columns target *
          reduceMatrix q (2 ^ log) columns columns identity + ∑ i ∈ members, branchIdeal i) *
        reduceMatrix q (2 ^ log) columns columns normalization)
      (normalizedTraceBound members inputBound cutoff targetGain identityGain normalizationGain
        targetPlainGain actionGain helperGain) := by
  have ht := approximate_right_action (by positivity) hsource htarget
  have hi := approximate_right_action (by positivity) ht hidentity
  have hb (i : α) (him : i ∈ members) := readout_branch_transfer hq ht
    (hrun i him) (hmask i him) (hsetup i him) (hswitch i him) (hmaskError i him)
    (hmessage i him) (haction i him) (hhelper i him)
  have hs := approximate_sum members _ _ _ hb
  obtain ⟨ei, hei, hbi⟩ := hi
  obtain ⟨es, hes, hbs⟩ := hs
  apply approximate_right_action (by positivity) _ hnormalization
  refine ⟨ei + es, ?_, coeffBound_add hbi hbs⟩
  rw [hei, hes, reduceMatrix_add]
  abel

end MxxExponentLut
