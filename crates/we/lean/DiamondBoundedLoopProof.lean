import DiamondProofParameters
import DiamondBoundedLayerProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem producer_initial_integer_state
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions) :
    ∃ (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
      initial = reduceMatrix q n 1 2 row * producer.initialBase +
        reduceMatrix q n 1 inner error ∧
      row 0 1 = (if message then 1 else 0) ∧
      CoeffBound row 1 ∧
      CoeffBound error params.diamond_error_max_coefficient_bound.toNat := by
  obtain ⟨secret, hsecret, hsecretBound⟩ := producer.secretRun.2
  obtain ⟨error, herror, herrorBound⟩ := producer.initialErrorRun.2.2.1
  have hmessage : producer.messageValue = if message then 1 else 0 := by
    rcases producer.messageRun with ⟨position, hposition, hvalue⟩
    cases message <;> fin_cases position <;> simp_all
  let row : ErrorMatrix n 1 2 := fun _ column ↦
    if column = 0 then secret 0 0 else if message then 1 else 0
  have hrow : producer.initialSelector = reduceMatrix q n 1 2 row := by
    funext i j
    fin_cases i
    fin_cases j
    · simpa [concatColumns, reduceMatrix, row, hsecret] using producer.initialSelectorRun 0 0
    · have h := producer.initialSelectorRun 0 1
      cases message <;> simpa [concatColumns, reduceMatrix, row, hmessage] using h
  refine ⟨row, error, ?_, rfl, ?_, herrorBound⟩
  · simpa only [hrow, herror] using producer.initialEquation
  · intro i j coefficient
    fin_cases i
    fin_cases j
    · change (secret 0 0 |>.coeff coefficient).natAbs ≤ 1
      have h := hsecretBound 0 0 coefficient
      omega
    · change ((if message then 1 else 0 : ErrorPoly n).coeff coefficient).natAbs ≤ 1
      have hone := Negacyclic.coeff_root_pow (R := Int) (by decide : 0 < n)
        (0 : Fin n) coefficient
      simp only [Fin.val_zero, pow_zero] at hone
      cases message <;> simp [hone]
      split <;> norm_num

#print axioms producer_initial_integer_state

theorem generated_bounded_injector_loop
    (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (decryptParams : Stage_decrypt.Params)
    (message : Bool) (initial : ExactMatrix q n 1 inner)
    (transitions : Fin transitionCount → ExactMatrix q n inner inner)
    (producer : InjectorRootWitness backend hashModel params message initial transitions)
    (digitBase count : Nat) (hdigits : 0 < digitBase)
    (hbase : (digitBase : Int) = params.diamond_digit_base)
    (hbaseParams : params.diamond_digit_base = decryptParams.diamond_digit_base)
    (hbatch : params.diamond_batch_bits = decryptParams.diamond_batch_bits)
    (hbatchNonneg : 0 ≤ decryptParams.diamond_batch_bits)
    (packed : Fin inputCount → Int)
    (hpacked : ∀ index : Fin inputCount, 0 ≤ packed index ∧ packed index < digitBase)
    (batch : Nat) (hbatchNat : (batch : Int) = params.diamond_batch_bits)
    (bits : Nat → Bool)
    (hpackedBits : ∀ index : Fin inputCount, ∀ bit, bit < batch →
      bits (index.val * batch + bit) = decide ((packed index / (2 ^ bit)) % 2 = 1))
    (P N : Nat → Nat) (hPzero : P 0 = 1)
    (hNzero : N 0 = params.diamond_error_max_coefficient_bound.toNat)
    (hPstep : ∀ layer, P (layer + 1) = n * P layer)
    (hNstep : ∀ layer, N (layer + 1) = 2 * n * P layer *
      params.diamond_error_max_coefficient_bound.toNat +
      inner * n * N layer * params.diamond_preimage_max_coefficient_bound.toNat)
    (states outputs : Fin stateCount → ExactMatrix q n 1 inner)
    (hinitial : ∀ state : Fin stateCount, Stage_decrypt.parallel_generatedRoot_2 backend decryptParams
      state.val initial (states state))
    (hrun : MxxIR.IterRuns
      (fun layer current next ↦ Stage_decrypt.sequential_generatedRoot_8 backend decryptParams
        layer (current, packed, transitions, ()) next) count states outputs) :
    ∃ commonSecret : ErrorPoly n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat count * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = count * DiamondProofParameters.stateCount + state.val ∧ row 0 0 = commonSecret ∧
        outputs state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row (P count) ∧ CoeffBound error (N count) ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret * (if bits (state.val - 1) then 1 else 0) := by
  obtain ⟨samples, hsamples⟩ := producer_integer_digit_samples backend hashModel params
    message initial transitions producer
  let Invariant := fun (layer : Nat) (values : Fin stateCount → ExactMatrix q n 1 inner) ↦
    ∃ commonSecret : ErrorPoly n, ∀ state : Fin stateCount,
      (state.val : Int) ≤ Int.ofNat layer * decryptParams.diamond_batch_bits →
      ∃ (position : Fin basePoolCount) (row : ErrorMatrix n 1 2) (error : ErrorMatrix n 1 inner),
        position.val = layer * DiamondProofParameters.stateCount + state.val ∧ row 0 0 = commonSecret ∧
        values state = reduceMatrix q n 1 2 row * producer.bases position +
          reduceMatrix q n 1 inner error ∧
        CoeffBound row (P layer) ∧ CoeffBound error (N layer) ∧
        reducePoly q n (row 0 1) =
          if state.val = 0 then (if message then 1 else 0)
          else reducePoly q n commonSecret * (if bits (state.val - 1) then 1 else 0)
  have hstart : Invariant 0 states := by
    obtain ⟨row, error, hequation, hmessage, hrowBound, herrorBound⟩ :=
      producer_initial_integer_state backend hashModel params message initial transitions producer
    refine ⟨row 0 0, ?_⟩
    intro state hactive
    have hstate : state = (0 : Fin stateCount) := by
      apply Fin.ext
      dsimp at hactive ⊢
      omega
    subst state
    rcases hinitial 0 with ⟨value, _, _, _, ⟨position, hposition, hvalue⟩, hout⟩
    have hp : position = (⟨1, by decide⟩ : Fin 2) := by
      apply Fin.ext
      dsimp at hposition ⊢
      omega
    subst position
    have hstateValue : states 0 = initial := hout.trans hvalue
    rcases producer.initialBaseRun with ⟨basePosition, hbasePosition, hbaseValue⟩
    refine ⟨basePosition, row, error, ?_, rfl, ?_, ?_, ?_, ?_⟩
    · dsimp at hbasePosition ⊢
      omega
    · exact hstateValue.trans (by simpa only [hbaseValue] using hequation)
    · simpa only [hPzero] using hrowBound
    · simpa only [hNzero] using herrorBound
    · simp only [Fin.val_zero, ite_true, hmessage]
      cases message <;> simp
  apply MxxIR.IterRuns.invariant (Invariant := Invariant) hstart _ hrun
  intro layer current next ih hstep
  obtain ⟨secret, hstates⟩ := ih
  obtain ⟨digit, samplePosition, _, _, hnext⟩ := generated_bounded_injector_layer backend
    hashModel params decryptParams message initial transitions producer samples hsamples
    digitBase layer hdigits hbase hbaseParams hbatch hbatchNonneg packed hpacked
    batch hbatchNat bits hpackedBits current next
    secret (P layer) (N layer) hstates hstep
  refine ⟨secret * samples samplePosition, ?_⟩
  intro state hactive
  obtain ⟨position, row, error, hposition, hrow, hequation, hrowBound, herrorBound, hmeaning⟩ :=
    hnext state
  refine ⟨position, row, error, hposition, hrow, hequation, ?_, ?_, ?_⟩
  · simpa only [hPstep] using hrowBound
  · simpa only [hNstep] using herrorBound
  · have hactiveNat : state.val ≤ (layer + 1) * batch := by
      rw [← hbatch, ← hbatchNat] at hactive
      change (state.val : Int) ≤ ((layer + 1 : Nat) : Int) * (batch : Int) at hactive
      exact_mod_cast hactive
    simpa only [hrow] using hmeaning hactiveNat

#print axioms generated_bounded_injector_loop

end DiamondGeneratedProof
