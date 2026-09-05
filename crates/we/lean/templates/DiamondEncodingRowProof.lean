import DiamondProofParameters
import DiamondIntegerInvariant
import DiamondTransitionProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem selected_boolean_scalar (bit : Bool) (value : ExactMatrix q n 1 1)
    (hrun : select (if bit then 1 else 0) [0, 1] value) :
    value 0 0 = (if bit then 1 else 0 : ExactPoly q n) := by
  rcases hrun with ⟨position, hposition, hvalue⟩
  cases bit <;> fin_cases position <;> simp_all

/-- Semantic second coordinate of the same integer row produced by the bounded
    transition. Packing consistency is a local input fact, not a noise premise. -/
theorem selector_integer_row_encoding
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (witness : InjectorSelectorWitness backend hashModel params slot secret publicMatrix target)
    (row nextRow : ErrorMatrix n 1 2) (commonSecret digitSecret : ErrorPoly n)
    (message : Bool) (bits : Nat → Bool) (batch layer lane : Nat) (digit : Int)
    (hrowFirst : row 0 0 = commonSecret)
    (hnextFirst : nextRow 0 0 = commonSecret * digitSecret)
    (hsecret : secret 0 0 = reducePoly q n digitSecret)
    (hrow : reduceMatrix q n 1 2 nextRow =
      reduceMatrix q n 1 2 row * witness.selector)
    (hbatch : params.diamond_batch_bits.toNat = batch)
    (hstate : witness.state = (lane : Int))
    (hfirstNew : witness.firstNew = ((layer * batch + 1 : Nat) : Int))
    (hdigit : witness.digit = digit)
    (hactive : lane ≤ (layer + 1) * batch)
    (hexisting : lane ≤ layer * batch → reducePoly q n (row 0 1) =
      if lane = 0 then (if message then 1 else 0)
      else reducePoly q n commonSecret * (if bits (lane - 1) then 1 else 0))
    (hbits : ∀ bit, bit < batch →
      bits (layer * batch + bit) = decide ((digit / (2 ^ bit)) % 2 = 1)) :
    reducePoly q n (nextRow 0 1) =
      if lane = 0 then (if message then 1 else 0)
      else reducePoly q n (nextRow 0 0) * (if bits (lane - 1) then 1 else 0) := by
  by_cases hold : lane ≤ layer * batch
  · have hbefore : witness.state < witness.firstNew := by
      rw [hstate, hfirstNew]
      exact_mod_cast Nat.lt_succ_of_le hold
    have h := selector_witness_integer_existing_coordinate backend hashModel params slot
      secret publicMatrix target witness row nextRow hrow hbefore
    rw [hstate, hexisting hold, hsecret] at h
    rw [hnextFirst]
    by_cases hzero : lane = 0
    · simpa [hzero] using h
    · have hzeroInt : (lane : Int) ≠ 0 := by exact_mod_cast hzero
      simpa [hzero, hzeroInt, mul_assoc, mul_left_comm, mul_comm] using h
  · let bit := lane - (layer * batch + 1)
    have hbit : bit < batch := by
      simp only [Nat.add_mul, Nat.one_mul] at hactive
      dsimp [bit]
      omega
    have hlane : lane = layer * batch + 1 + bit := by dsimp [bit]; omega
    have hmatch : witness.state = witness.firstNew + (bit : Int) := by
      rw [hstate, hfirstNew]
      exact_mod_cast hlane
    obtain ⟨value, hselect, hcoord⟩ := selector_witness_integer_new_coordinate backend hashModel
      params slot secret publicMatrix target witness row nextRow bit hrow hmatch
      (by simpa only [hbatch] using hbit)
    have hvalue := selected_boolean_scalar
      (decide ((witness.digit / (2 ^ bit)) % 2 = 1)) value hselect
    have hindex : lane - 1 = layer * batch + bit := by omega
    have hnonzero : lane ≠ 0 := by omega
    rw [hcoord, hvalue, hdigit, hrowFirst, hsecret, hnextFirst]
    simp only [hnonzero, ite_false, hindex, hbits bit hbit, reducePoly_mul]

theorem generated_existing_source_index
    (backend : BackendContext) (params : Stage_decrypt.Params) (layer lane : Nat)
    (index : Int) (hbefore : (lane : Int) < (layer : Int) * params.diamond_batch_bits + 1)
    (hrun : Stage_decrypt.parallel_sequential_generatedRoot_8_5 backend params layer lane
      ((layer : Int) * params.diamond_batch_bits + 1) index) : index = (lane : Int) := by
  rcases hrun with ⟨value, _, _, _, hselect, hout⟩
  have hflag : decide ((layer : Int) * params.diamond_batch_bits + 1 ≤ (lane : Int)) =
      false := decide_eq_false (by omega)
  simp only [Int.ofNat_eq_natCast] at hselect
  simp only [hflag, Bool.false_eq_true, ite_false] at hselect
  rcases hselect with ⟨position, hposition, hvalue⟩
  have hp : position = (0 : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  exact hout.trans hvalue

#print axioms generated_existing_source_index
#print axioms selector_integer_row_encoding

end DiamondGeneratedProof
