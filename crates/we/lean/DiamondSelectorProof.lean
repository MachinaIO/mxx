import DiamondProofParameters
import Stage_encrypt

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_selector_no_match
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot bit : Nat) (digit state firstNew : Int)
    (prior output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hstate : state ≠ firstNew + Int.ofNat bit)
    (hrun : Stage_encrypt.sequential_parallel_generatedRoot_72_21 backend hashModel params
      slot bit (prior, digit, state, firstNew, secret, ()) output) : output = prior := by
  dsimp only [Stage_encrypt.sequential_parallel_generatedRoot_72_21] at hrun
  rcases hrun with ⟨bitValue, top, special, selected, _, _, _, _, _, _, _, _, _, _,
    hselect, hout⟩
  have hflag : decide (state = firstNew + Int.ofNat bit) = false := decide_eq_false hstate
  rw [hflag, if_neg (by decide)] at hselect
  rcases hselect with ⟨position, hposition, hvalue⟩
  have hp : position = (⟨0, by decide⟩ : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  exact hout.trans hvalue

theorem generated_selector_match
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot bit : Nat) (digit state firstNew : Int)
    (prior output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hstate : state = firstNew + Int.ofNat bit)
    (hrun : Stage_encrypt.sequential_parallel_generatedRoot_72_21 backend hashModel params
      slot bit (prior, digit, state, firstNew, secret, ()) output) :
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((digit / (2 ^ bit)) % 2 = 1) then 1 else 0) [0, 1] bitValue ∧
      output 0 0 = secret 0 0 ∧ output 0 1 = secret 0 0 * bitValue 0 0 ∧
      output 1 0 = 0 ∧ output 1 1 = 0 := by
  dsimp only [Stage_encrypt.sequential_parallel_generatedRoot_72_21] at hrun
  rcases hrun with ⟨bitValue, top, special, selected, _, _, _, _, hbit, htop, hspecial,
    _, _, _, hselect, hout⟩
  have hflag : decide (state = firstNew + Int.ofNat bit) = true := decide_eq_true hstate
  rw [hflag, if_pos rfl] at hselect
  rcases hselect with ⟨position, hposition, hvalue⟩
  have hp : position = (⟨1, by decide⟩ : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  have ho : output = special := hout.trans hvalue
  rw [ho]
  refine ⟨bitValue, ?_, ?_, ?_, ?_, ?_⟩
  · exact hbit
  · simpa [concatRows, concatColumns] using (hspecial 0 0).trans (htop 0 0)
  · simpa [concatRows, concatColumns, matrixMulScalarLeft] using
      (hspecial 0 1).trans (htop 0 1)
  · simpa [concatRows] using hspecial 1 0
  · simpa [concatRows] using hspecial 1 1

theorem generated_selector_scan_existing
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hstate : state < firstNew)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) : output = initial := by
  apply MxxIR.IterRuns.invariant (Invariant := fun _ value ↦ value = initial) rfl _ hrun
  intro bit current next hcurrent hstep
  have hnomatch : state ≠ firstNew + Int.ofNat bit := by
    change state ≠ firstNew + (bit : Int)
    omega
  exact (generated_selector_no_match backend hashModel params slot bit digit state firstNew
    current next secret hnomatch hstep).trans hcurrent

#print axioms generated_selector_scan_existing

theorem generated_selector_scan_new
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count targetBit : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hstate : state = firstNew + Int.ofNat targetBit) (hbit : targetBit < count)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) :
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((digit / (2 ^ targetBit)) % 2 = 1) then 1 else 0) [0, 1] bitValue ∧
      output 0 0 = secret 0 0 ∧ output 0 1 = secret 0 0 * bitValue 0 0 ∧
      output 1 0 = 0 ∧ output 1 1 = 0 := by
  let Post := fun (value : ExactMatrix q n 2 2) ↦
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((digit / (2 ^ targetBit)) % 2 = 1) then 1 else 0) [0, 1] bitValue ∧
      value 0 0 = secret 0 0 ∧ value 0 1 = secret 0 0 * bitValue 0 0 ∧
      value 1 0 = 0 ∧ value 1 1 = 0
  have hinvariant : count ≤ targetBit ∨ Post output := by
    apply MxxIR.IterRuns.invariant
      (Invariant := fun i value ↦ i ≤ targetBit ∨ Post value) (Or.inl (Nat.zero_le _)) _ hrun
    intro i current next ih hstep
    by_cases heq : i = targetBit
    · subst i
      exact Or.inr (generated_selector_match backend hashModel params slot targetBit
        digit state firstNew current next secret hstate hstep)
    · by_cases hbefore : i < targetBit
      · exact Or.inl (by omega)
      · have hpost : Post current := ih.resolve_left (by omega)
        have hnomatch : state ≠ firstNew + Int.ofNat i := by
          change state = firstNew + (targetBit : Int) at hstate
          change state ≠ firstNew + (i : Int)
          omega
        have hnext := generated_selector_no_match backend hashModel params slot i digit
          state firstNew current next secret hnomatch hstep
        exact Or.inr (hnext.symm ▸ hpost)
  exact hinvariant.resolve_left (by omega)

#print axioms generated_selector_scan_new

theorem generated_existing_selector_action
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2)
    (secret tail : ExactMatrix q n 1 1) (row : ExactMatrix q n 1 2)
    (hstate : state < firstNew) (hdiagonal : concatDiagonal secret tail initial)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) :
    (row * output) 0 0 = row 0 0 * secret 0 0 ∧
      (row * output) 0 1 = row 0 1 * tail 0 0 := by
  rw [generated_selector_scan_existing backend hashModel params slot count digit state
    firstNew initial output secret hstate hrun]
  unfold concatDiagonal at hdiagonal
  constructor
  · simp [Matrix.mul_apply, hdiagonal]
  · simp [Matrix.mul_apply, Fin.sum_univ_two, hdiagonal]

theorem generated_new_selector_action
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count targetBit : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2)
    (secret : ExactMatrix q n 1 1) (row : ExactMatrix q n 1 2)
    (hstate : state = firstNew + Int.ofNat targetBit) (hbit : targetBit < count)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) :
    ∃ bitValue : ExactMatrix q n 1 1,
      select (if decide ((digit / (2 ^ targetBit)) % 2 = 1) then 1 else 0) [0, 1] bitValue ∧
      (row * output) 0 0 = row 0 0 * secret 0 0 ∧
      (row * output) 0 1 = (row 0 0 * secret 0 0) * bitValue 0 0 := by
  obtain ⟨bitValue, hselect, h00, h01, h10, h11⟩ :=
    generated_selector_scan_new backend hashModel params slot count targetBit digit state
      firstNew initial output secret hstate hbit hrun
  refine ⟨bitValue, hselect, ?_, ?_⟩
  · simp [Matrix.mul_apply, Fin.sum_univ_two, h00, h10]
  · simp [Matrix.mul_apply, Fin.sum_univ_two, h01, h11, mul_assoc]

#print axioms generated_new_selector_action

end DiamondGeneratedProof
