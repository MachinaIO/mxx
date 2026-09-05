import DiamondProofParameters
import DiamondSelectorProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem generated_selector_scan_first_column
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot count : Nat) (digit state firstNew : Int)
    (initial output : ExactMatrix q n 2 2) (secret : ExactMatrix q n 1 1)
    (hinitial : initial 0 0 = secret 0 0 ∧ initial 1 0 = 0)
    (hrun : MxxIR.IterRuns
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_72_21
        backend hashModel params slot bit (current, digit, state, firstNew, secret, ()) next)
      count initial output) : output 0 0 = secret 0 0 ∧ output 1 0 = 0 := by
  apply MxxIR.IterRuns.invariant
    (Invariant := fun _ value ↦ value 0 0 = secret 0 0 ∧ value 1 0 = 0) hinitial _ hrun
  intro bit current next ih hstep
  by_cases hstate : state = firstNew + Int.ofNat bit
  · obtain ⟨_, _, h00, _, h10, _⟩ := generated_selector_match backend hashModel params
      slot bit digit state firstNew current next secret hstate hstep
    exact ⟨h00, h10⟩
  · have heq := generated_selector_no_match backend hashModel params slot bit digit state
      firstNew current next secret hstate hstep
    exact heq.symm ▸ ih

theorem generated_target_shared_secret
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    (slot : Nat) (secret : ExactMatrix q n 1 1)
    (publicMatrix target : ExactMatrix q n 2 inner)
    (hrun : Stage_encrypt.parallel_generatedRoot_72 backend hashModel params slot
      (secret, publicMatrix, ()) target) :
    ∃ (selector : ExactMatrix q n 2 2) (error : ExactMatrix q n 2 inner),
      selector 0 0 = secret 0 0 ∧ selector 1 0 = 0 ∧
      gaussianSample params.diamond_error_sigma params.diamond_error_max_coefficient_bound error ∧
      target = selector * publicMatrix + error := by
  dsimp only [Stage_encrypt.parallel_generatedRoot_72] at hrun
  rcases hrun with ⟨regular, initialZero, initial, selector, error,
    _, hregular, hzero, _, _, _, hselect, _, _, _, _, hscan, herror, htarget⟩
  have hr : regular 0 0 = secret 0 0 ∧ regular 1 0 = 0 := by
    constructor
    · simpa [concatDiagonal] using hregular 0 0
    · simpa [concatDiagonal] using hregular 1 0
  have hz : initialZero 0 0 = secret 0 0 ∧ initialZero 1 0 = 0 := by
    constructor
    · simpa [concatDiagonal] using hzero 0 0
    · simpa [concatDiagonal] using hzero 1 0
  have hi : initial 0 0 = secret 0 0 ∧ initial 1 0 = 0 := by
    rcases hselect with ⟨position, _, hvalue⟩
    fin_cases position
    · exact hvalue.symm ▸ hr
    · exact hvalue.symm ▸ hz
  obtain ⟨h00, h10⟩ := generated_selector_scan_first_column backend hashModel params slot
    _ _ _ _ initial selector secret hi hscan
  exact ⟨selector, error, h00, h10, herror, htarget⟩

#print axioms generated_target_shared_secret

end DiamondGeneratedProof
