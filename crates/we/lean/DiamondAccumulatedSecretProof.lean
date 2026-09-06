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
      (fun bit current next ↦ Stage_encrypt.sequential_parallel_generatedRoot_60_22
        backend hashModel params slot bit (current, state, firstNew, secret, digit, ()) next)
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

end DiamondGeneratedProof
