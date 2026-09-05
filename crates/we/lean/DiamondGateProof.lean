import Stage_encrypt
import Backend
import DiamondProofParameters

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem active_product_gate (backend : BackendContext) (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (active : Int)
    (left right base output : ExactMatrix q n 1 ell)
    (hactive : (lane : Int) < active)
    (hrun : Stage_encrypt.parallel_sequential_generatedRoot_32_14 backend hashModel
      params layer lane (4, left, right, active, base, ()) output) :
    ∃ digits : ExactMatrix q n ell ell,
      gadgetDecomposeRuns backend params.diamond_gadget_base params.diamond_digit_count
        right digits ∧ output = left * digits := by
  dsimp only [Stage_encrypt.parallel_sequential_generatedRoot_32_14] at hrun
  rcases hrun with ⟨digits, selected, masked, hdecomp, _, _, _, hselect,
    _, _, _, hmask, hout⟩
  have hflag : decide (Int.ofNat lane ≤ active - 1) = true := by
    apply decide_eq_true
    change (lane : Int) ≤ active - 1
    omega
  rw [hflag, if_pos rfl] at hmask
  rcases hselect with ⟨position, hposition, hselected⟩
  have hp : position = (⟨4, by decide⟩ : Fin 6) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  rcases hmask with ⟨position, hposition, hmasked⟩
  have hp : position = (⟨1, by decide⟩ : Fin 2) := by
    apply Fin.ext
    dsimp at hposition ⊢
    omega
  subst position
  refine ⟨digits, hdecomp, ?_⟩
  simpa only [List.get, MxxRuntime.matrixMul] using hout.trans (hmasked.trans hselected)

theorem active_product_gate_bounded (hashModel : HashModel)
    (params : Stage_encrypt.Params) (layer lane : Nat) (active : Int)
    (left right base output : ExactMatrix q n 1 ell)
    (hactive : (lane : Int) < active)
    (hrun : Stage_encrypt.parallel_sequential_generatedRoot_32_14 DiamondBackend.backend
      hashModel params layer lane (4, left, right, active, base, ()) output) :
    ∃ digits : ExactMatrix q n ell ell,
      output = left * digits ∧ PreimageWithin digits D ∧
      digits = regularDecomposeMatrix DiamondBackend.layout0 right ∧
      regularGadgetMatrix DiamondBackend.layout0 *
        regularDecomposeMatrix DiamondBackend.layout0 right = right := by
  obtain ⟨digits, hdecomp, hout⟩ :=
    active_product_gate _ _ _ _ _ _ _ _ _ _ hactive hrun
  rcases hdecomp with ⟨layout, hlookup, _, _, hwidth, hdigits⟩
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlookup.symm
  subst layout
  have hd : digits = regularDecomposeMatrix DiamondBackend.layout0 right := by
    simpa only [castMatrixRows] using hdigits
  subst digits
  refine ⟨_, hout, ?_, rfl, ?_⟩
  · exact regularDecomposeMatrix_bounded DiamondBackend.layout0 right (by decide) (by decide)
  · exact regularGadgetMatrix_reconstruct DiamondBackend.layout0 right (by decide) (by decide)

#print axioms active_product_gate_bounded

end DiamondGeneratedProof
