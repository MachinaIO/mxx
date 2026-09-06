import DiamondBooleanGateProof

open Mxx.Primitives MxxRuntime
open DiamondProofParameters

namespace DiamondGeneratedProof

theorem active_product_gate (backend : BackendContext)
    (params : Stage_encrypt.Params) (lane : Nat) (active : Int)
    (left right base output : ExactMatrix q n 1 ell)
    (hactive : (lane : Int) < active)
    (hrun : ∃ digits selected,
      gadgetDecomposeRuns backend params.diamond_gadget_base params.diamond_digit_count
        right digits ∧
      MxxRuntime.select 4 [base - base, base, left, base - left, left * digits,
        left + right - matrixMulScalarRight (left * digits)
          (matrixPolynomial [2] : ExactMatrix q n 1 1)] selected ∧
      MxxRuntime.select (if decide (Int.ofNat lane ≤ active - 1) then 1 else 0)
        [base - base, selected] output) :
    ∃ digits : ExactMatrix q n ell ell,
      gadgetDecomposeRuns backend params.diamond_gadget_base params.diamond_digit_count
        right digits ∧ output = left * digits := by
  obtain ⟨digits, hdecomp, hout⟩ := generated_public_gate_selection backend params lane
    active (⟨4, by decide⟩ : Fin 6) left right base output hactive hrun
  exact ⟨digits, hdecomp, hout⟩

theorem active_product_gate_bounded
    (params : Stage_encrypt.Params) (lane : Nat) (active : Int)
    (left right base output : ExactMatrix q n 1 ell)
    (hactive : (lane : Int) < active)
    (hrun : ∃ digits selected,
      gadgetDecomposeRuns DiamondBackend.backend params.diamond_gadget_base params.diamond_digit_count
        right digits ∧
      MxxRuntime.select 4 [base - base, base, left, base - left, left * digits,
        left + right - matrixMulScalarRight (left * digits)
          (matrixPolynomial [2] : ExactMatrix q n 1 1)] selected ∧
      MxxRuntime.select (if decide (Int.ofNat lane ≤ active - 1) then 1 else 0)
        [base - base, selected] output) :
    ∃ digits : ExactMatrix q n ell ell,
      output = left * digits ∧ PreimageWithin digits D ∧
      digits = regularDecomposeMatrix DiamondBackend.layout0 right ∧
      regularGadgetMatrix DiamondBackend.layout0 *
        regularDecomposeMatrix DiamondBackend.layout0 right = right := by
  obtain ⟨digits, hdecomp, hout⟩ :=
    active_product_gate DiamondBackend.backend params lane active left right base output hactive hrun
  rcases hdecomp with ⟨layout, hlookup, _, _, hwidth, hdigits, _⟩
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlookup.symm
  subst layout
  have hd : digits = regularDecomposeMatrix DiamondBackend.layout0 right := by
    simpa [castMatrixRows] using hdigits
  refine ⟨digits, hout, ?_, hd, ?_⟩
  · rw [hd]
    exact regularDecomposeMatrix_bounded DiamondBackend.layout0 right (by decide) (by decide)
  · exact regularGadgetMatrix_reconstruct DiamondBackend.layout0 right (by decide) (by decide) (by decide)

#print axioms active_product_gate_bounded

end DiamondGeneratedProof
