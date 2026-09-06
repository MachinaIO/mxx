import Stage_encrypt
import DiamondProofParameters
import Backend
import Decoder

open Mxx.Primitives MxxRuntime

namespace DiamondGeneratedProof

open DiamondProofParameters

set_option maxRecDepth 8192
set_option maxHeartbeats 1000000

/-- Only values and relations projected from one actual encryption root. -/
structure FinalPublicWitness (backend : BackendContext) (params : Stage_encrypt.Params)
    (decoder key : ExactMatrix q n inner 1) (one : ExactMatrix q n inner ell)
    (digits : ExactMatrix q n ell 1) (half : ExactMatrix q n 1 1)
    (publicInputs : Fin stateCount → ExactMatrix q n 1 ell)
    (publicCircuit : ExactMatrix q n 1 ell) where
  base : ExactMatrix q n 2 inner
  uniform : ExactMatrix q n 1 1
  target : ExactMatrix q n 1 1
  gadget : ExactMatrix q n 1 ell
  decoderTarget : ExactMatrix q n 2 1
  keyTarget : ExactMatrix q n 2 1
  oneTarget : ExactMatrix q n 2 ell
  decompositionRun : gadgetDecomposeRuns backend params.diamond_gadget_base
    params.diamond_digit_count target digits
  gadgetRun : gadgetMatrixRuns backend params.diamond_gadget_base ell gadget
  decoderRows : concatRows (uniform + (publicInputs 0 - publicCircuit) * digits)
    (0 : ExactMatrix q n 1 1) decoderTarget
  keyRows : concatRows uniform half keyTarget
  oneRows : concatRows (publicInputs 0 - gadget) (0 : ExactMatrix q n 1 ell) oneTarget
  decoderEquation : base * decoder = decoderTarget
  keyEquation : base * key = keyTarget
  oneEquation : base * one = oneTarget
  halfEquation : half = matrixPolynomial [MxxIR.roundDiv params.diamond_modulus 2]

theorem generated_final_public_witness
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    {inputs outputs} (rootWitness : Stage_encrypt.generatedRoot.Witness)
    (hbody : Stage_encrypt.generatedRoot.body backend hashModel params inputs outputs rootWitness) :
    ∃ witness : FinalPublicWitness backend params outputs.1 outputs.2.2.1 outputs.2.2.2.1
      outputs.2.2.2.2.2.1 (matrixPolynomial [MxxIR.roundDiv params.diamond_modulus 2])
      outputs.2.2.2.2.1 rootWitness.w_30_0,
      witness.base = rootWitness.w_2_0 ∧ witness.gadget = rootWitness.w_54_0 := by
  dsimp only [Stage_encrypt.generatedRoot.body, Stage_encrypt.generatedRoot.constraints_0,
    Stage_encrypt.generatedRoot.constraints_1] at hbody
  have hpublic : rootWitness.w_9_0 = rootWitness.w_8_0 0 := by
    obtain ⟨i, hi, hv⟩ : familyGetStatic rootWitness.w_8_0 0 rootWitness.w_9_0 := by tauto
    have hiz : i = 0 := Fin.ext (by change i.val = 0; omega)
    simpa [hiz] using hv
  have hd : rootWitness.w_2_0 * rootWitness.w_39_0 = rootWitness.w_38_0 := by
    apply preimageRunsDispatched_equation (by decide) (by decide)
    exact (show preimageRunsDispatched backend rootWitness.w_2_0 rootWitness.w_3_0
      rootWitness.w_38_0 params.diamond_preimage_max_coefficient_bound.toNat
      rootWitness.w_39_0 from by tauto)
  have hk : rootWitness.w_2_0 * rootWitness.w_53_0 = rootWitness.w_52_0 := by
    apply preimageRunsDispatched_equation (by decide) (by decide)
    exact (show preimageRunsDispatched backend rootWitness.w_2_0 rootWitness.w_3_0
      rootWitness.w_52_0 params.diamond_preimage_max_coefficient_bound.toNat
      rootWitness.w_53_0 from by tauto)
  have ho : rootWitness.w_2_0 * rootWitness.w_58_0 = rootWitness.w_57_0 := by
    apply preimageRunsDispatched_equation (by decide) (by decide)
    exact (show preimageRunsDispatched backend rootWitness.w_2_0 rootWitness.w_3_0
      rootWitness.w_57_0 params.diamond_preimage_max_coefficient_bound.toNat
      rootWitness.w_58_0 from by tauto)
  have hdecomp : gadgetDecomposeRuns backend params.diamond_gadget_base
      params.diamond_digit_count rootWitness.w_33_0 rootWitness.w_34_0 := by tauto
  have hg : gadgetMatrixRuns backend params.diamond_gadget_base ell rootWitness.w_54_0 := by tauto
  have hdr : concatRows (rootWitness.w_6_0 +
      (rootWitness.w_8_0 0 - rootWitness.w_30_0) * rootWitness.w_34_0)
      (0 : ExactMatrix q n 1 1) rootWitness.w_38_0 := by rw [← hpublic]; tauto
  have hkr : concatRows rootWitness.w_6_0 (matrixPolynomial [MxxIR.roundDiv params.diamond_modulus 2])
      rootWitness.w_52_0 := by tauto
  have hor : concatRows (rootWitness.w_8_0 0 - rootWitness.w_54_0)
      (0 : ExactMatrix q n 1 ell) rootWitness.w_57_0 := by rw [← hpublic]; tauto
  repeat' obtain ⟨_, hbody⟩ := hbody
  exact ⟨⟨rootWitness.w_2_0, rootWitness.w_6_0, rootWitness.w_33_0,
    rootWitness.w_54_0, rootWitness.w_38_0, rootWitness.w_52_0, rootWitness.w_57_0,
    hdecomp, hg, hdr, hkr, hor, hd, hk, ho, rfl⟩, rfl, rfl⟩

/-- Canonical regular decomposition and integer digit bound use the exact registered
layout that also constructs the public gadget. -/
theorem final_public_digits (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit) :
    w.gadget * digits = w.target ∧
      ∃ lift : ErrorMatrix n ell 1, digits = reduceMatrix q n ell 1 lift ∧ CoeffBound lift D := by
  obtain ⟨layout, hlayout, _, _, hwidth, hd, _⟩ := w.decompositionRun
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlayout.symm
  subst layout
  have hd' : digits = regularDecomposeMatrix DiamondBackend.layout0 w.target := by
    simpa [castMatrixRows] using hd
  obtain ⟨layout, hlayout, _, _, hwidth, hg⟩ := w.gadgetRun
  have hl : layout = DiamondBackend.layout0 := by
    simpa [DiamondBackend.backend] using hlayout.symm
  subst layout
  have hg' : w.gadget = regularGadgetMatrix (n := n) DiamondBackend.layout0 := by
    simpa [castMatrixColumns] using hg
  constructor
  · exact (congrArg₂ (fun g d ↦ g * d) hg' hd').trans
      (regularGadgetMatrix_reconstruct _ _ (by decide) (by decide) (by decide))
  · have hb := regularDecomposeMatrix_bounded DiamondBackend.layout0 w.target
      (by decide) (by decide)
    obtain ⟨lift, hlift, hbound⟩ := hb
    exact ⟨lift, hd'.trans hlift, hbound⟩

theorem final_selector_rows {q n columns : Nat}
    (selector : ExactMatrix q n 1 2) (left right : ExactMatrix q n 1 columns)
    (target : ExactMatrix q n 2 columns) (hrows : concatRows left right target)
    (secret message : ExactMatrix q n 1 1)
    (hs : secret 0 0 = selector 0 0) (hm : message 0 0 = selector 0 1) :
    selector * target = secret * left + message * right := by
  funext i j
  fin_cases i
  have h0 := hrows 0 j
  have h1 := hrows 1 j
  simp [Matrix.mul_apply, Fin.sum_univ_two, h0, h1, hs, hm]

/-- The final public cancellation follows from the actual three sampled preimages.
The circuit term is the accepting Boolean encoding at the same secret and gadget. -/
theorem final_public_cancellation (backend : BackendContext) (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness backend params decoder key one digits half publicInputs publicCircuit)
    (selector : ExactMatrix q n 1 2) :
    let secret : ExactMatrix q n 1 1 := fun _ _ ↦ selector 0 0
    let message : ExactMatrix q n 1 1 := fun _ _ ↦ selector 0 1
    selector * (w.base * decoder) + message * half =
      selector * (w.base * key) +
        (selector * (w.base * one) - (secret * publicCircuit - secret * w.gadget)) * digits := by
  dsimp only
  rw [w.decoderEquation, w.keyEquation, w.oneEquation]
  rw [final_selector_rows selector _ _ _ w.decoderRows
      (fun _ _ ↦ selector 0 0) (fun _ _ ↦ selector 0 1) rfl rfl,
    final_selector_rows selector _ _ _ w.keyRows
      (fun _ _ ↦ selector 0 0) (fun _ _ ↦ selector 0 1) rfl rfl,
    final_selector_rows selector _ _ _ w.oneRows
      (fun _ _ ↦ selector 0 0) (fun _ _ ↦ selector 0 1) rfl rfl]
  simp only [Matrix.mul_zero, add_zero, Matrix.mul_add, Matrix.mul_sub, Matrix.sub_mul,
    Matrix.mul_assoc]
  abel

/-- Negative rounded half is floor(q/2) modulo q, for either parity. -/
theorem final_negative_rounded_half (modulus dimension : Nat) :
    -((MxxIR.roundDiv (modulus : Int) 2 : Int) : ExactPoly modulus dimension) =
      (((modulus / 2 : Nat) : Int) : ExactPoly modulus dimension) := by
  have hr : MxxIR.roundDiv (modulus : Int) 2 = (((modulus + 1) / 2 : Nat) : Int) := by
    unfold MxxIR.roundDiv
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num)]
    omega
  rw [hr]
  have hi : -(((modulus + 1) / 2 : Nat) : Int) =
      ((modulus / 2 : Nat) : Int) - (modulus : Int) := by omega
  have hz := congrArg (fun z : Int => (z : ZMod modulus)) hi
  simp only [Int.cast_neg, Int.cast_sub, Int.cast_natCast, ZMod.natCast_self,
    sub_zero] at hz
  simpa only [map_neg, map_natCast, Int.cast_natCast] using
    congrArg (algebraMap (ZMod modulus) (ExactPoly modulus dimension)) hz

/-- The encrypted negative rounded half uses the exact decoder message center. -/
theorem final_public_message_center (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit) (hq : params.diamond_modulus = (q : Int)) (message : Bool) :
    -(if message then half else 0) =
      matrixPolynomial [(MxxWe.messageCenter q message : Int)] := by
  rw [w.halfEquation, hq]
  cases message
  · simp only [MxxWe.messageCenter, Bool.false_eq_true, ↓reduceIte, Nat.cast_zero]
    funext i j
    simp [MxxRuntime.matrixPolynomial]
  · simp only [↓reduceIte, MxxWe.messageCenter, MxxWe.half]
    funext i j
    change -((MxxIR.roundDiv (q : Int) 2 : Int) + _ * 0 : ExactPoly q n) =
      ((q / 2 : Nat) : Int) + _ * 0
    simp only [mul_zero, add_zero]
    exact final_negative_rounded_half q n

#print axioms generated_final_public_witness
#print axioms final_public_digits
#print axioms final_public_cancellation
#print axioms final_public_message_center

end DiamondGeneratedProof
