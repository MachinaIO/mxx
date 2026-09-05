import DiamondFinalPublicProof
import DiamondProofParameters
import DiamondFinalDecryptionProof

open Mxx.Primitives MxxRuntime

namespace DiamondGeneratedProof

open DiamondProofParameters

set_option maxRecDepth 8192
set_option maxHeartbeats 1000000

theorem final_sampled_preimage_bound {q n rows inner columns P : Nat}
    {backend : BackendContext} {base : ExactMatrix q n rows inner}
    {trapdoor : TrapdoorValue (ExactMatrix q n rows inner) Unit}
    {target : ExactMatrix q n rows columns} {preimage : ExactMatrix q n inner columns}
    (hkind : trapdoor.kind = .sampledSecret)
    (hrun : preimageRunsDispatched backend base trapdoor target P preimage) :
    PreimageWithin preimage P := by
  rcases hrun.2 with h | h
  · exact preimageRuns_bounded h
  · cases hkind.symm.trans h.1

/-- All three final preimages use the sampled trapdoor selected from this actual root's
sampled base pool. Their integer cutoffs therefore apply without a public-gadget fallback. -/
theorem generated_final_preimages_bounded
    (backend : BackendContext) (hashModel : HashModel) (params : Stage_encrypt.Params)
    {inputs outputs} (hrun : Stage_encrypt.generatedRoot backend hashModel params inputs outputs) :
    PreimageWithin outputs.1 params.diamond_preimage_max_coefficient_bound.toNat ∧
    PreimageWithin outputs.2.2.1 params.diamond_preimage_max_coefficient_bound.toNat ∧
    PreimageWithin outputs.2.2.2.1 params.diamond_preimage_max_coefficient_bound.toNat := by
  dsimp only [Stage_encrypt.generatedRoot] at hrun
  rcases hrun with ⟨w0, w1, t1, w2, t2, base, trapdoor, w6, uniform, w8, publicInputs,
    publicOne, w19, w20, w26, w27, w32, w35, publicCircuit, w38, target, digits,
    decoderTarget, decoder, w46, w51, w52, w53, w55, keyTarget, key, gadget,
    oneTarget, one, w65, w66, t66, w67, w68, w69, w70, w71, w72, w73,
    w74, w75, t75, w76, w77, w78, h⟩
  have hsample (i : Fin basePoolCount) : (t1 i).kind = .sampledSecret := by
    have hr : Stage_encrypt.parallel_generatedRoot_1 backend hashModel params i ()
        (w1 i, t1 i, ()) := by tauto
    obtain ⟨td, pub, htd, hout⟩ := hr
    have ht : t1 i = td := congrArg (fun o ↦ o.2.1) hout
    exact ht ▸ trapdoorSample_sampled htd
  have ht : trapdoor.kind = .sampledSecret := by
    obtain ⟨i, _, hi⟩ : familyGetStatic t2 0 trapdoor := by tauto
    have hr : Stage_encrypt.parallel_generatedRoot_2 backend hashModel params i
        (w0 i, w1, t1, ()) (w2 i, t2 i, ()) := by tauto
    obtain ⟨pub, td, _, _, _, _, _, ⟨j, _, hj⟩, hout⟩ := hr
    have htd : t2 i = td := congrArg (fun o ↦ o.2.1) hout
    rw [hi, htd, hj]
    exact hsample j
  have hd : PreimageWithin decoder params.diamond_preimage_max_coefficient_bound.toNat := by
    apply final_sampled_preimage_bound ht
    exact (show preimageRunsDispatched backend base trapdoor decoderTarget
      params.diamond_preimage_max_coefficient_bound.toNat decoder from by tauto)
  have hk : PreimageWithin key params.diamond_preimage_max_coefficient_bound.toNat := by
    apply final_sampled_preimage_bound ht
    exact (show preimageRunsDispatched backend base trapdoor keyTarget
      params.diamond_preimage_max_coefficient_bound.toNat key from by tauto)
  have ho : PreimageWithin one params.diamond_preimage_max_coefficient_bound.toNat := by
    apply final_sampled_preimage_bound ht
    exact (show preimageRunsDispatched backend base trapdoor oneTarget
      params.diamond_preimage_max_coefficient_bound.toNat one from by tauto)
  repeat' obtain ⟨_, h⟩ := h
  exact ⟨hd, hk, ho⟩

/-- A local state encoding multiplied by its actual bounded preimage supplies an integer
error witness; no approximation of the projected output is assumed. -/
theorem final_state_project {q n inner columns B P : Nat} (hn : 0 < n)
    {state ideal : ExactMatrix q n 1 inner} {preimage : ExactMatrix q n inner columns}
    (hstate : Approx state ideal B) (hpreimage : PreimageWithin preimage P) :
    Approx (state * preimage) (ideal * preimage) (inner * n * B * P) := by
  obtain ⟨error, heq, hbound⟩ := hstate
  obtain ⟨lift, hlift, hliftBound⟩ := hpreimage
  refine ⟨error * lift, ?_, coeffBound_mul hn hbound hliftBound⟩
  rw [heq, Matrix.add_mul]
  exact congrArg (fun e ↦ ideal * preimage + e)
    ((congrArg (fun p ↦ reduceMatrix q n 1 inner error * p) hlift).trans
      (reduceMatrix_mul q n 1 inner columns error lift).symm)

theorem final_actual_encodings (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit) (selector : ExactMatrix q n 1 2)
    (state : ExactMatrix q n 1 inner) {B P : Nat}
    (hstate : Approx state (selector * w.base) B)
    (hd : PreimageWithin decoder P) (hk : PreimageWithin key P)
    (ho : PreimageWithin one P) :
    Approx (state * decoder) (selector * w.decoderTarget) (projection * B * P) ∧
    Approx (state * key) (selector * w.keyTarget) (projection * B * P) ∧
    Approx (state * one) (selector * w.oneTarget) (projection * B * P) := by
  refine ⟨?_, ?_, ?_⟩
  · simpa only [Matrix.mul_assoc, w.decoderEquation] using
      final_state_project (by decide : 0 < n) hstate hd
  · simpa only [Matrix.mul_assoc, w.keyEquation] using
      final_state_project (by decide : 0 < n) hstate hk
  · simpa only [Matrix.mul_assoc, w.oneEquation] using
      final_state_project (by decide : 0 < n) hstate ho

/-- The local same-base injector encoding and accepting-circuit encoding imply the actual
final cancellation bound. Root linkage and acceptance induction supply these two local
premises; the output encoding equations and final residual are derived here. -/
theorem final_residual_from_state (params : Stage_encrypt.Params)
    {decoder key one digits half publicInputs publicCircuit}
    (w : FinalPublicWitness DiamondBackend.backend params decoder key one digits half
      publicInputs publicCircuit) (hq : params.diamond_modulus = (q : Int))
    (selector : ExactMatrix q n 1 2) (message : Bool)
    (hmessage : selector 0 1 = if message then 1 else 0)
    (state : ExactMatrix q n 1 inner) (circuit : ExactMatrix q n 1 ell)
    {B P BH : Nat} (hstate : Approx state (selector * w.base) B)
    (hd : PreimageWithin decoder P) (hk : PreimageWithin key P)
    (ho : PreimageWithin one P)
    (hcircuit : Approx circuit
      (matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1) publicCircuit -
        matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1) w.gadget) BH) :
    Approx (state * decoder - (state * key + (state * one - circuit) * digits))
      (matrixPolynomial [(MxxWe.messageCenter q message : Int)])
      (2 * (projection * B * P) + a * (projection * B * P + BH)) := by
  let secret : ExactMatrix q n 1 1 := fun _ _ ↦ selector 0 0
  let msg : ExactMatrix q n 1 1 := fun _ _ ↦ selector 0 1
  obtain ⟨ed, hed, hbd⟩ := final_state_project (by decide : 0 < n) hstate hd
  obtain ⟨ek, hek, hbk⟩ := final_state_project (by decide : 0 < n) hstate hk
  obtain ⟨eo, heo, hbo⟩ := final_state_project (by decide : 0 < n) hstate ho
  obtain ⟨ec, hec, hbc⟩ := hcircuit
  obtain ⟨_, dlift, hdlift, hdliftBound⟩ := final_public_digits params w
  have hm : msg * half = if message then half else 0 := by
    funext i j
    fin_cases i
    fin_cases j
    cases message <;> simp [Matrix.mul_apply, msg, hmessage]
  have hcenter : -(msg * half) =
      matrixPolynomial [(MxxWe.messageCenter q message : Int)] :=
    (congrArg Neg.neg hm).trans (final_public_message_center params w hq message)
  have hdec : state * decoder =
      ((selector * w.base) * decoder + msg * half) + -(msg * half) +
        reduceMatrix q n 1 1 ed := by rw [hed]; abel
  have hpublic : (selector * w.base) * decoder + msg * half =
      (selector * w.base) * key +
        ((selector * w.base) * one - (secret * publicCircuit - secret * w.gadget)) *
          reduceMatrix q n ell 1 dlift := by
    simpa only [← hdlift, Matrix.mul_assoc] using final_public_cancellation
      DiamondBackend.backend params w selector
  have hresult := final_encoding_approx (by decide : 0 < n)
    (state * decoder) (state * key) (-(msg * half))
    ((selector * w.base) * decoder + msg * half) ((selector * w.base) * key)
    (state * one) circuit ((selector * w.base) * one)
    (secret * publicCircuit - secret * w.gadget) ed ek eo ec dlift
    hdec hek heo hec hpublic hbd hbk hbo hbc hdliftBound
  simpa only [← hdlift, hcenter] using hresult

/-- Version consuming the actual encryption run: all three integer preimage bounds are
derived from its sampled trapdoor cutoff. Only local injector/circuit induction remains. -/
theorem final_residual_from_encrypt_run
    (hashModel : HashModel) (params : Stage_encrypt.Params) {inputs outputs}
    (hrun : Stage_encrypt.generatedRoot DiamondBackend.backend hashModel params inputs outputs)
    (w : FinalPublicWitness DiamondBackend.backend params outputs.1 outputs.2.2.1
      outputs.2.2.2.1 outputs.2.2.2.2.2.1 outputs.2.2.2.2.2.2.2.2.1
      outputs.2.2.2.2.1 outputs.2.2.2.2.2.2.2.2.2.1)
    (hq : params.diamond_modulus = (q : Int))
    (selector : ExactMatrix q n 1 2) (message : Bool)
    (hmessage : selector 0 1 = if message then 1 else 0)
    (state : ExactMatrix q n 1 inner) (circuit : ExactMatrix q n 1 ell)
    {B BH : Nat} (hstate : Approx state (selector * w.base) B)
    (hcircuit : Approx circuit
      (matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1)
          outputs.2.2.2.2.2.2.2.2.2.1 -
        matrixMul ((fun _ _ ↦ selector 0 0) : ExactMatrix q n 1 1) w.gadget) BH) :
    let B0 := projection * B * params.diamond_preimage_max_coefficient_bound.toNat
    Approx (state * outputs.1 - (state * outputs.2.2.1 +
      (state * outputs.2.2.2.1 - circuit) * outputs.2.2.2.2.2.1))
      (matrixPolynomial [(MxxWe.messageCenter q message : Int)])
      (2 * B0 + a * (B0 + BH)) := by
  obtain ⟨hd, hk, ho⟩ := generated_final_preimages_bounded DiamondBackend.backend
    hashModel params hrun
  exact final_residual_from_state params w hq selector message hmessage state circuit
    hstate hd hk ho hcircuit

#print axioms generated_final_preimages_bounded
#print axioms final_state_project
#print axioms final_actual_encodings
#print axioms final_residual_from_state
#print axioms final_residual_from_encrypt_run

end DiamondGeneratedProof
