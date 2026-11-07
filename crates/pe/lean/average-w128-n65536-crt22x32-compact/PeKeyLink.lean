import PeDecryptBits

/-!
# The key of the predicate

The keygen outputs in the form of `decrypt_bits`: the anchor preimages with their slot targets, and
the extension samples with their bounds.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 16000000 in
theorem keygen_link {model : HashModel} {tape : SampleTape} {path : List Nat}
    {td : TrapdoorValue (ExactMatrix Q N 1 501) Unit} (hkind : td.kind = .sampledSecret) {key : ByteArray}
    {K0 K1 : ExactMatrix Q N 1 44} {BQ : ExactMatrix PQ N 1 501}
    {w : Stage_keygen.generatedRoot.Witness} {o : KeygenOutputs}
    (F : Stage_keygen.generatedRoot.Facts (⟨(td, key, K0, K1, BQ, ()), w⟩ :
      Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params) o) :
    ∃ p' : Fin 128 → ErrorMatrix N 501 1, (∀ i, CoeffBound (p' i) keyCutoff) ∧
      (∀ i, anchorW (⟨(td, key, K0, K1, BQ, ()), w⟩ :
        Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params) i =
          reduceMatrix Q N 501 1 (p' i)) ∧
      Stage_keygen.generatedRoot.w_7_0 (⟨(td, key, K0, K1, BQ, ()), w⟩ :
        Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params) =
          reduceMatrix Q N 88 1 (keySample tape (path ++ [7])) ∧
      CoeffBound (keySample (rows := 88) tape (path ++ [7])) keyCutoff ∧
      ∀ i, σs i (trapdoorPublic td * reduceMatrix Q N 501 1 (p' i)) =
        slotResidual model key
          (messageRow model key 0 - joinRow ![K0, K1] * reduceMatrix Q N 88 1 (keySample tape (path ++ [7])))
          (Stage_keygen.generatedRoot.w_10_0 (⟨(td, key, K0, K1, BQ, ()), w⟩ :
            Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params))
          (expand (fun _ _ ↦ BQ 0 0))
          (Stage_keygen.generatedRoot.w_21_0 (⟨(td, key, K0, K1, BQ, ()), w⟩ :
            Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params) i)
          (Stage_keygen.generatedRoot.w_22_0 (⟨(td, key, K0, K1, BQ, ()), w⟩ :
            Stage_keygen.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_6_params) i) i := by
  have hA := keygen_anchors hkind F
  choose hpub p' hp' hb using hA
  obtain ⟨e7, b7⟩ := key_sample F.c0.h7_0
  refine ⟨p', hb, hp', e7, b7, fun i ↦ ?_⟩
  rw [← hp' i, hpub i, keygen_targets F i i.isLt]
  simp only [σs]
  rw [subM_comp (slotAut_odd i) (slotInv_odd i) odd_one (by rw [slot_inv]; rfl), subM_one_index]

end PeProof
