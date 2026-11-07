import PeDecode

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut

def anchorIn {model : HashModel} {tape : SampleTape} {path : List Nat}
    (ctx : Stage_decode.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_9_params) :
    Fin 256 → ExactMatrix Q N 295 1 := Stage_decode.generatedRoot.w_10_0 ctx

noncomputable def roundedAnchor (model : HashModel) (key : ByteArray)
    (bQ : ExactMatrix PQ N 1 295) : ExactMatrix Q N 1 295 :=
  modulusSwitch bQ - modulusSwitch (anchorSecond model key)

noncomputable def compressedAnchor (model : HashModel) (key : ByteArray)
    (bQ : ExactMatrix PQ N 1 295) : ExactMatrix Q N 1 30 :=
  expand (fun _ _ ↦ bQ 0 0) - expand (fun _ _ ↦ anchorSecond model key 0 0)

set_option maxHeartbeats 16000000 in
theorem decode_phase {model : HashModel} {tape : SampleTape} {path : List Nat}
    {ctx : Stage_decode.generatedRoot.Ctx Backend.backend model tape path GeneratedClaim.stage_9_params}
    {o : Fin 1 → Int} (F : Stage_decode.generatedRoot.Facts ctx o) :
    let key := Stage_decode.generatedRoot.w_1_0 ctx
    let bQ := Stage_decode.generatedRoot.w_5_0 ctx
    let phase := (Stage_decode.generatedRoot.w_0_0 ctx - messageRow model key 1) -
      (∑ i : Fin 256, substituteMatrix (slotAut i) (slotAut_odd i) (roundedAnchor model key bQ * anchorIn ctx i) +
        ∑ i : Fin 256, substituteMatrix (slotAut i) (slotAut_odd i) (compressedAnchor model key bQ) *
          Stage_decode.generatedRoot.w_27_0 ctx i +
        ∑ p : Fin 11520, helperValue model key (Stage_decode.generatedRoot.w_31_0 ctx)
          (Stage_decode.generatedRoot.w_32_0 ctx) p * packedHelpers (Stage_decode.generatedRoot.w_36_0 ctx) p +
        (joinRow ![Stage_decode.generatedRoot.w_39_0 ctx, Stage_decode.generatedRoot.w_40_0 ctx] -
          joinRow ![Stage_decode.generatedRoot.w_42_0 ctx, Stage_decode.generatedRoot.w_43_0 ctx]) *
          Stage_decode.generatedRoot.w_46_0 ctx)
    ∀ k : Fin 1, o k = ((2 * (((phase 0 0).coeff (0 : Fin N)).val : Int) + (Q : Int) / 2) / (Q : Int)) % 2 := by
  intro key bQ phase
  have e2 : Stage_decode.generatedRoot.w_2_0 ctx = messageRow model key 1 := hashSample_eq F.c0.h2_0
  have e7 : Stage_decode.generatedRoot.w_7_0 ctx = anchorSecond model key := hashSample_eq F.c0.h7_0
  have e6 := F.c0.h6_0.2.2.2.2
  have e8 := F.c0.h8_0.2.2.2.2
  rw [e7] at e8
  have e9 : Stage_decode.generatedRoot.w_9_0 ctx = roundedAnchor model key bQ := by
    show matrixSub _ _ = _
    rw [e6, e8]
    rfl
  have hA := stream_sum
    (fun row : Fin 256 ↦ substituteMatrix (slotAut row) (slotAut_odd row)
      (Stage_decode.generatedRoot.w_9_0 ctx * anchorIn ctx row))
    (fun row current next bound body ↦ anchorStep bound body) F.c0.h11_0
  simp only [Stage_decode.generatedRoot.w_4_0, zero_add] at hA
  rw [e9] at hA
  obtain ⟨_, e523⟩ := sliceMatrix_column F.c0.h13_0
  obtain ⟨_, e530⟩ := sliceMatrix_column F.c0.h20_0
  have e529 : Stage_decode.generatedRoot.w_19_0 ctx = anchorSecond model key := hashSample_eq F.c0.h19_0
  have e527 : Stage_decode.generatedRoot.w_17_0 ctx = expand (fun _ _ ↦ bQ 0 0) := by
    rw [expand_read F.c0.h14_0 F.c0.h15_0 F.c0.h17_0]
    congr 1
    funext a b
    rw [Subsingleton.elim a 0, Subsingleton.elim b 0, e523]
    rfl
  have e534 : Stage_decode.generatedRoot.w_24_0 ctx = expand (fun _ _ ↦ anchorSecond model key 0 0) := by
    rw [expand_read F.c0.h21_0 F.c0.h22_0 F.c0.h24_0]
    congr 1
    funext a b
    rw [Subsingleton.elim a 0, Subsingleton.elim b 0, e530, e529]
    rfl
  have e536 : Stage_decode.generatedRoot.w_26_0 ctx = compressedAnchor model key bQ := by
    show matrixSub (Stage_decode.generatedRoot.w_17_0 ctx) (Stage_decode.generatedRoot.w_24_0 ctx) = _
    rw [e527, e534]
    rfl
  have hC := compactLoop (model := model) (tape := tape)
    (path := path) (params := GeneratedClaim.stage_9_params) F.c0.h28_0
  rw [e536] at hC
  have e544 := gadgetMatrixRuns_unique F.c0.h34_0 g0_runs
  have h545 := F.c0.h35_0
  rw [e544] at h545
  have e545 : Stage_decode.generatedRoot.w_35_0 ctx = zeroG := concat_joinRow (r := ![0, g0]) h545
  have h547 := F.c1.h37_0
  rw [e545] at h547
  have hH := stream_sum
    (fun row : Fin 256 ↦ ∑ lane : Fin 45,
      matrixMul (helperValue model key (Stage_decode.generatedRoot.w_31_0 ctx)
        (Stage_decode.generatedRoot.w_32_0 ctx) ⟨45 * row + lane, by omega⟩)
        ((fun entry _ ↦ Stage_decode.generatedRoot.w_36_0 ctx row entry lane) : ExactMatrix Q N 60 1))
    (fun row current next bound body ↦ helperContractRow bound body) h547
  simp only [Stage_decode.generatedRoot.w_30_0, zero_add] at hH
  have hH' : Stage_decode.generatedRoot.w_37_0 ctx = ∑ pair : Fin 11520,
      helperValue model key (Stage_decode.generatedRoot.w_31_0 ctx)
        (Stage_decode.generatedRoot.w_32_0 ctx) pair *
        packedHelpers (Stage_decode.generatedRoot.w_36_0 ctx) pair := by
    rw [packed_sum, hH]
    refine Finset.sum_congr rfl fun row _ ↦ Finset.sum_congr rfl fun lane _ ↦ ?_
    congr 1
    unfold packedHelpers
    have div : (45 * row.val + lane.val) / 45 = row.val := by omega
    have mod : (45 * row.val + lane.val) % 45 = lane.val := by omega
    simp only [div, mod]
  have e551 := concat_joinRow (r := ![_, _]) F.c1.h41_0
  have e554 := concat_joinRow (r := ![_, _]) F.c1.h44_0
  have hphase : Stage_decode.generatedRoot.w_49_0 ctx = phase := by
    show matrixSub (matrixSub (Stage_decode.generatedRoot.w_0_0 ctx) (Stage_decode.generatedRoot.w_2_0 ctx))
      (matrixAdd (matrixAdd (matrixAdd (Stage_decode.generatedRoot.w_11_0 ctx) (Stage_decode.generatedRoot.w_28_0 ctx))
        (Stage_decode.generatedRoot.w_37_0 ctx))
        (matrixMul (matrixSub (Stage_decode.generatedRoot.w_41_0 ctx) (Stage_decode.generatedRoot.w_44_0 ctx))
          (Stage_decode.generatedRoot.w_46_0 ctx))) = _
    rw [e2, hA, hC, hH', e551, e554]
    rfl
  intro k
  rw [F.c1.hout]
  obtain ⟨-, -, -, -, idx, hidx, -, hval⟩ := F.c1.h50_0 k
  have hk : k = 0 := Subsingleton.elim _ _
  have hi : idx = (0 : Fin N) := Fin.ext (by simpa [hk] using hidx)
  rw [hi] at hval
  show Stage_decode.generatedRoot.w_50_0 ctx k = _
  rw [hval, hphase]

end PeProof
