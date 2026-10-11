import Requirement_0
import W_Requirement_0
import ExponentSampling

/-! Every run of the average-case relations of `Requirement_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Requirement_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Requirement_0.sequential_generatedRoot_6.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_6 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_6 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_6
  obtain ⟨x1, x2, x3, x4, x5, hb⟩ := h
  exact ⟨x1, x2, x3, x4, x5, hb⟩


theorem Requirement_0.sequential_sequential_generatedRoot_8_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_8_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_8_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_8_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_8.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_8.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_8.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_8_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_8.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_8 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_8 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_8
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_8.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_10_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_10_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_10_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_10_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_10.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_10.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_10.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_10_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_10.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_10 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_10 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_10
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_10.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_12_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_12_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_12_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_12_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_12.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_12.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_12.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_12_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_12.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_12 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_12 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_12
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_12.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_14_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_14_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_14_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_14_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_14.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_14.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_14.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_14_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_14.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_14 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_14 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_14
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_14.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_16_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_16_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_16_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_16_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_16.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_16.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_16.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_16_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_16.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_16 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_16 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_16
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_16.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_18_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_18_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_18_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_18_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_18.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_18.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_18.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_18_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_18.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_18 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_18 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_18
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_18.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_20_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_20_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_20_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_20_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_20.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_20.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_20.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_20_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_20.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_20 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_20 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_20
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_20.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_22_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_22_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_22_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_22_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_22.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_22.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_22.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_22_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_22.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_22 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_22 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_22
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_22.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_24_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_24_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_24_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_24_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_24.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_24.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_24.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_24_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_24.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_24 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_24 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_24
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_24.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_26_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_26_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_26_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_26_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_26.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_26.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_26.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_26_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_26.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_26 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_26 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_26
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_26.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_28_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_28_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_28_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_28_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_28.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_28.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_28.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_28_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_28.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_28 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_28 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_28
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_28.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_30_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_30_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_30_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_30_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_30.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_30.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_30.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_30_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_30.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_30 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_30 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_30
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_30.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_32_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_32_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_32_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_32_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_32.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_32.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_32.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_32_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_32.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_32 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_32 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_32
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_32.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_34_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_34_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_34_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_34_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_34.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_34.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_34.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_34_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_34.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_34 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_34 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_34
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_34.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_36_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_36_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_36_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_36_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_36.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_36.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_36.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_36_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_36.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_36 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_36 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_36
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_36.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_38_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_38_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_38_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_38_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_38.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_38.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_38.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_38_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_38.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_38 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_38 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_38
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_38.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_40_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_40_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_40_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_40_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_40.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_40.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_40.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_40_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_40.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_40 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_40 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_40
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_40.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_42_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_42_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_42_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_42_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_42.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_42.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_42.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_42_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_42.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_42 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_42 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_42
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_42.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_44_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_44_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_44_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_44_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_44.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_44.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_44.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_44_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_44.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_44 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_44 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_44
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_44.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_46_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_46_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_46_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_46_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_46.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_46.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_46.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_46_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_46.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_46 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_46 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_46
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_46.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_48_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_48_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_48_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_48_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_48.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_48.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_48.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_48_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_48.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_48 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_48 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_48
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_48.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_50_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_50_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_50_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_50_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_50.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_50.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_50.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_50_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_50.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_50 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_50 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_50
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_50.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_52_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_52_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_52_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_52_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_52.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_52.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_52.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_52_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_52.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_52 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_52 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_52
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_52.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_54_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_54_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_54_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_54_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_54.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_54.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_54.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_54_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_54.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_54 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_54 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_54
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_54.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_56_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_56_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_56_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_56_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_56.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_56.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_56.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_56_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_56.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_56 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_56 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_56
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_56.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_58_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_58_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_58_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_58_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_58.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_58.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_58.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_58_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_58.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_58 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_58 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_58
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_58.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_60_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_60_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_60_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_60_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_60.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_60.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_60.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_60_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_60.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_60 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_60 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_60
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_60.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_62_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_62_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_62_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_62_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_62.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_62.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_62.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_62_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_62.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_62 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_62 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_62
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_62.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_64_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_64_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_64_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_64_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_64.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_64.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_64.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_64_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_64.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_64 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_64 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_64
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_64.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_66_5.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_66_5 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_66_5 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_66_5
  obtain ⟨x1, hb⟩ := h
  exact ⟨x1, hb⟩


theorem Requirement_0.sequential_generatedRoot_66.constraints_0.bridge {tape path params params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_66.constraints_0 tape path params i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs) : W_Requirement_0.sequential_generatedRoot_66.constraints_0 params' i_0 w_3_0 w_2_0 w_4_0 w_5_0 w_11_0 outputs := by
  obtain ⟨a1, a2, a3⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_66_5.bridge hs)), a3⟩


theorem Requirement_0.sequential_generatedRoot_66.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_66 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_66 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_66
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_5_0⟩, (Requirement_0.sequential_generatedRoot_66.constraints_0.bridge hb)⟩


theorem Requirement_0.sequential_sequential_generatedRoot_78_10.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_78_10 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_78_10 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_78_10
  obtain ⟨x1, x2, x3, hb⟩ := h
  exact ⟨x1, x2, x3, hb⟩


theorem Requirement_0.sequential_sequential_generatedRoot_78_14.bridge {tape path params params' x3 i_1 inputs outputs}
    (h : Requirement_0.sequential_sequential_generatedRoot_78_14 tape path params x3 i_1 inputs outputs) : W_Requirement_0.sequential_sequential_generatedRoot_78_14 params' x3 i_1 inputs outputs := by
  delta W_Requirement_0.sequential_sequential_generatedRoot_78_14
  obtain ⟨x1, x2, x3, hb⟩ := h
  exact ⟨x1, x2, x3, hb⟩


theorem Requirement_0.sequential_generatedRoot_78.constraints_0.bridge {tape path params params' i_0 w_2_0 w_8_0 w_9_0 w_13_0 w_3_0 w_4_0 w_7_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_25_0 outputs}
    (h : Requirement_0.sequential_generatedRoot_78.constraints_0 tape path params i_0 w_2_0 w_8_0 w_9_0 w_13_0 w_3_0 w_4_0 w_7_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_25_0 outputs) : W_Requirement_0.sequential_generatedRoot_78.constraints_0 params' i_0 w_2_0 w_8_0 w_9_0 w_13_0 w_3_0 w_4_0 w_7_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_25_0 outputs := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9, a10⟩ := h
  exact ⟨a1, a2, a3, a4, (Requirement_0.iterRuns_mono a5 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_78_10.bridge hs)), a6, (Requirement_0.iterRuns_mono a7 (fun _ _ _ hs ↦ Requirement_0.sequential_sequential_generatedRoot_78_14.bridge hs)), a8, a9, a10⟩


theorem Requirement_0.sequential_generatedRoot_78.bridge {tape path params params' i_0 inputs outputs}
    (h : Requirement_0.sequential_generatedRoot_78 tape path params i_0 inputs outputs) : W_Requirement_0.sequential_generatedRoot_78 params' i_0 inputs outputs := by
  delta W_Requirement_0.sequential_generatedRoot_78
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_4_0, wit.w_10_0, wit.w_14_0⟩, (Requirement_0.sequential_generatedRoot_78.constraints_0.bridge hb)⟩


theorem Requirement_0.generatedRoot.constraints_1.bridge {tape path params params' w_3_0 w_15_0 w_17_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs}
    (h : Requirement_0.generatedRoot.constraints_1 tape path params w_3_0 w_15_0 w_17_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs) : W_Requirement_0.generatedRoot.constraints_1 params' w_3_0 w_15_0 w_17_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_38.bridge hs)), a3, (Requirement_0.iterRuns_mono a4 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_40.bridge hs)), a5, (Requirement_0.iterRuns_mono a6 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_42.bridge hs)), a7, (Requirement_0.iterRuns_mono a8 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_44.bridge hs)), a9, (Requirement_0.iterRuns_mono a10 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_46.bridge hs)), a11, (Requirement_0.iterRuns_mono a12 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_48.bridge hs)), a13, (Requirement_0.iterRuns_mono a14 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_50.bridge hs)), a15, (Requirement_0.iterRuns_mono a16 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_52.bridge hs)), a17, (Requirement_0.iterRuns_mono a18 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_54.bridge hs)), a19, (Requirement_0.iterRuns_mono a20 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_56.bridge hs)), a21, (Requirement_0.iterRuns_mono a22 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_58.bridge hs)), a23, (Requirement_0.iterRuns_mono a24 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_60.bridge hs)), a25, (Requirement_0.iterRuns_mono a26 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_62.bridge hs)), a27, (Requirement_0.iterRuns_mono a28 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_64.bridge hs)), a29, (Requirement_0.iterRuns_mono a30 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_66.bridge hs)), a31, (Requirement_0.iterRuns_mono a32 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_78.bridge hs)), a33⟩


theorem Requirement_0.generatedRoot.constraints_0.bridge {tape path params params' w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs}
    (h : Requirement_0.generatedRoot.constraints_0 tape path params w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs) : W_Requirement_0.generatedRoot.constraints_0 params' w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_77_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_76_0 w_78_0 outputs := by
  obtain ⟨a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33⟩ := h
  exact ⟨a1, (Requirement_0.iterRuns_mono a2 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_6.bridge hs)), a3, (Requirement_0.iterRuns_mono a4 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_8.bridge hs)), a5, (Requirement_0.iterRuns_mono a6 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_10.bridge hs)), a7, (Requirement_0.iterRuns_mono a8 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_12.bridge hs)), a9, (Requirement_0.iterRuns_mono a10 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_14.bridge hs)), a11, (Requirement_0.iterRuns_mono a12 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_16.bridge hs)), a13, (Requirement_0.iterRuns_mono a14 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_18.bridge hs)), a15, (Requirement_0.iterRuns_mono a16 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_20.bridge hs)), a17, (Requirement_0.iterRuns_mono a18 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_22.bridge hs)), a19, (Requirement_0.iterRuns_mono a20 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_24.bridge hs)), a21, (Requirement_0.iterRuns_mono a22 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_26.bridge hs)), a23, (Requirement_0.iterRuns_mono a24 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_28.bridge hs)), a25, (Requirement_0.iterRuns_mono a26 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_30.bridge hs)), a27, (Requirement_0.iterRuns_mono a28 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_32.bridge hs)), a29, (Requirement_0.iterRuns_mono a30 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_34.bridge hs)), a31, (Requirement_0.iterRuns_mono a32 (fun _ _ _ hs ↦ Requirement_0.sequential_generatedRoot_36.bridge hs)), (Requirement_0.generatedRoot.constraints_1.bridge a33)⟩


theorem Requirement_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Requirement_0.generatedRoot tape path params inputs outputs) : W_Requirement_0.generatedRoot params' inputs outputs := by
  delta W_Requirement_0.generatedRoot
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_6_0, wit.w_8_0, wit.w_10_0, wit.w_12_0, wit.w_14_0, wit.w_16_0, wit.w_18_0, wit.w_20_0, wit.w_22_0, wit.w_24_0, wit.w_26_0, wit.w_28_0, wit.w_30_0, wit.w_32_0, wit.w_34_0, wit.w_36_0, wit.w_38_0, wit.w_40_0, wit.w_42_0, wit.w_44_0, wit.w_46_0, wit.w_48_0, wit.w_50_0, wit.w_52_0, wit.w_54_0, wit.w_56_0, wit.w_58_0, wit.w_60_0, wit.w_62_0, wit.w_64_0, wit.w_66_0, wit.w_78_0⟩, (Requirement_0.generatedRoot.constraints_0.bridge hb)⟩
