import Bridge_Stage_import_0_Part_0
import Stage_import_0_Part_1
import W_Stage_import_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_import_0_Part_1` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_import_0_Part_1.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_import_0.generatedRoot.constraints_0.bridge {tape path params params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2}
    (h : Stage_import_0.generatedRoot.constraints_0 tape path params w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2) : W_Stage_import_0.generatedRoot.constraints_0 params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_import_0.parallel_generatedRoot_6.bridge (h2 i))), h3, (fun i ↦ (Stage_import_0.parallel_generatedRoot_11.bridge (h4 i))), h5, (fun i ↦ (Stage_import_0.parallel_generatedRoot_14.bridge (h6 i))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_import_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_import_0.generatedRoot tape path params inputs outputs) : W_Stage_import_0.generatedRoot params' inputs outputs := by
  delta W_Stage_import_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_11_0, wit1.w_11_1, wit1.w_11_2, wit1.w_14_0, wit1.w_14_1, wit1.w_14_2⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(by obtain ⟨h5, h6⟩ := h3; exact ⟨(by obtain ⟨h7, h8⟩ := h5; exact ⟨(by obtain ⟨h9, h10⟩ := h7; exact ⟨(by obtain ⟨h11, h12⟩ := h9; exact ⟨(by obtain ⟨h13, h14⟩ := h11; exact ⟨(by obtain ⟨h15, h16⟩ := h13; exact ⟨(Stage_import_0.generatedRoot.constraints_0.bridge h15), h16⟩), h14⟩), h12⟩), h10⟩), h8⟩), h6⟩), h4⟩)⟩)
