import Stage_import_0
import W_Stage_import_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_import_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_import_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_import_0.parallel_generatedRoot_6.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_6 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_6 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_6
  exact h


theorem Stage_import_0.parallel_generatedRoot_11.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_11 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_11 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_11
  exact h


theorem Stage_import_0.parallel_generatedRoot_14.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_14 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_14 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_14
  exact h


theorem Stage_import_0.parallel_generatedRoot_17.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_17 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_17 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_17
  exact h


theorem Stage_import_0.parallel_generatedRoot_20.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_20 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_20 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_20
  exact h


theorem Stage_import_0.parallel_generatedRoot_23.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_23 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_23 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_23
  exact h


theorem Stage_import_0.parallel_generatedRoot_26.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_26 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_26 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_26
  exact h


theorem Stage_import_0.parallel_generatedRoot_29.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_29 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_29 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_29
  exact h


theorem Stage_import_0.parallel_generatedRoot_32.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_32 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_32 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_32
  exact h


theorem Stage_import_0.parallel_generatedRoot_35.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_35 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_35 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_35
  exact h


theorem Stage_import_0.parallel_generatedRoot_38.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_38 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_38 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_38
  exact h


theorem Stage_import_0.parallel_generatedRoot_41.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_41 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_41 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_41
  exact h


theorem Stage_import_0.parallel_generatedRoot_42.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_42 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_42 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_42
  exact h


theorem Stage_import_0.parallel_generatedRoot_43.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_43 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_43 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_43
  exact h


theorem Stage_import_0.parallel_generatedRoot_44.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_44 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_44 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_44
  exact h


theorem Stage_import_0.parallel_generatedRoot_45.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_45 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_45 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_45
  exact h


theorem Stage_import_0.parallel_generatedRoot_46.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_46 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_46 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_46
  exact h


theorem Stage_import_0.parallel_generatedRoot_47.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_47 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_47 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_47
  exact h


theorem Stage_import_0.parallel_generatedRoot_48.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_48 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_48 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_48
  exact h


theorem Stage_import_0.parallel_generatedRoot_49.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_49 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_49 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_49
  exact h


theorem Stage_import_0.parallel_generatedRoot_50.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_50 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_50 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_50
  exact h


theorem Stage_import_0.parallel_generatedRoot_51.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_51 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_51 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_51
  exact h


theorem Stage_import_0.parallel_generatedRoot_52.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_52 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_52 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_52
  exact h


theorem Stage_import_0.parallel_generatedRoot_53.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_53 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_53 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_53
  exact h


theorem Stage_import_0.parallel_generatedRoot_54.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_54 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_54 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_54
  exact h


theorem Stage_import_0.parallel_generatedRoot_55.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_55 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_55 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_55
  exact h


theorem Stage_import_0.parallel_generatedRoot_56.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_56 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_56 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_56
  exact h


theorem Stage_import_0.parallel_generatedRoot_57.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_57 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_57 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_57
  exact h


theorem Stage_import_0.parallel_generatedRoot_58.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_58 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_58 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_58
  exact h


theorem Stage_import_0.parallel_generatedRoot_59.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_59 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_59 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_59
  exact h


theorem Stage_import_0.parallel_generatedRoot_60.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_60 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_60 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_60
  exact h


theorem Stage_import_0.parallel_generatedRoot_61.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_61 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_61 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_61
  exact h


theorem Stage_import_0.parallel_generatedRoot_62.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_62 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_62 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_62
  exact h


theorem Stage_import_0.parallel_generatedRoot_63.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_63 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_63 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_63
  exact h


theorem Stage_import_0.parallel_generatedRoot_64.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_64 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_64 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_64
  exact h


theorem Stage_import_0.parallel_generatedRoot_65.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_65 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_65 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_65
  exact h


theorem Stage_import_0.parallel_generatedRoot_66.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_66 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_66 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_66
  exact h


theorem Stage_import_0.parallel_generatedRoot_67.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_67 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_67 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_67
  exact h


theorem Stage_import_0.parallel_generatedRoot_68.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_68 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_68 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_68
  exact h


theorem Stage_import_0.parallel_generatedRoot_69.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_69 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_69 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_69
  exact h


theorem Stage_import_0.parallel_generatedRoot_70.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_70 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_70 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_70
  exact h


theorem Stage_import_0.parallel_generatedRoot_71.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_71 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_71 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_71
  exact h


theorem Stage_import_0.parallel_generatedRoot_72.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_72 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_72 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_72
  exact h


theorem Stage_import_0.parallel_generatedRoot_73.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_73 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_73 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_73
  exact h


theorem Stage_import_0.parallel_generatedRoot_74.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_74 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_74 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_74
  exact h


theorem Stage_import_0.parallel_generatedRoot_75.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_75 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_75 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_75
  exact h


theorem Stage_import_0.parallel_generatedRoot_76.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_76 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_76 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_76
  exact h


theorem Stage_import_0.parallel_generatedRoot_77.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_import_0.parallel_generatedRoot_77 tape path params x3 inputs outputs) : W_Stage_import_0.parallel_generatedRoot_77 params' x3 inputs outputs := by
  delta W_Stage_import_0.parallel_generatedRoot_77
  exact h


theorem Stage_import_0.generatedRoot.constraints_0.bridge {tape path params params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2}
    (h : Stage_import_0.generatedRoot.constraints_0 tape path params w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2) : W_Stage_import_0.generatedRoot.constraints_0 params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_import_0.parallel_generatedRoot_6.bridge (h2 i))), h3, (fun i ↦ (Stage_import_0.parallel_generatedRoot_11.bridge (h4 i))), h5, (fun i ↦ (Stage_import_0.parallel_generatedRoot_14.bridge (h6 i))), h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩)


theorem Stage_import_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_import_0.generatedRoot tape path params inputs outputs) : W_Stage_import_0.generatedRoot params' inputs outputs := by
  delta W_Stage_import_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_11_0, wit1.w_11_1, wit1.w_11_2, wit1.w_14_0, wit1.w_14_1, wit1.w_14_2⟩, (by obtain ⟨h3, h4⟩ := h2; exact ⟨(Stage_import_0.generatedRoot.constraints_0.bridge h3), h4⟩)⟩)
