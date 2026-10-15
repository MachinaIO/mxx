import Stage_keys_import_0_0
import W_Stage_keys_import_0_0
import ExponentSampling

/-! Every run of the average-case relations of `Stage_keys_import_0_0` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_keys_import_0_0.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_keys_import_0_0.parallel_generatedRoot_6.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_6 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_6 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_6
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_11.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_11 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_11 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_11
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_14.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_14 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_14 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_14
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_19.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_19 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_19 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_19
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_22.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_22 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_22 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_22
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_25.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_25 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_25 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_25
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_28.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_28 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_28 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_28
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_31.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_31 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_31 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_31
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_34.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_34 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_34 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_34
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_37.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_37 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_37 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_37
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_40.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_40 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_40 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_40
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_43.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_43 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_43 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_43
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_46.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_46 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_46 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_46
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_47.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_47 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_47 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_47
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_48.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_48 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_48 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_48
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_49.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_49 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_49 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_49
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_50.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_50 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_50 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_50
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_51.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_51 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_51 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_51
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_52.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_52 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_52 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_52
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_53.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_53 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_53 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_53
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_54.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_54 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_54 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_54
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_55.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_55 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_55 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_55
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_56.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_56 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_56 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_56
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_57.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_57 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_57 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_57
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_58.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_58 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_58 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_58
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_59.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_59 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_59 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_59
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_60.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_60 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_60 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_60
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_61.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_61 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_61 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_61
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_62.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_62 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_62 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_62
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_63.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_63 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_63 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_63
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_64.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_64 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_64 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_64
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_65.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_65 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_65 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_65
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_66.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_66 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_66 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_66
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_67.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_67 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_67 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_67
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_68.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_68 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_68 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_68
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_69.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_69 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_69 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_69
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_70.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_70 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_70 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_70
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_71.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_71 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_71 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_71
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_72.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_72 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_72 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_72
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_73.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_73 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_73 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_73
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_74.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_74 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_74 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_74
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_75.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_75 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_75 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_75
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_76.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_76 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_76 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_76
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_77.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_77 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_77 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_77
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_78.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_78 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_78 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_78
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_79.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_79 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_79 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_79
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_80.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_80 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_80 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_80
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_81.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_81 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_81 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_81
  exact h


theorem Stage_keys_import_0_0.parallel_generatedRoot_82.bridge {tape path params params' x3 inputs outputs}
    (h : Stage_keys_import_0_0.parallel_generatedRoot_82 tape path params x3 inputs outputs) : W_Stage_keys_import_0_0.parallel_generatedRoot_82 params' x3 inputs outputs := by
  delta W_Stage_keys_import_0_0.parallel_generatedRoot_82
  exact h


theorem Stage_keys_import_0_0.generatedRoot.constraints_3.bridge {tape path params params' w_0_0 w_1_0 w_7_0 w_8_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 outputs}
    (h : Stage_keys_import_0_0.generatedRoot.constraints_3 tape path params w_0_0 w_1_0 w_7_0 w_8_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 outputs) : W_Stage_keys_import_0_0.generatedRoot.constraints_3 params' w_0_0 w_1_0 w_7_0 w_8_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 w_82_0 w_82_1 outputs := by
  exact (by obtain ⟨h1, h2, h3⟩ := h; exact ⟨h1, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_82.bridge (h2 i))), h3⟩)


theorem Stage_keys_import_0_0.generatedRoot.constraints_2.bridge {tape path params params' w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1}
    (h : Stage_keys_import_0_0.generatedRoot.constraints_2 tape path params w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1) : W_Stage_keys_import_0_0.generatedRoot.constraints_2 params' w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_66_0 w_66_1 w_67_0 w_67_1 w_68_0 w_68_1 w_69_0 w_69_1 w_70_0 w_70_1 w_71_0 w_71_1 w_72_0 w_72_1 w_73_0 w_73_1 w_74_0 w_74_1 w_75_0 w_75_1 w_76_0 w_76_1 w_77_0 w_77_1 w_78_0 w_78_1 w_79_0 w_79_1 w_80_0 w_80_1 w_81_0 w_81_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_66.bridge (h2 i))), h3, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_67.bridge (h4 i))), h5, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_68.bridge (h6 i))), h7, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_69.bridge (h8 i))), h9, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_70.bridge (h10 i))), h11, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_71.bridge (h12 i))), h13, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_72.bridge (h14 i))), h15, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_73.bridge (h16 i))), h17, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_74.bridge (h18 i))), h19, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_75.bridge (h20 i))), h21, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_76.bridge (h22 i))), h23, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_77.bridge (h24 i))), h25, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_78.bridge (h26 i))), h27, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_79.bridge (h28 i))), h29, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_80.bridge (h30 i))), h31, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_81.bridge (h32 i)))⟩)


theorem Stage_keys_import_0_0.generatedRoot.constraints_1.bridge {tape path params params' w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1}
    (h : Stage_keys_import_0_0.generatedRoot.constraints_1 tape path params w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1) : W_Stage_keys_import_0_0.generatedRoot.constraints_1 params' w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_50_0 w_50_1 w_51_0 w_51_1 w_52_0 w_52_1 w_53_0 w_53_1 w_54_0 w_54_1 w_55_0 w_55_1 w_56_0 w_56_1 w_57_0 w_57_1 w_58_0 w_58_1 w_59_0 w_59_1 w_60_0 w_60_1 w_61_0 w_61_1 w_62_0 w_62_1 w_63_0 w_63_1 w_64_0 w_64_1 w_65_0 w_65_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_50.bridge (h2 i))), h3, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_51.bridge (h4 i))), h5, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_52.bridge (h6 i))), h7, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_53.bridge (h8 i))), h9, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_54.bridge (h10 i))), h11, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_55.bridge (h12 i))), h13, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_56.bridge (h14 i))), h15, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_57.bridge (h16 i))), h17, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_58.bridge (h18 i))), h19, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_59.bridge (h20 i))), h21, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_60.bridge (h22 i))), h23, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_61.bridge (h24 i))), h25, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_62.bridge (h26 i))), h27, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_63.bridge (h28 i))), h29, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_64.bridge (h30 i))), h31, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_65.bridge (h32 i)))⟩)


theorem Stage_keys_import_0_0.generatedRoot.constraints_0.bridge {tape path params params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1}
    (h : Stage_keys_import_0_0.generatedRoot.constraints_0 tape path params w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1) : W_Stage_keys_import_0_0.generatedRoot.constraints_0 params' w_2_0 w_3_0 w_4_0 w_5_0 w_9_0 w_10_0 w_12_0 w_13_0 w_15_0 w_16_0 w_17_0 w_18_0 w_20_0 w_21_0 w_23_0 w_24_0 w_26_0 w_27_0 w_29_0 w_30_0 w_32_0 w_33_0 w_35_0 w_36_0 w_38_0 w_39_0 w_41_0 w_42_0 w_44_0 w_45_0 w_6_0 w_6_1 w_6_2 w_11_0 w_11_1 w_11_2 w_14_0 w_14_1 w_14_2 w_19_0 w_19_1 w_19_2 w_22_0 w_22_1 w_25_0 w_25_1 w_28_0 w_28_1 w_31_0 w_31_1 w_34_0 w_34_1 w_37_0 w_37_1 w_40_0 w_40_1 w_43_0 w_43_1 w_46_0 w_46_1 w_47_0 w_47_1 w_48_0 w_48_1 w_49_0 w_49_1 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_6.bridge (h2 i))), h3, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_11.bridge (h4 i))), h5, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_14.bridge (h6 i))), h7, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_19.bridge (h8 i))), h9, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_22.bridge (h10 i))), h11, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_25.bridge (h12 i))), h13, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_28.bridge (h14 i))), h15, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_31.bridge (h16 i))), h17, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_34.bridge (h18 i))), h19, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_37.bridge (h20 i))), h21, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_40.bridge (h22 i))), h23, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_43.bridge (h24 i))), h25, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_46.bridge (h26 i))), h27, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_47.bridge (h28 i))), h29, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_48.bridge (h30 i))), h31, (fun i ↦ (Stage_keys_import_0_0.parallel_generatedRoot_49.bridge (h32 i)))⟩)


theorem Stage_keys_import_0_0.generatedRoot.bridge {tape path params params' inputs outputs}
    (h : Stage_keys_import_0_0.generatedRoot tape path params inputs outputs) : W_Stage_keys_import_0_0.generatedRoot params' inputs outputs := by
  delta W_Stage_keys_import_0_0.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_6_0, wit1.w_6_1, wit1.w_6_2, wit1.w_11_0, wit1.w_11_1, wit1.w_11_2, wit1.w_14_0, wit1.w_14_1, wit1.w_14_2, wit1.w_19_0, wit1.w_19_1, wit1.w_19_2, wit1.w_22_0, wit1.w_22_1, wit1.w_25_0, wit1.w_25_1, wit1.w_28_0, wit1.w_28_1, wit1.w_31_0, wit1.w_31_1, wit1.w_34_0, wit1.w_34_1, wit1.w_37_0, wit1.w_37_1, wit1.w_40_0, wit1.w_40_1, wit1.w_43_0, wit1.w_43_1, wit1.w_46_0, wit1.w_46_1, wit1.w_47_0, wit1.w_47_1, wit1.w_48_0, wit1.w_48_1, wit1.w_49_0, wit1.w_49_1, wit1.w_50_0, wit1.w_50_1, wit1.w_51_0, wit1.w_51_1, wit1.w_52_0, wit1.w_52_1, wit1.w_53_0, wit1.w_53_1, wit1.w_54_0, wit1.w_54_1, wit1.w_55_0, wit1.w_55_1, wit1.w_56_0, wit1.w_56_1, wit1.w_57_0, wit1.w_57_1, wit1.w_58_0, wit1.w_58_1, wit1.w_59_0, wit1.w_59_1, wit1.w_60_0, wit1.w_60_1, wit1.w_61_0, wit1.w_61_1, wit1.w_62_0, wit1.w_62_1, wit1.w_63_0, wit1.w_63_1, wit1.w_64_0, wit1.w_64_1, wit1.w_65_0, wit1.w_65_1, wit1.w_66_0, wit1.w_66_1, wit1.w_67_0, wit1.w_67_1, wit1.w_68_0, wit1.w_68_1, wit1.w_69_0, wit1.w_69_1, wit1.w_70_0, wit1.w_70_1, wit1.w_71_0, wit1.w_71_1, wit1.w_72_0, wit1.w_72_1, wit1.w_73_0, wit1.w_73_1, wit1.w_74_0, wit1.w_74_1, wit1.w_75_0, wit1.w_75_1, wit1.w_76_0, wit1.w_76_1, wit1.w_77_0, wit1.w_77_1, wit1.w_78_0, wit1.w_78_1, wit1.w_79_0, wit1.w_79_1, wit1.w_80_0, wit1.w_80_1, wit1.w_81_0, wit1.w_81_1, wit1.w_82_0, wit1.w_82_1⟩, (by obtain ⟨h3, h4, h5, h6⟩ := h2; exact ⟨(Stage_keys_import_0_0.generatedRoot.constraints_0.bridge h3), (Stage_keys_import_0_0.generatedRoot.constraints_1.bridge h4), (Stage_keys_import_0_0.generatedRoot.constraints_2.bridge h5), (Stage_keys_import_0_0.generatedRoot.constraints_3.bridge h6)⟩)⟩)
