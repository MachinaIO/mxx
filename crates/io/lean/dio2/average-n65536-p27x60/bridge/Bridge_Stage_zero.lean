import Stage_zero
import W_Stage_zero
import ExponentSampling

/-! Every run of the average-case relations of `Stage_zero` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_zero.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_zero.parallel_generatedRoot_1.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_1 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_1 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_1
  exact h


theorem Stage_zero.parallel_generatedRoot_2.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_2 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_2 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_2
  exact h


theorem Stage_zero.parallel_generatedRoot_3.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_3 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_3 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_3
  exact h


theorem Stage_zero.parallel_generatedRoot_4.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_4 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_4 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_4
  exact h


theorem Stage_zero.parallel_generatedRoot_5.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_5 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_5 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_5
  exact h


theorem Stage_zero.parallel_generatedRoot_6.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_6 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_6 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_6
  exact h


theorem Stage_zero.parallel_generatedRoot_7.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_7 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_7 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_7
  exact h


theorem Stage_zero.parallel_generatedRoot_8.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_8 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_8 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_8
  exact h


theorem Stage_zero.parallel_generatedRoot_9.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_9 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_9 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_9
  exact h


theorem Stage_zero.parallel_generatedRoot_10.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_10 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_10 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_10
  exact h


theorem Stage_zero.parallel_generatedRoot_11.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_11 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_11 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_11
  exact h


theorem Stage_zero.parallel_generatedRoot_12.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_12 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_12 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_12
  exact h


theorem Stage_zero.parallel_generatedRoot_13.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_13 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_13 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_13
  exact h


theorem Stage_zero.parallel_generatedRoot_14.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_14 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_14 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_14
  exact h


theorem Stage_zero.parallel_generatedRoot_15.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_15 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_15 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_15
  exact h


theorem Stage_zero.parallel_generatedRoot_16.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_16 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_16 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_16
  exact h


theorem Stage_zero.parallel_generatedRoot_17.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_17 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_17 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_17
  exact h


theorem Stage_zero.parallel_generatedRoot_18.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_18 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_18 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_18
  exact h


theorem Stage_zero.parallel_generatedRoot_19.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_19 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_19 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_19
  exact h


theorem Stage_zero.parallel_generatedRoot_20.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_20 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_20 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_20
  exact h


theorem Stage_zero.parallel_generatedRoot_21.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_21 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_21 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_21
  exact h


theorem Stage_zero.parallel_generatedRoot_22.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_22 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_22 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_22
  exact h


theorem Stage_zero.parallel_generatedRoot_23.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_23 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_23 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_23
  exact h


theorem Stage_zero.parallel_generatedRoot_24.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_24 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_24 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_24
  exact h


theorem Stage_zero.parallel_generatedRoot_25.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_25 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_25 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_25
  exact h


theorem Stage_zero.parallel_generatedRoot_26.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_26 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_26 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_26
  exact h


theorem Stage_zero.parallel_generatedRoot_27.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_27 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_27 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_27
  exact h


theorem Stage_zero.parallel_generatedRoot_28.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_28 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_28 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_28
  exact h


theorem Stage_zero.parallel_generatedRoot_29.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_29 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_29 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_29
  exact h


theorem Stage_zero.parallel_generatedRoot_30.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_30 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_30 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_30
  exact h


theorem Stage_zero.parallel_generatedRoot_31.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_31 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_31 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_31
  exact h


theorem Stage_zero.parallel_generatedRoot_32.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_32 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_32 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_32
  exact h


theorem Stage_zero.parallel_generatedRoot_33.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_33 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_33 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_33
  exact h


theorem Stage_zero.parallel_generatedRoot_34.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_34 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_34 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_34
  exact h


theorem Stage_zero.parallel_generatedRoot_35.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_35 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_35 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_35
  exact h


theorem Stage_zero.parallel_generatedRoot_36.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_36 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_36 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_36
  exact h


theorem Stage_zero.parallel_generatedRoot_37.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_37 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_37 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_37
  exact h


theorem Stage_zero.parallel_generatedRoot_38.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_38 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_38 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_38
  exact h


theorem Stage_zero.parallel_generatedRoot_39.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_39 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_39 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_39
  exact h


theorem Stage_zero.parallel_generatedRoot_40.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_40 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_40 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_40
  exact h


theorem Stage_zero.parallel_generatedRoot_41.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_41 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_41 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_41
  exact h


theorem Stage_zero.parallel_generatedRoot_42.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_42 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_42 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_42
  exact h


theorem Stage_zero.parallel_generatedRoot_43.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_43 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_43 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_43
  exact h


theorem Stage_zero.parallel_generatedRoot_44.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_44 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_44 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_44
  exact h


theorem Stage_zero.parallel_generatedRoot_45.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_45 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_45 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_45
  exact h


theorem Stage_zero.parallel_generatedRoot_46.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_46 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_46 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_46
  exact h


theorem Stage_zero.parallel_generatedRoot_47.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_47 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_47 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_47
  exact h


theorem Stage_zero.parallel_generatedRoot_48.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_48 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_48 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_48
  exact h


theorem Stage_zero.parallel_generatedRoot_49.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_49 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_49 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_49
  exact h


theorem Stage_zero.parallel_generatedRoot_50.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_50 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_50 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_50
  exact h


theorem Stage_zero.parallel_generatedRoot_51.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_51 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_51 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_51
  exact h


theorem Stage_zero.parallel_generatedRoot_52.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_52 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_52 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_52
  exact h


theorem Stage_zero.parallel_generatedRoot_53.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_53 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_53 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_53
  exact h


theorem Stage_zero.parallel_generatedRoot_54.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_54 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_54 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_54
  exact h


theorem Stage_zero.parallel_generatedRoot_55.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_55 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_55 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_55
  exact h


theorem Stage_zero.parallel_generatedRoot_56.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_56 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_56 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_56
  exact h


theorem Stage_zero.parallel_generatedRoot_57.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_57 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_57 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_57
  exact h


theorem Stage_zero.parallel_generatedRoot_58.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_58 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_58 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_58
  exact h


theorem Stage_zero.parallel_generatedRoot_59.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_59 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_59 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_59
  exact h


theorem Stage_zero.parallel_generatedRoot_60.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_60 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_60 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_60
  exact h


theorem Stage_zero.parallel_generatedRoot_61.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_61 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_61 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_61
  exact h


theorem Stage_zero.parallel_generatedRoot_62.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_62 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_62 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_62
  exact h


theorem Stage_zero.parallel_generatedRoot_63.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_63 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_63 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_63
  exact h


theorem Stage_zero.parallel_generatedRoot_64.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_64 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_64 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_64
  exact h


theorem Stage_zero.parallel_generatedRoot_65.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_65 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_65 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_65
  exact h


theorem Stage_zero.parallel_generatedRoot_66.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_66 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_66 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_66
  exact h


theorem Stage_zero.parallel_generatedRoot_67.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_67 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_67 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_67
  exact h


theorem Stage_zero.parallel_generatedRoot_68.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_68 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_68 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_68
  exact h


theorem Stage_zero.parallel_generatedRoot_69.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_69 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_69 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_69
  exact h


theorem Stage_zero.parallel_generatedRoot_70.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_70 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_70 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_70
  exact h


theorem Stage_zero.parallel_generatedRoot_71.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_71 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_71 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_71
  exact h


theorem Stage_zero.parallel_generatedRoot_72.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_72 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_72 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_72
  exact h


theorem Stage_zero.parallel_generatedRoot_73.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_73 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_73 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_73
  exact h


theorem Stage_zero.parallel_generatedRoot_74.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_74 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_74 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_74
  exact h


theorem Stage_zero.parallel_generatedRoot_75.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_75 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_75 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_75
  exact h


theorem Stage_zero.parallel_generatedRoot_76.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_76 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_76 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_76
  exact h


theorem Stage_zero.parallel_generatedRoot_77.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_77 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_77 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_77
  exact h


theorem Stage_zero.parallel_generatedRoot_78.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_78 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_78 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_78
  exact h


theorem Stage_zero.parallel_generatedRoot_79.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_79 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_79 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_79
  exact h


theorem Stage_zero.parallel_generatedRoot_80.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_80 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_80 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_80
  exact h


theorem Stage_zero.parallel_generatedRoot_81.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_81 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_81 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_81
  exact h


theorem Stage_zero.parallel_generatedRoot_82.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_82 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_82 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_82
  exact h


theorem Stage_zero.parallel_generatedRoot_83.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_83 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_83 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_83
  exact h


theorem Stage_zero.parallel_generatedRoot_84.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_84 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_84 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_84
  exact h


theorem Stage_zero.parallel_generatedRoot_85.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_85 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_85 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_85
  exact h


theorem Stage_zero.parallel_generatedRoot_86.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_86 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_86 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_86
  exact h


theorem Stage_zero.parallel_generatedRoot_87.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_87 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_87 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_87
  exact h


theorem Stage_zero.parallel_generatedRoot_88.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_88 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_88 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_88
  exact h


theorem Stage_zero.parallel_generatedRoot_89.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_89 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_89 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_89
  exact h


theorem Stage_zero.parallel_generatedRoot_90.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_90 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_90 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_90
  exact h


theorem Stage_zero.parallel_generatedRoot_91.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_91 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_91 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_91
  exact h


theorem Stage_zero.parallel_generatedRoot_92.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_92 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_92 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_92
  exact h


theorem Stage_zero.parallel_generatedRoot_93.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_93 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_93 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_93
  exact h


theorem Stage_zero.parallel_generatedRoot_94.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_94 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_94 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_94
  exact h


theorem Stage_zero.parallel_generatedRoot_95.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_95 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_95 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_95
  exact h


theorem Stage_zero.parallel_generatedRoot_96.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_96 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_96 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_96
  exact h


theorem Stage_zero.parallel_generatedRoot_97.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_97 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_97 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_97
  exact h


theorem Stage_zero.parallel_generatedRoot_98.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_98 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_98 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_98
  exact h


theorem Stage_zero.parallel_generatedRoot_99.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_99 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_99 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_99
  exact h


theorem Stage_zero.parallel_generatedRoot_100.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_100 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_100 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_100
  exact h


theorem Stage_zero.parallel_generatedRoot_101.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_101 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_101 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_101
  exact h


theorem Stage_zero.parallel_generatedRoot_102.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_102 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_102 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_102
  exact h


theorem Stage_zero.parallel_generatedRoot_103.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_103 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_103 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_103
  exact h


theorem Stage_zero.parallel_generatedRoot_104.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_104 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_104 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_104
  exact h


theorem Stage_zero.parallel_generatedRoot_105.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_105 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_105 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_105
  exact h


theorem Stage_zero.parallel_generatedRoot_106.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_106 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_106 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_106
  exact h


theorem Stage_zero.parallel_generatedRoot_107.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_107 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_107 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_107
  exact h


theorem Stage_zero.parallel_generatedRoot_108.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_108 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_108 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_108
  exact h


theorem Stage_zero.parallel_generatedRoot_109.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_109 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_109 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_109
  exact h


theorem Stage_zero.parallel_generatedRoot_110.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_110 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_110 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_110
  exact h


theorem Stage_zero.parallel_generatedRoot_111.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_111 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_111 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_111
  exact h


theorem Stage_zero.parallel_generatedRoot_112.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_112 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_112 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_112
  exact h


theorem Stage_zero.parallel_generatedRoot_113.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_113 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_113 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_113
  exact h


theorem Stage_zero.parallel_generatedRoot_114.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_114 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_114 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_114
  exact h


theorem Stage_zero.parallel_generatedRoot_115.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_115 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_115 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_115
  exact h


theorem Stage_zero.parallel_generatedRoot_116.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_116 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_116 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_116
  exact h


theorem Stage_zero.parallel_generatedRoot_117.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_117 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_117 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_117
  exact h


theorem Stage_zero.parallel_generatedRoot_118.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_118 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_118 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_118
  exact h


theorem Stage_zero.parallel_generatedRoot_119.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_119 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_119 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_119
  exact h


theorem Stage_zero.parallel_generatedRoot_120.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_120 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_120 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_120
  exact h


theorem Stage_zero.parallel_generatedRoot_121.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_121 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_121 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_121
  exact h


theorem Stage_zero.parallel_generatedRoot_122.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_122 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_122 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_122
  exact h


theorem Stage_zero.parallel_generatedRoot_123.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_123 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_123 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_123
  exact h


theorem Stage_zero.parallel_generatedRoot_124.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_124 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_124 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_124
  exact h


theorem Stage_zero.parallel_generatedRoot_125.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_125 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_125 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_125
  exact h


theorem Stage_zero.parallel_generatedRoot_126.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_126 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_126 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_126
  exact h


theorem Stage_zero.parallel_generatedRoot_127.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_127 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_127 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_127
  exact h


theorem Stage_zero.parallel_generatedRoot_128.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_128 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_128 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_128
  exact h


theorem Stage_zero.parallel_generatedRoot_129.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_129 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_129 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_129
  exact h


theorem Stage_zero.parallel_generatedRoot_130.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_130 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_130 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_130
  exact h


theorem Stage_zero.parallel_generatedRoot_131.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_131 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_131 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_131
  exact h


theorem Stage_zero.parallel_generatedRoot_132.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_132 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_132 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_132
  exact h


theorem Stage_zero.parallel_generatedRoot_133.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_133 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_133 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_133
  exact h


theorem Stage_zero.parallel_generatedRoot_134.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_134 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_134 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_134
  exact h


theorem Stage_zero.parallel_generatedRoot_135.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_135 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_135 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_135
  exact h


theorem Stage_zero.parallel_generatedRoot_136.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_136 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_136 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_136
  exact h


theorem Stage_zero.parallel_generatedRoot_137.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_137 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_137 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_137
  exact h


theorem Stage_zero.parallel_generatedRoot_138.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_138 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_138 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_138
  exact h


theorem Stage_zero.parallel_generatedRoot_139.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_139 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_139 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_139
  exact h


theorem Stage_zero.parallel_generatedRoot_140.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_140 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_140 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_140
  exact h


theorem Stage_zero.parallel_generatedRoot_141.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_141 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_141 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_141
  exact h


theorem Stage_zero.parallel_generatedRoot_142.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_142 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_142 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_142
  exact h


theorem Stage_zero.parallel_generatedRoot_143.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_143 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_143 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_143
  exact h


theorem Stage_zero.parallel_generatedRoot_144.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_144 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_144 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_144
  exact h


theorem Stage_zero.parallel_generatedRoot_145.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_145 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_145 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_145
  exact h


theorem Stage_zero.parallel_generatedRoot_146.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_146 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_146 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_146
  exact h


theorem Stage_zero.parallel_generatedRoot_147.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_147 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_147 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_147
  exact h


theorem Stage_zero.parallel_generatedRoot_148.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_148 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_148 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_148
  exact h


theorem Stage_zero.parallel_generatedRoot_149.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_149 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_149 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_149
  exact h


theorem Stage_zero.parallel_generatedRoot_150.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_150 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_150 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_150
  exact h


theorem Stage_zero.parallel_generatedRoot_151.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_151 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_151 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_151
  exact h


theorem Stage_zero.parallel_generatedRoot_152.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_152 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_152 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_152
  exact h


theorem Stage_zero.parallel_generatedRoot_153.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_153 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_153 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_153
  exact h


theorem Stage_zero.parallel_generatedRoot_154.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_154 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_154 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_154
  exact h


theorem Stage_zero.parallel_generatedRoot_155.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_155 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_155 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_155
  exact h


theorem Stage_zero.parallel_generatedRoot_156.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_156 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_156 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_156
  exact h


theorem Stage_zero.parallel_generatedRoot_157.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_157 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_157 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_157
  exact h


theorem Stage_zero.parallel_generatedRoot_158.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_158 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_158 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_158
  exact h


theorem Stage_zero.parallel_generatedRoot_159.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_159 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_159 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_159
  exact h


theorem Stage_zero.parallel_generatedRoot_160.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_160 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_160 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_160
  exact h


theorem Stage_zero.parallel_generatedRoot_161.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_161 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_161 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_161
  exact h


theorem Stage_zero.parallel_generatedRoot_162.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_162 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_162 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_162
  exact h


theorem Stage_zero.parallel_generatedRoot_163.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_163 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_163 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_163
  exact h


theorem Stage_zero.parallel_generatedRoot_164.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_164 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_164 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_164
  exact h


theorem Stage_zero.parallel_generatedRoot_165.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_165 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_165 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_165
  exact h


theorem Stage_zero.parallel_generatedRoot_166.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_166 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_166 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_166
  exact h


theorem Stage_zero.parallel_generatedRoot_167.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_167 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_167 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_167
  exact h


theorem Stage_zero.parallel_generatedRoot_168.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_168 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_168 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_168
  exact h


theorem Stage_zero.parallel_generatedRoot_169.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_169 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_169 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_169
  exact h


theorem Stage_zero.parallel_generatedRoot_170.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_170 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_170 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_170
  exact h


theorem Stage_zero.parallel_generatedRoot_171.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_171 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_171 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_171
  exact h


theorem Stage_zero.parallel_generatedRoot_172.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_172 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_172 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_172
  exact h


theorem Stage_zero.parallel_generatedRoot_173.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_173 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_173 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_173
  exact h


theorem Stage_zero.parallel_generatedRoot_174.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_174 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_174 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_174
  exact h


theorem Stage_zero.parallel_generatedRoot_175.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_175 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_175 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_175
  exact h


theorem Stage_zero.parallel_generatedRoot_176.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_176 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_176 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_176
  exact h


theorem Stage_zero.parallel_generatedRoot_177.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_177 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_177 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_177
  exact h


theorem Stage_zero.parallel_generatedRoot_178.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_178 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_178 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_178
  exact h


theorem Stage_zero.parallel_generatedRoot_179.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_179 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_179 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_179
  exact h


theorem Stage_zero.parallel_generatedRoot_180.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_180 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_180 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_180
  exact h


theorem Stage_zero.parallel_generatedRoot_181.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_181 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_181 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_181
  exact h


theorem Stage_zero.parallel_generatedRoot_182.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_182 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_182 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_182
  exact h


theorem Stage_zero.parallel_generatedRoot_183.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_183 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_183 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_183
  exact h


theorem Stage_zero.parallel_generatedRoot_184.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_184 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_184 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_184
  exact h


theorem Stage_zero.parallel_generatedRoot_185.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_185 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_185 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_185
  exact h


theorem Stage_zero.parallel_generatedRoot_186.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_186 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_186 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_186
  exact h


theorem Stage_zero.parallel_generatedRoot_187.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_187 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_187 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_187
  exact h


theorem Stage_zero.parallel_generatedRoot_188.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_188 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_188 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_188
  exact h


theorem Stage_zero.parallel_generatedRoot_189.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_189 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_189 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_189
  exact h


theorem Stage_zero.parallel_generatedRoot_190.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_190 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_190 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_190
  exact h


theorem Stage_zero.parallel_generatedRoot_191.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_191 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_191 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_191
  exact h


theorem Stage_zero.parallel_generatedRoot_192.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_192 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_192 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_192
  exact h


theorem Stage_zero.parallel_generatedRoot_193.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_193 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_193 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_193
  exact h


theorem Stage_zero.parallel_generatedRoot_194.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_194 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_194 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_194
  exact h


theorem Stage_zero.parallel_generatedRoot_195.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_195 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_195 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_195
  exact h


theorem Stage_zero.parallel_generatedRoot_196.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_196 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_196 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_196
  exact h


theorem Stage_zero.parallel_generatedRoot_197.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_197 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_197 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_197
  exact h


theorem Stage_zero.parallel_generatedRoot_198.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_198 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_198 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_198
  exact h


theorem Stage_zero.parallel_generatedRoot_199.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_199 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_199 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_199
  exact h


theorem Stage_zero.parallel_generatedRoot_200.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_200 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_200 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_200
  exact h


theorem Stage_zero.parallel_generatedRoot_201.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_201 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_201 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_201
  exact h


theorem Stage_zero.parallel_generatedRoot_202.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_202 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_202 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_202
  exact h


theorem Stage_zero.parallel_generatedRoot_203.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_203 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_203 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_203
  exact h


theorem Stage_zero.parallel_generatedRoot_204.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_204 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_204 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_204
  exact h


theorem Stage_zero.parallel_generatedRoot_205.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_205 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_205 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_205
  exact h


theorem Stage_zero.parallel_generatedRoot_206.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_206 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_206 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_206
  exact h


theorem Stage_zero.parallel_generatedRoot_207.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_207 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_207 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_207
  exact h


theorem Stage_zero.parallel_generatedRoot_208.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_208 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_208 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_208
  exact h


theorem Stage_zero.parallel_generatedRoot_209.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_209 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_209 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_209
  exact h


theorem Stage_zero.parallel_generatedRoot_210.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_210 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_210 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_210
  exact h


theorem Stage_zero.parallel_generatedRoot_211.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_211 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_211 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_211
  exact h


theorem Stage_zero.parallel_generatedRoot_212.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_212 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_212 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_212
  exact h


theorem Stage_zero.parallel_generatedRoot_213.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_213 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_213 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_213
  exact h


theorem Stage_zero.parallel_generatedRoot_214.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_214 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_214 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_214
  exact h


theorem Stage_zero.parallel_generatedRoot_215.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_215 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_215 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_215
  exact h


theorem Stage_zero.parallel_generatedRoot_216.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_216 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_216 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_216
  exact h


theorem Stage_zero.parallel_generatedRoot_217.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_217 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_217 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_217
  exact h


theorem Stage_zero.parallel_generatedRoot_218.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_218 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_218 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_218
  exact h


theorem Stage_zero.parallel_generatedRoot_219.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_219 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_219 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_219
  exact h


theorem Stage_zero.parallel_generatedRoot_220.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_220 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_220 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_220
  exact h


theorem Stage_zero.parallel_generatedRoot_221.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_221 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_221 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_221
  exact h


theorem Stage_zero.parallel_generatedRoot_222.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_222 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_222 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_222
  exact h


theorem Stage_zero.parallel_generatedRoot_223.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_223 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_223 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_223
  exact h


theorem Stage_zero.parallel_generatedRoot_224.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_224 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_224 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_224
  exact h


theorem Stage_zero.parallel_generatedRoot_225.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_225 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_225 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_225
  exact h


theorem Stage_zero.parallel_generatedRoot_226.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_226 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_226 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_226
  exact h


theorem Stage_zero.parallel_generatedRoot_227.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_227 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_227 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_227
  exact h


theorem Stage_zero.parallel_generatedRoot_228.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_228 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_228 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_228
  exact h


theorem Stage_zero.parallel_generatedRoot_229.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_229 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_229 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_229
  exact h


theorem Stage_zero.parallel_generatedRoot_230.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_230 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_230 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_230
  exact h


theorem Stage_zero.parallel_generatedRoot_231.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_231 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_231 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_231
  exact h


theorem Stage_zero.parallel_generatedRoot_232.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_232 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_232 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_232
  exact h


theorem Stage_zero.parallel_generatedRoot_233.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_233 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_233 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_233
  exact h


theorem Stage_zero.parallel_generatedRoot_234.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_234 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_234 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_234
  exact h


theorem Stage_zero.parallel_generatedRoot_235.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_235 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_235 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_235
  exact h


theorem Stage_zero.parallel_generatedRoot_236.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_236 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_236 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_236
  exact h


theorem Stage_zero.parallel_generatedRoot_237.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_237 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_237 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_237
  exact h


theorem Stage_zero.parallel_generatedRoot_238.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_238 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_238 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_238
  exact h


theorem Stage_zero.parallel_generatedRoot_239.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_239 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_239 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_239
  exact h


theorem Stage_zero.parallel_generatedRoot_240.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_240 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_240 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_240
  exact h


theorem Stage_zero.parallel_generatedRoot_241.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_241 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_241 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_241
  exact h


theorem Stage_zero.parallel_generatedRoot_242.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_242 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_242 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_242
  exact h


theorem Stage_zero.parallel_generatedRoot_243.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_243 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_243 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_243
  exact h


theorem Stage_zero.parallel_generatedRoot_244.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_244 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_244 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_244
  exact h


theorem Stage_zero.parallel_generatedRoot_245.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_245 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_245 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_245
  exact h


theorem Stage_zero.parallel_generatedRoot_246.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_246 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_246 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_246
  exact h


theorem Stage_zero.parallel_generatedRoot_247.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_247 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_247 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_247
  exact h


theorem Stage_zero.parallel_generatedRoot_248.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_248 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_248 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_248
  exact h


theorem Stage_zero.parallel_generatedRoot_249.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_249 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_249 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_249
  exact h


theorem Stage_zero.parallel_generatedRoot_250.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_250 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_250 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_250
  exact h


theorem Stage_zero.parallel_generatedRoot_251.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_251 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_251 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_251
  exact h


theorem Stage_zero.parallel_generatedRoot_252.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_252 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_252 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_252
  exact h


theorem Stage_zero.parallel_generatedRoot_253.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_253 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_253 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_253
  exact h


theorem Stage_zero.parallel_generatedRoot_254.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_254 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_254 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_254
  exact h


theorem Stage_zero.parallel_generatedRoot_255.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_255 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_255 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_255
  exact h


theorem Stage_zero.parallel_generatedRoot_256.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_256 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_256 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_256
  exact h


theorem Stage_zero.parallel_generatedRoot_257.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_257 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_257 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_257
  exact h


theorem Stage_zero.parallel_generatedRoot_258.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_258 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_258 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_258
  exact h


theorem Stage_zero.parallel_generatedRoot_259.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_259 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_259 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_259
  exact h


theorem Stage_zero.parallel_generatedRoot_260.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_260 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_260 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_260
  exact h


theorem Stage_zero.parallel_generatedRoot_261.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_261 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_261 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_261
  exact h


theorem Stage_zero.parallel_generatedRoot_262.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_262 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_262 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_262
  exact h


theorem Stage_zero.parallel_generatedRoot_263.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_263 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_263 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_263
  exact h


theorem Stage_zero.parallel_generatedRoot_264.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_264 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_264 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_264
  exact h


theorem Stage_zero.parallel_generatedRoot_265.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_265 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_265 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_265
  exact h


theorem Stage_zero.parallel_generatedRoot_266.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_266 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_266 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_266
  exact h


theorem Stage_zero.parallel_generatedRoot_267.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_267 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_267 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_267
  exact h


theorem Stage_zero.parallel_generatedRoot_268.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_268 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_268 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_268
  exact h


theorem Stage_zero.parallel_generatedRoot_269.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_269 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_269 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_269
  exact h


theorem Stage_zero.parallel_generatedRoot_270.bridge {tape path params params' x3 x4 outputs}
    (h : Stage_zero.parallel_generatedRoot_270 tape path params x3 x4 outputs) : W_Stage_zero.parallel_generatedRoot_270 params' x3 x4 outputs := by
  delta W_Stage_zero.parallel_generatedRoot_270
  exact h


theorem Stage_zero.generatedRoot.constraints_16.bridge {tape path params params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 outputs}
    (h : Stage_zero.generatedRoot.constraints_16 tape path params w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 outputs) : W_Stage_zero.generatedRoot.constraints_16 params' w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 w_257_0 w_258_0 w_259_0 w_260_0 w_261_0 w_262_0 w_263_0 w_264_0 w_265_0 w_266_0 w_267_0 w_268_0 w_269_0 w_270_0 w_271_0 w_272_0 w_273_0 w_274_0 outputs := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_257.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_258.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_259.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_260.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_261.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_262.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_263.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_264.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_265.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_266.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_267.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_268.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_269.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_270.bridge (h28 i))), h29⟩)


theorem Stage_zero.generatedRoot.constraints_15.bridge {tape path params params' w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0}
    (h : Stage_zero.generatedRoot.constraints_15 tape path params w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0) : W_Stage_zero.generatedRoot.constraints_15 params' w_241_0 w_242_0 w_243_0 w_244_0 w_245_0 w_246_0 w_247_0 w_248_0 w_249_0 w_250_0 w_251_0 w_252_0 w_253_0 w_254_0 w_255_0 w_256_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_241.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_242.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_243.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_244.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_245.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_246.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_247.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_248.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_249.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_250.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_251.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_252.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_253.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_254.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_255.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_256.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_14.bridge {tape path params params' w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0}
    (h : Stage_zero.generatedRoot.constraints_14 tape path params w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0) : W_Stage_zero.generatedRoot.constraints_14 params' w_225_0 w_226_0 w_227_0 w_228_0 w_229_0 w_230_0 w_231_0 w_232_0 w_233_0 w_234_0 w_235_0 w_236_0 w_237_0 w_238_0 w_239_0 w_240_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_225.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_226.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_227.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_228.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_229.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_230.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_231.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_232.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_233.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_234.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_235.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_236.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_237.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_238.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_239.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_240.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_13.bridge {tape path params params' w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0}
    (h : Stage_zero.generatedRoot.constraints_13 tape path params w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0) : W_Stage_zero.generatedRoot.constraints_13 params' w_209_0 w_210_0 w_211_0 w_212_0 w_213_0 w_214_0 w_215_0 w_216_0 w_217_0 w_218_0 w_219_0 w_220_0 w_221_0 w_222_0 w_223_0 w_224_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_209.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_210.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_211.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_212.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_213.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_214.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_215.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_216.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_217.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_218.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_219.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_220.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_221.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_222.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_223.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_224.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_12.bridge {tape path params params' w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0}
    (h : Stage_zero.generatedRoot.constraints_12 tape path params w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0) : W_Stage_zero.generatedRoot.constraints_12 params' w_193_0 w_194_0 w_195_0 w_196_0 w_197_0 w_198_0 w_199_0 w_200_0 w_201_0 w_202_0 w_203_0 w_204_0 w_205_0 w_206_0 w_207_0 w_208_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_193.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_194.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_195.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_196.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_197.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_198.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_199.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_200.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_201.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_202.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_203.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_204.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_205.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_206.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_207.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_208.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_11.bridge {tape path params params' w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0}
    (h : Stage_zero.generatedRoot.constraints_11 tape path params w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0) : W_Stage_zero.generatedRoot.constraints_11 params' w_177_0 w_178_0 w_179_0 w_180_0 w_181_0 w_182_0 w_183_0 w_184_0 w_185_0 w_186_0 w_187_0 w_188_0 w_189_0 w_190_0 w_191_0 w_192_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_177.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_178.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_179.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_180.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_181.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_182.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_183.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_184.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_185.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_186.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_187.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_188.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_189.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_190.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_191.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_192.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_10.bridge {tape path params params' w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0}
    (h : Stage_zero.generatedRoot.constraints_10 tape path params w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0) : W_Stage_zero.generatedRoot.constraints_10 params' w_161_0 w_162_0 w_163_0 w_164_0 w_165_0 w_166_0 w_167_0 w_168_0 w_169_0 w_170_0 w_171_0 w_172_0 w_173_0 w_174_0 w_175_0 w_176_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_161.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_162.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_163.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_164.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_165.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_166.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_167.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_168.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_169.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_170.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_171.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_172.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_173.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_174.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_175.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_176.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_9.bridge {tape path params params' w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0}
    (h : Stage_zero.generatedRoot.constraints_9 tape path params w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0) : W_Stage_zero.generatedRoot.constraints_9 params' w_145_0 w_146_0 w_147_0 w_148_0 w_149_0 w_150_0 w_151_0 w_152_0 w_153_0 w_154_0 w_155_0 w_156_0 w_157_0 w_158_0 w_159_0 w_160_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_145.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_146.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_147.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_148.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_149.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_150.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_151.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_152.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_153.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_154.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_155.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_156.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_157.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_158.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_159.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_160.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_8.bridge {tape path params params' w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0}
    (h : Stage_zero.generatedRoot.constraints_8 tape path params w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0) : W_Stage_zero.generatedRoot.constraints_8 params' w_129_0 w_130_0 w_131_0 w_132_0 w_133_0 w_134_0 w_135_0 w_136_0 w_137_0 w_138_0 w_139_0 w_140_0 w_141_0 w_142_0 w_143_0 w_144_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_129.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_130.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_131.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_132.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_133.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_134.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_135.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_136.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_137.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_138.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_139.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_140.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_141.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_142.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_143.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_144.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_7.bridge {tape path params params' w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0}
    (h : Stage_zero.generatedRoot.constraints_7 tape path params w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0) : W_Stage_zero.generatedRoot.constraints_7 params' w_113_0 w_114_0 w_115_0 w_116_0 w_117_0 w_118_0 w_119_0 w_120_0 w_121_0 w_122_0 w_123_0 w_124_0 w_125_0 w_126_0 w_127_0 w_128_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_113.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_114.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_115.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_116.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_117.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_118.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_119.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_120.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_121.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_122.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_123.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_124.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_125.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_126.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_127.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_128.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_6.bridge {tape path params params' w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0}
    (h : Stage_zero.generatedRoot.constraints_6 tape path params w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0) : W_Stage_zero.generatedRoot.constraints_6 params' w_97_0 w_98_0 w_99_0 w_100_0 w_101_0 w_102_0 w_103_0 w_104_0 w_105_0 w_106_0 w_107_0 w_108_0 w_109_0 w_110_0 w_111_0 w_112_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_97.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_98.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_99.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_100.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_101.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_102.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_103.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_104.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_105.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_106.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_107.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_108.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_109.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_110.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_111.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_112.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_5.bridge {tape path params params' w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0}
    (h : Stage_zero.generatedRoot.constraints_5 tape path params w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0) : W_Stage_zero.generatedRoot.constraints_5 params' w_81_0 w_82_0 w_83_0 w_84_0 w_85_0 w_86_0 w_87_0 w_88_0 w_89_0 w_90_0 w_91_0 w_92_0 w_93_0 w_94_0 w_95_0 w_96_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_81.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_82.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_83.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_84.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_85.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_86.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_87.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_88.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_89.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_90.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_91.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_92.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_93.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_94.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_95.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_96.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_4.bridge {tape path params params' w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0}
    (h : Stage_zero.generatedRoot.constraints_4 tape path params w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0) : W_Stage_zero.generatedRoot.constraints_4 params' w_65_0 w_66_0 w_67_0 w_68_0 w_69_0 w_70_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_77_0 w_78_0 w_79_0 w_80_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_65.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_66.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_67.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_68.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_69.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_70.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_71.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_72.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_73.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_74.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_75.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_76.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_77.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_78.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_79.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_80.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_3.bridge {tape path params params' w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0}
    (h : Stage_zero.generatedRoot.constraints_3 tape path params w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0) : W_Stage_zero.generatedRoot.constraints_3 params' w_49_0 w_50_0 w_51_0 w_52_0 w_53_0 w_54_0 w_55_0 w_56_0 w_57_0 w_58_0 w_59_0 w_60_0 w_61_0 w_62_0 w_63_0 w_64_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_49.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_50.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_51.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_52.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_53.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_54.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_55.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_56.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_57.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_58.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_59.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_60.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_61.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_62.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_63.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_64.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_2.bridge {tape path params params' w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0}
    (h : Stage_zero.generatedRoot.constraints_2 tape path params w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0) : W_Stage_zero.generatedRoot.constraints_2 params' w_33_0 w_34_0 w_35_0 w_36_0 w_37_0 w_38_0 w_39_0 w_40_0 w_41_0 w_42_0 w_43_0 w_44_0 w_45_0 w_46_0 w_47_0 w_48_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_33.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_34.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_35.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_36.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_37.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_38.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_39.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_40.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_41.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_42.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_43.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_44.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_45.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_46.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_47.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_48.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_1.bridge {tape path params params' w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0}
    (h : Stage_zero.generatedRoot.constraints_1 tape path params w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0) : W_Stage_zero.generatedRoot.constraints_1 params' w_17_0 w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_23_0 w_24_0 w_25_0 w_26_0 w_27_0 w_28_0 w_29_0 w_30_0 w_31_0 w_32_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_17.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_18.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_19.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_20.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_21.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_22.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_23.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_24.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_25.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_26.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_27.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_28.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_29.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_30.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_31.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_32.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.constraints_0.bridge {tape path params params' w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0}
    (h : Stage_zero.generatedRoot.constraints_0 tape path params w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0) : W_Stage_zero.generatedRoot.constraints_0 params' w_1_0 w_2_0 w_3_0 w_4_0 w_5_0 w_6_0 w_7_0 w_8_0 w_9_0 w_10_0 w_11_0 w_12_0 w_13_0 w_14_0 w_15_0 w_16_0 := by
  exact (by obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, h30, h31, h32⟩ := h; exact ⟨h1, (fun i ↦ (Stage_zero.parallel_generatedRoot_1.bridge (h2 i))), h3, (fun i ↦ (Stage_zero.parallel_generatedRoot_2.bridge (h4 i))), h5, (fun i ↦ (Stage_zero.parallel_generatedRoot_3.bridge (h6 i))), h7, (fun i ↦ (Stage_zero.parallel_generatedRoot_4.bridge (h8 i))), h9, (fun i ↦ (Stage_zero.parallel_generatedRoot_5.bridge (h10 i))), h11, (fun i ↦ (Stage_zero.parallel_generatedRoot_6.bridge (h12 i))), h13, (fun i ↦ (Stage_zero.parallel_generatedRoot_7.bridge (h14 i))), h15, (fun i ↦ (Stage_zero.parallel_generatedRoot_8.bridge (h16 i))), h17, (fun i ↦ (Stage_zero.parallel_generatedRoot_9.bridge (h18 i))), h19, (fun i ↦ (Stage_zero.parallel_generatedRoot_10.bridge (h20 i))), h21, (fun i ↦ (Stage_zero.parallel_generatedRoot_11.bridge (h22 i))), h23, (fun i ↦ (Stage_zero.parallel_generatedRoot_12.bridge (h24 i))), h25, (fun i ↦ (Stage_zero.parallel_generatedRoot_13.bridge (h26 i))), h27, (fun i ↦ (Stage_zero.parallel_generatedRoot_14.bridge (h28 i))), h29, (fun i ↦ (Stage_zero.parallel_generatedRoot_15.bridge (h30 i))), h31, (fun i ↦ (Stage_zero.parallel_generatedRoot_16.bridge (h32 i)))⟩)


theorem Stage_zero.generatedRoot.bridge {tape path params params' x3 outputs}
    (h : Stage_zero.generatedRoot tape path params x3 outputs) : W_Stage_zero.generatedRoot params' x3 outputs := by
  delta W_Stage_zero.generatedRoot
  exact (by obtain ⟨wit1, h2⟩ := h; exact ⟨⟨wit1.w_1_0, wit1.w_2_0, wit1.w_3_0, wit1.w_4_0, wit1.w_5_0, wit1.w_6_0, wit1.w_7_0, wit1.w_8_0, wit1.w_9_0, wit1.w_10_0, wit1.w_11_0, wit1.w_12_0, wit1.w_13_0, wit1.w_14_0, wit1.w_15_0, wit1.w_16_0, wit1.w_17_0, wit1.w_18_0, wit1.w_19_0, wit1.w_20_0, wit1.w_21_0, wit1.w_22_0, wit1.w_23_0, wit1.w_24_0, wit1.w_25_0, wit1.w_26_0, wit1.w_27_0, wit1.w_28_0, wit1.w_29_0, wit1.w_30_0, wit1.w_31_0, wit1.w_32_0, wit1.w_33_0, wit1.w_34_0, wit1.w_35_0, wit1.w_36_0, wit1.w_37_0, wit1.w_38_0, wit1.w_39_0, wit1.w_40_0, wit1.w_41_0, wit1.w_42_0, wit1.w_43_0, wit1.w_44_0, wit1.w_45_0, wit1.w_46_0, wit1.w_47_0, wit1.w_48_0, wit1.w_49_0, wit1.w_50_0, wit1.w_51_0, wit1.w_52_0, wit1.w_53_0, wit1.w_54_0, wit1.w_55_0, wit1.w_56_0, wit1.w_57_0, wit1.w_58_0, wit1.w_59_0, wit1.w_60_0, wit1.w_61_0, wit1.w_62_0, wit1.w_63_0, wit1.w_64_0, wit1.w_65_0, wit1.w_66_0, wit1.w_67_0, wit1.w_68_0, wit1.w_69_0, wit1.w_70_0, wit1.w_71_0, wit1.w_72_0, wit1.w_73_0, wit1.w_74_0, wit1.w_75_0, wit1.w_76_0, wit1.w_77_0, wit1.w_78_0, wit1.w_79_0, wit1.w_80_0, wit1.w_81_0, wit1.w_82_0, wit1.w_83_0, wit1.w_84_0, wit1.w_85_0, wit1.w_86_0, wit1.w_87_0, wit1.w_88_0, wit1.w_89_0, wit1.w_90_0, wit1.w_91_0, wit1.w_92_0, wit1.w_93_0, wit1.w_94_0, wit1.w_95_0, wit1.w_96_0, wit1.w_97_0, wit1.w_98_0, wit1.w_99_0, wit1.w_100_0, wit1.w_101_0, wit1.w_102_0, wit1.w_103_0, wit1.w_104_0, wit1.w_105_0, wit1.w_106_0, wit1.w_107_0, wit1.w_108_0, wit1.w_109_0, wit1.w_110_0, wit1.w_111_0, wit1.w_112_0, wit1.w_113_0, wit1.w_114_0, wit1.w_115_0, wit1.w_116_0, wit1.w_117_0, wit1.w_118_0, wit1.w_119_0, wit1.w_120_0, wit1.w_121_0, wit1.w_122_0, wit1.w_123_0, wit1.w_124_0, wit1.w_125_0, wit1.w_126_0, wit1.w_127_0, wit1.w_128_0, wit1.w_129_0, wit1.w_130_0, wit1.w_131_0, wit1.w_132_0, wit1.w_133_0, wit1.w_134_0, wit1.w_135_0, wit1.w_136_0, wit1.w_137_0, wit1.w_138_0, wit1.w_139_0, wit1.w_140_0, wit1.w_141_0, wit1.w_142_0, wit1.w_143_0, wit1.w_144_0, wit1.w_145_0, wit1.w_146_0, wit1.w_147_0, wit1.w_148_0, wit1.w_149_0, wit1.w_150_0, wit1.w_151_0, wit1.w_152_0, wit1.w_153_0, wit1.w_154_0, wit1.w_155_0, wit1.w_156_0, wit1.w_157_0, wit1.w_158_0, wit1.w_159_0, wit1.w_160_0, wit1.w_161_0, wit1.w_162_0, wit1.w_163_0, wit1.w_164_0, wit1.w_165_0, wit1.w_166_0, wit1.w_167_0, wit1.w_168_0, wit1.w_169_0, wit1.w_170_0, wit1.w_171_0, wit1.w_172_0, wit1.w_173_0, wit1.w_174_0, wit1.w_175_0, wit1.w_176_0, wit1.w_177_0, wit1.w_178_0, wit1.w_179_0, wit1.w_180_0, wit1.w_181_0, wit1.w_182_0, wit1.w_183_0, wit1.w_184_0, wit1.w_185_0, wit1.w_186_0, wit1.w_187_0, wit1.w_188_0, wit1.w_189_0, wit1.w_190_0, wit1.w_191_0, wit1.w_192_0, wit1.w_193_0, wit1.w_194_0, wit1.w_195_0, wit1.w_196_0, wit1.w_197_0, wit1.w_198_0, wit1.w_199_0, wit1.w_200_0, wit1.w_201_0, wit1.w_202_0, wit1.w_203_0, wit1.w_204_0, wit1.w_205_0, wit1.w_206_0, wit1.w_207_0, wit1.w_208_0, wit1.w_209_0, wit1.w_210_0, wit1.w_211_0, wit1.w_212_0, wit1.w_213_0, wit1.w_214_0, wit1.w_215_0, wit1.w_216_0, wit1.w_217_0, wit1.w_218_0, wit1.w_219_0, wit1.w_220_0, wit1.w_221_0, wit1.w_222_0, wit1.w_223_0, wit1.w_224_0, wit1.w_225_0, wit1.w_226_0, wit1.w_227_0, wit1.w_228_0, wit1.w_229_0, wit1.w_230_0, wit1.w_231_0, wit1.w_232_0, wit1.w_233_0, wit1.w_234_0, wit1.w_235_0, wit1.w_236_0, wit1.w_237_0, wit1.w_238_0, wit1.w_239_0, wit1.w_240_0, wit1.w_241_0, wit1.w_242_0, wit1.w_243_0, wit1.w_244_0, wit1.w_245_0, wit1.w_246_0, wit1.w_247_0, wit1.w_248_0, wit1.w_249_0, wit1.w_250_0, wit1.w_251_0, wit1.w_252_0, wit1.w_253_0, wit1.w_254_0, wit1.w_255_0, wit1.w_256_0, wit1.w_257_0, wit1.w_258_0, wit1.w_259_0, wit1.w_260_0, wit1.w_261_0, wit1.w_262_0, wit1.w_263_0, wit1.w_264_0, wit1.w_265_0, wit1.w_266_0, wit1.w_267_0, wit1.w_268_0, wit1.w_269_0, wit1.w_270_0⟩, (by obtain ⟨h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, h16, h17, h18, h19⟩ := h2; exact ⟨(Stage_zero.generatedRoot.constraints_0.bridge h3), (Stage_zero.generatedRoot.constraints_1.bridge h4), (Stage_zero.generatedRoot.constraints_2.bridge h5), (Stage_zero.generatedRoot.constraints_3.bridge h6), (Stage_zero.generatedRoot.constraints_4.bridge h7), (Stage_zero.generatedRoot.constraints_5.bridge h8), (Stage_zero.generatedRoot.constraints_6.bridge h9), (Stage_zero.generatedRoot.constraints_7.bridge h10), (Stage_zero.generatedRoot.constraints_8.bridge h11), (Stage_zero.generatedRoot.constraints_9.bridge h12), (Stage_zero.generatedRoot.constraints_10.bridge h13), (Stage_zero.generatedRoot.constraints_11.bridge h14), (Stage_zero.generatedRoot.constraints_12.bridge h15), (Stage_zero.generatedRoot.constraints_13.bridge h16), (Stage_zero.generatedRoot.constraints_14.bridge h17), (Stage_zero.generatedRoot.constraints_15.bridge h18), (Stage_zero.generatedRoot.constraints_16.bridge h19)⟩)⟩)
