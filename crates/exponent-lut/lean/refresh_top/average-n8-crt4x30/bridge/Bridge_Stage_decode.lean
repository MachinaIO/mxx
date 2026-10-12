import Stage_decode
import W_Stage_decode
import ExponentSampling

/-! Every run of the average-case relations of `Stage_decode` is a run of its worst-case relations
(written by `generator/gen_bridge.py`). -/

set_option maxRecDepth 65536
set_option maxHeartbeats 64000000

theorem Stage_decode.iterRuns_mono {State : Type} {f g : Nat → State → State → Prop} {count : Nat}
    {a b : State} (h : MxxIR.IterRuns f count a b) (hfg : ∀ i c n, f i c n → g i c n) :
    MxxIR.IterRuns g count a b := by
  induction h with
  | zero s => exact .zero s
  | step _ hs ih => exact .step ih (hfg _ _ _ hs)


theorem Stage_decode.generatedRoot.bridge {backend hashModel tape path params params' inputs outputs}
    (h : Stage_decode.generatedRoot backend hashModel tape path params inputs outputs) : W_Stage_decode.generatedRoot backend hashModel params' inputs outputs := by
  delta W_Stage_decode.generatedRoot
  obtain ⟨wit, hb⟩ := h
  exact ⟨⟨wit.w_2_0, wit.w_6_0, wit.w_11_0, wit.w_16_0, wit.w_21_0, wit.w_25_0, wit.w_30_0, wit.w_35_0, wit.w_40_0, wit.w_45_0, wit.w_50_0, wit.w_55_0, wit.w_61_0, wit.w_62_0, wit.w_64_0, wit.w_74_0, wit.w_75_0, wit.w_79_0, wit.w_80_0, wit.w_80_1, wit.w_80_2, wit.w_80_3, wit.w_80_4, wit.w_80_5, wit.w_80_6, wit.w_80_7, wit.w_89_0, wit.w_90_0, wit.w_90_1, wit.w_90_2, wit.w_90_3, wit.w_90_4, wit.w_90_5, wit.w_90_6, wit.w_90_7, wit.w_99_0, wit.w_100_0, wit.w_100_1, wit.w_100_2, wit.w_100_3, wit.w_100_4, wit.w_100_5, wit.w_100_6, wit.w_100_7, wit.w_109_0, wit.w_110_0, wit.w_110_1, wit.w_110_2, wit.w_110_3, wit.w_110_4, wit.w_110_5, wit.w_110_6, wit.w_110_7, wit.w_119_0, wit.w_120_0, wit.w_120_1, wit.w_120_2, wit.w_120_3, wit.w_120_4, wit.w_120_5, wit.w_120_6, wit.w_120_7, wit.w_129_0, wit.w_130_0, wit.w_130_1, wit.w_130_2, wit.w_130_3, wit.w_130_4, wit.w_130_5, wit.w_130_6, wit.w_130_7, wit.w_140_0, wit.w_144_0, wit.w_149_0, wit.w_154_0, wit.w_158_0, wit.w_159_0, wit.w_161_0, wit.w_169_0, wit.w_170_0, wit.w_170_1, wit.w_170_2, wit.w_170_3, wit.w_170_4, wit.w_170_5, wit.w_170_6, wit.w_170_7, wit.w_179_0, wit.w_180_0, wit.w_180_1, wit.w_180_2, wit.w_180_3, wit.w_180_4, wit.w_180_5, wit.w_180_6, wit.w_180_7, wit.w_189_0, wit.w_190_0, wit.w_190_1, wit.w_190_2, wit.w_190_3, wit.w_190_4, wit.w_190_5, wit.w_190_6, wit.w_190_7, wit.w_199_0, wit.w_200_0, wit.w_200_1, wit.w_200_2, wit.w_200_3, wit.w_200_4, wit.w_200_5, wit.w_200_6, wit.w_200_7, wit.w_209_0, wit.w_210_0, wit.w_210_1, wit.w_210_2, wit.w_210_3, wit.w_210_4, wit.w_210_5, wit.w_210_6, wit.w_210_7, wit.w_219_0, wit.w_220_0, wit.w_220_1, wit.w_220_2, wit.w_220_3, wit.w_220_4, wit.w_220_5, wit.w_220_6, wit.w_220_7⟩, hb⟩
