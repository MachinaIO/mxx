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
  exact ⟨⟨wit.w_3_0, wit.w_7_0, wit.w_12_0, wit.w_17_0, wit.w_22_0, wit.w_27_0, wit.w_32_0, wit.w_37_0, wit.w_43_0, wit.w_44_0, wit.w_46_0, wit.w_56_0, wit.w_57_0, wit.w_60_0, wit.w_62_0, wit.w_63_0, wit.w_63_1, wit.w_63_2, wit.w_63_3, wit.w_63_4, wit.w_63_5, wit.w_63_6, wit.w_63_7, wit.w_72_0, wit.w_73_0, wit.w_73_1, wit.w_73_2, wit.w_73_3, wit.w_73_4, wit.w_73_5, wit.w_73_6, wit.w_73_7, wit.w_82_0, wit.w_83_0, wit.w_83_1, wit.w_83_2, wit.w_83_3, wit.w_83_4, wit.w_83_5, wit.w_83_6, wit.w_83_7, wit.w_92_0, wit.w_93_0, wit.w_93_1, wit.w_93_2, wit.w_93_3, wit.w_93_4, wit.w_93_5, wit.w_93_6, wit.w_93_7, wit.w_102_0, wit.w_103_0, wit.w_103_1, wit.w_103_2, wit.w_103_3, wit.w_103_4, wit.w_103_5, wit.w_103_6, wit.w_103_7, wit.w_112_0, wit.w_113_0, wit.w_113_1, wit.w_113_2, wit.w_113_3, wit.w_113_4, wit.w_113_5, wit.w_113_6, wit.w_113_7, wit.w_123_0, wit.w_124_0, wit.w_126_0, wit.w_133_0, wit.w_135_0, wit.w_136_0, wit.w_136_1, wit.w_136_2, wit.w_136_3, wit.w_136_4, wit.w_136_5, wit.w_136_6, wit.w_136_7, wit.w_145_0, wit.w_146_0, wit.w_146_1, wit.w_146_2, wit.w_146_3, wit.w_146_4, wit.w_146_5, wit.w_146_6, wit.w_146_7, wit.w_155_0, wit.w_156_0, wit.w_156_1, wit.w_156_2, wit.w_156_3, wit.w_156_4, wit.w_156_5, wit.w_156_6, wit.w_156_7, wit.w_165_0, wit.w_166_0, wit.w_166_1, wit.w_166_2, wit.w_166_3, wit.w_166_4, wit.w_166_5, wit.w_166_6, wit.w_166_7, wit.w_175_0, wit.w_176_0, wit.w_176_1, wit.w_176_2, wit.w_176_3, wit.w_176_4, wit.w_176_5, wit.w_176_6, wit.w_176_7, wit.w_185_0, wit.w_186_0, wit.w_186_1, wit.w_186_2, wit.w_186_3, wit.w_186_4, wit.w_186_5, wit.w_186_6, wit.w_186_7⟩, hb⟩
