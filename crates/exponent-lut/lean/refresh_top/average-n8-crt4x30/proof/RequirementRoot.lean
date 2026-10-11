import ReqLabel
import ReqSigma

/-! The facts the requirement checks, from its run (written by `generator/gen_requirement.py`). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
theorem requirementFacts {w_3_0 : Fin 5 → Int} {w_4_0 : Fin 5 → Int} {w_5_0 : Fin 5 → Int} {w_7_0 : Fin 240 → Int} {w_9_0 : Fin 240 → Int} {w_11_0 : Fin 240 → Int} {w_13_0 : Fin 240 → Int} {w_15_0 : Fin 40 → Int} {w_17_0 : Fin 40 → Int} {w_19_0 : Fin 800 → Int} {w_21_0 : Fin 800 → Int} {w_23_0 : Fin 800 → Int} {w_25_0 : Fin 800 → Int} {w_27_0 : Fin 800 → Int} {w_29_0 : Fin 800 → Int} {w_31_0 : Fin 800 → Int} {w_33_0 : Fin 800 → Int} {w_35_0 : Fin 800 → Int} {w_37_0 : Fin 800 → Int} {w_39_0 : Fin 800 → Int} {w_41_0 : Fin 800 → Int} {w_43_0 : Fin 800 → Int} {w_45_0 : Fin 800 → Int} {w_47_0 : Fin 800 → Int} {w_49_0 : Fin 800 → Int} {w_51_0 : Fin 800 → Int} {w_53_0 : Fin 800 → Int} {w_55_0 : Fin 800 → Int} {w_57_0 : Fin 800 → Int} {w_59_0 : Fin 800 → Int} {w_61_0 : Fin 800 → Int} {w_63_0 : Fin 800 → Int} {w_65_0 : Fin 800 → Int} {w_68_0 : Int} {w_70_0 : Int} {w_77_0 : Fin 8 → Int} {out : Bool}
    (h : W_Requirement_0.generatedRoot W_GeneratedClaim.requirement_0_params (w_3_0, w_4_0, w_5_0, w_7_0, w_9_0, w_11_0, w_13_0, w_15_0, w_17_0, w_19_0, w_21_0, w_23_0, w_25_0, w_27_0, w_29_0, w_31_0, w_33_0, w_35_0, w_37_0, w_39_0, w_41_0, w_43_0, w_45_0, w_47_0, w_49_0, w_51_0, w_53_0, w_55_0, w_57_0, w_59_0, w_61_0, w_63_0, w_65_0, w_68_0, w_70_0, w_77_0, ()) out)
    (ht : out = true) :
    (∀ k : Fin 5, w_3_0 k = (labelExp k : Int) ∧ w_4_0 k * w_4_0 k = 1 ∧ w_5_0 k * 2 + (1 - w_4_0 k) * 8 = w_3_0 k * 4) ∧ (∀ o : Fin 48, ∑ k : Fin 5, w_7_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 48, ∑ k : Fin 5, w_9_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 48, ∑ k : Fin 5, w_11_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 48, ∑ k : Fin 5, w_13_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 8, ∑ k : Fin 5, w_15_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 8, ∑ k : Fin 5, w_17_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_19_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_21_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_23_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_25_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_27_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_29_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_31_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_33_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_35_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_37_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_39_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_41_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_43_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_45_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_47_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_49_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_51_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_53_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_55_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_57_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_59_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_61_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_63_0 (cellIndex o k) = 1) ∧ (∀ o : Fin 160, ∑ k : Fin 5, w_65_0 (cellIndex o k) = 1) ∧ w_68_0 = 1 - w_70_0 ∧ (∀ i : Fin 8, w_77_0 i = 1 - ((∑ k : Fin 5, w_15_0 (cellIndex i k) * w_3_0 k + ∑ k : Fin 5, w_17_0 (cellIndex i k) * w_3_0 k) % 8) / 4) := by
  obtain ⟨w, h⟩ := h
  unfold W_Requirement_0.generatedRoot.body at h
  extract_lets -merge _params w_3_0 w_4_0 w_5_0 w_7_0 w_9_0 w_11_0 w_13_0 w_15_0 w_17_0 w_19_0 w_21_0 w_23_0 w_25_0 w_27_0 w_29_0 w_31_0 w_33_0 w_35_0 w_37_0 w_39_0 w_41_0 w_43_0 w_45_0 w_47_0 w_49_0 w_51_0 w_53_0 w_55_0 w_57_0 w_59_0 w_61_0 w_63_0 w_65_0 w_68_0 w_70_0 w_77_0 w_0_0 w_1_0 w_2_0 w_6_0 w_8_0 w_10_0 w_12_0 w_14_0 w_16_0 w_18_0 w_20_0 w_22_0 w_24_0 w_26_0 w_28_0 w_30_0 w_32_0 w_34_0 w_36_0 w_38_0 w_40_0 w_42_0 w_44_0 w_46_0 w_48_0 w_50_0 w_52_0 w_54_0 w_56_0 w_58_0 w_60_0 w_62_0 w_64_0 w_66_0 w_67_0 w_69_0 w_71_0 w_72_0 w_73_0 w_74_0 w_75_0 w_76_0 w_78_0 at h
  unfold W_Requirement_0.generatedRoot.constraints_0 at h
  obtain ⟨-, h6_0, -, h8_0, -, h10_0, -, h12_0, -, h14_0, -, h16_0, -, h18_0, -, h20_0, -, h22_0, -, h24_0, -, h26_0, -, h28_0, -, h30_0, -, h32_0, -, h34_0, -, h36_0, h⟩ := h
  unfold W_Requirement_0.generatedRoot.constraints_1 at h
  obtain ⟨-, h38_0, -, h40_0, -, h42_0, -, h44_0, -, h46_0, -, h48_0, -, h50_0, -, h52_0, -, h54_0, -, h56_0, -, h58_0, -, h60_0, -, h62_0, -, h64_0, -, h66_0, -, h78_0, hout⟩ := h

  subst ht
  have a222 := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 8, w_77_0 ⟨i, hi⟩ = 1 - ((∑ k : Fin 5, w_15_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) * w_3_0 k + ∑ k : Fin 5, w_17_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) * w_3_0 k) % 8) / 4)
    (fun i c n _ hb hn ↦ reqSigma hb hn) h78_0 hout.symm
  obtain ⟨h220, f222⟩ := a222
  simp only [w_76_0, w_74_0, w_73_0, w_72_0, w_71_0, w_69_0, w_67_0, w_75_0, decide_eq_true_eq,
    ite_add_ite_eq_two] at h220
  obtain ⟨h210, f212⟩ := h220
  obtain ⟨i66, f66⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_65_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck66 hb hn) h66_0 (by simpa using h210)
  obtain ⟨i64, f64⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_63_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck64 hb hn) h64_0 (by simpa using i66)
  obtain ⟨i62, f62⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_61_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck62 hb hn) h62_0 (by simpa using i64)
  obtain ⟨i60, f60⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_59_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck60 hb hn) h60_0 (by simpa using i62)
  obtain ⟨i58, f58⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_57_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck58 hb hn) h58_0 (by simpa using i60)
  obtain ⟨i56, f56⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_55_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck56 hb hn) h56_0 (by simpa using i58)
  obtain ⟨i54, f54⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_53_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck54 hb hn) h54_0 (by simpa using i56)
  obtain ⟨i52, f52⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_51_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck52 hb hn) h52_0 (by simpa using i54)
  obtain ⟨i50, f50⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_49_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck50 hb hn) h50_0 (by simpa using i52)
  obtain ⟨i48, f48⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_47_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck48 hb hn) h48_0 (by simpa using i50)
  obtain ⟨i46, f46⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_45_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck46 hb hn) h46_0 (by simpa using i48)
  obtain ⟨i44, f44⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_43_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck44 hb hn) h44_0 (by simpa using i46)
  obtain ⟨i42, f42⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_41_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck42 hb hn) h42_0 (by simpa using i44)
  obtain ⟨i40, f40⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_39_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck40 hb hn) h40_0 (by simpa using i42)
  obtain ⟨i38, f38⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_37_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck38 hb hn) h38_0 (by simpa using i40)
  obtain ⟨i36, f36⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_35_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck36 hb hn) h36_0 (by simpa using i38)
  obtain ⟨i34, f34⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_33_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck34 hb hn) h34_0 (by simpa using i36)
  obtain ⟨i32, f32⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_31_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck32 hb hn) h32_0 (by simpa using i34)
  obtain ⟨i30, f30⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_29_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck30 hb hn) h30_0 (by simpa using i32)
  obtain ⟨i28, f28⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_27_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck28 hb hn) h28_0 (by simpa using i30)
  obtain ⟨i26, f26⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_25_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck26 hb hn) h26_0 (by simpa using i28)
  obtain ⟨i24, f24⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_23_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck24 hb hn) h24_0 (by simpa using i26)
  obtain ⟨i22, f22⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_21_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck22 hb hn) h22_0 (by simpa using i24)
  obtain ⟨i20, f20⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 160, ∑ k : Fin 5, w_19_0 (cellIndex (⟨i, hi⟩ : Fin 160) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck20 hb hn) h20_0 (by simpa using i22)
  obtain ⟨i18, f18⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 8, ∑ k : Fin 5, w_17_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck18 hb hn) h18_0 (by simpa using i20)
  obtain ⟨i16, f16⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 8, ∑ k : Fin 5, w_15_0 (cellIndex (⟨i, hi⟩ : Fin 8) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck16 hb hn) h16_0 (by simpa using i18)
  obtain ⟨i14, f14⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 48, ∑ k : Fin 5, w_13_0 (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck14 hb hn) h14_0 (by simpa using i16)
  obtain ⟨i12, f12⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 48, ∑ k : Fin 5, w_11_0 (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck12 hb hn) h12_0 (by simpa using i14)
  obtain ⟨i10, f10⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 48, ∑ k : Fin 5, w_9_0 (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck10 hb hn) h10_0 (by simpa using i12)
  obtain ⟨i8, f8⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 48, ∑ k : Fin 5, w_7_0 (cellIndex (⟨i, hi⟩ : Fin 48) k) = 1)
    (fun i c n _ hb hn ↦ reqCheck8 hb hn) h8_0 (by simpa using i10)
  obtain ⟨i6, f6⟩ := IterRuns.all_of_step (fun i ↦ ∀ hi : i < 5, w_3_0 ⟨i, hi⟩ = (labelExp ⟨i, hi⟩ : Int) ∧
      w_4_0 ⟨i, hi⟩ * w_4_0 ⟨i, hi⟩ = 1 ∧ w_5_0 ⟨i, hi⟩ * 2 + (1 - w_4_0 ⟨i, hi⟩) * 8 = w_3_0 ⟨i, hi⟩ * 4)
    (fun i c n _ hb hn ↦ reqLabel hb hn) h6_0 (by simpa using i8)
  refine ⟨fun k ↦ f6 k.val (by simpa using k.isLt) k.isLt, fun o ↦ f8 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f10 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f12 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f14 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f16 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f18 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f20 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f22 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f24 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f26 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f28 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f30 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f32 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f34 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f36 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f38 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f40 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f42 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f44 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f46 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f48 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f50 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f52 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f54 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f56 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f58 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f60 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f62 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f64 o.val (by simpa using o.isLt) o.isLt, fun o ↦ f66 o.val (by simpa using o.isLt) o.isLt, f212.symm ▸ rfl,
    fun i ↦ f222 i.val (by simpa using i.isLt) i.isLt⟩

end RefreshTopProof
