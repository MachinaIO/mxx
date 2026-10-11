import TopLemmas
import W_Stage_preprocess_0_0
import ExponentScale
import ExponentRuntime

/-! The RHS of the PRF cells the preprocessing computes (written by `generator/gen_pre_rhs.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `35`. -/
theorem preRhs35 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_35 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q6 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q6 N 1 10 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_35 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q6 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q6 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q6 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q6 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `35`. -/
theorem preTop35 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_35 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q6 10 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 53, 57, 44, 32, 49, 52, 53, 44, 32, 50, 57, 44, 32, 49, 54, 51, 44, 32, 56, 57, 44, 32, 49, 49, 57, 44, 32, 49, 55, 56, 44, 32, 50, 49, 44, 32, 49, 53, 52, 44, 32, 49, 54, 50, 44, 32, 49, 55, 51, 44, 32, 49, 51, 55, 44, 32, 49, 54, 50, 44, 32, 50, 52, 52, 44, 32, 54, 51, 44, 32, 50, 52, 55, 44, 32, 50, 50, 56, 44, 32, 49, 53, 52, 44, 32, 49, 57, 54, 44, 32, 55, 44, 32, 49, 51, 55, 44, 32, 50, 51, 54, 44, 32, 49, 51, 57, 44, 32, 54, 55, 44, 32, 50, 48, 51, 44, 32, 49, 55, 52, 44, 32, 53, 52, 44, 32, 51, 48, 44, 32, 57, 56, 44, 32, 50, 53, 53, 44, 32, 49, 50, 48, 44, 32, 49, 53, 54, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_35 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_35.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `39`. -/
theorem preRhs39 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_39 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_39 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `39`. -/
theorem preTop39 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_39 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 53, 57, 44, 32, 49, 52, 53, 44, 32, 50, 57, 44, 32, 49, 54, 51, 44, 32, 56, 57, 44, 32, 49, 49, 57, 44, 32, 49, 55, 56, 44, 32, 50, 49, 44, 32, 49, 53, 52, 44, 32, 49, 54, 50, 44, 32, 49, 55, 51, 44, 32, 49, 51, 55, 44, 32, 49, 54, 50, 44, 32, 50, 52, 52, 44, 32, 54, 51, 44, 32, 50, 52, 55, 44, 32, 50, 50, 56, 44, 32, 49, 53, 52, 44, 32, 49, 57, 54, 44, 32, 55, 44, 32, 49, 51, 55, 44, 32, 50, 51, 54, 44, 32, 49, 51, 57, 44, 32, 54, 55, 44, 32, 50, 48, 51, 44, 32, 49, 55, 52, 44, 32, 53, 52, 44, 32, 51, 48, 44, 32, 57, 56, 44, 32, 50, 53, 53, 44, 32, 49, 50, 48, 44, 32, 49, 53, 54, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_39 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_39.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `210`. -/
theorem preRhs210 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_210 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q6 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q6 N 1 10 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_210 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q6 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q6 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q6 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q6 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `210`. -/
theorem preTop210 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_210 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q6 10 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 44, 32, 50, 54, 44, 32, 49, 51, 52, 44, 32, 49, 56, 52, 44, 32, 49, 56, 50, 44, 32, 49, 57, 52, 44, 32, 55, 53, 44, 32, 52, 49, 44, 32, 50, 52, 57, 44, 32, 51, 57, 44, 32, 49, 53, 52, 44, 32, 50, 50, 44, 32, 55, 55, 44, 32, 49, 51, 50, 44, 32, 54, 50, 44, 32, 51, 55, 44, 32, 50, 52, 53, 44, 32, 50, 52, 50, 44, 32, 49, 55, 50, 44, 32, 49, 54, 53, 44, 32, 49, 55, 53, 44, 32, 56, 49, 44, 32, 49, 56, 55, 44, 32, 49, 54, 51, 44, 32, 49, 48, 57, 44, 32, 57, 52, 44, 32, 49, 53, 55, 44, 32, 49, 48, 51, 44, 32, 50, 52, 54, 44, 32, 53, 44, 32, 49, 49, 57, 44, 32, 57, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_210 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_210.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `214`. -/
theorem preRhs214 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_214 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_214 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `214`. -/
theorem preTop214 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_214 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 55, 101, 55, 98, 52, 99, 99, 56, 98, 48, 100, 51, 52, 53, 99, 99, 102, 57, 55, 57, 50, 48, 102, 102, 48, 48, 55, 49, 48, 55, 52, 49, 57, 54, 56, 54, 56, 101, 56, 51, 51, 56, 101, 97, 51, 49, 99, 97, 52, 100, 102, 52, 54, 48, 99, 49, 49, 99, 99, 100, 53, 97, 51, 54, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 56, 44, 32, 50, 54, 44, 32, 49, 51, 52, 44, 32, 49, 56, 52, 44, 32, 49, 56, 50, 44, 32, 49, 57, 52, 44, 32, 55, 53, 44, 32, 52, 49, 44, 32, 50, 52, 57, 44, 32, 51, 57, 44, 32, 49, 53, 52, 44, 32, 50, 50, 44, 32, 55, 55, 44, 32, 49, 51, 50, 44, 32, 54, 50, 44, 32, 51, 55, 44, 32, 50, 52, 53, 44, 32, 50, 52, 50, 44, 32, 49, 55, 50, 44, 32, 49, 54, 53, 44, 32, 49, 55, 53, 44, 32, 56, 49, 44, 32, 49, 56, 55, 44, 32, 49, 54, 51, 44, 32, 49, 48, 57, 44, 32, 57, 52, 44, 32, 49, 53, 55, 44, 32, 49, 48, 51, 44, 32, 50, 52, 54, 44, 32, 53, 44, 32, 49, 49, 57, 44, 32, 57, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_214 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_214.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `381`. -/
theorem preRhs381 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_381 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_381 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `381`. -/
theorem preTop381 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_381 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_381 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_381.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `385`. -/
theorem preRhs385 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_385 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_385 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `385`. -/
theorem preTop385 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_385 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_385 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_385.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `20`. -/
theorem preRhs20 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_20 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_20 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `20`. -/
theorem preTop20 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_20 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_20 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_20.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `24`. -/
theorem preRhs24 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_24 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_24 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `24`. -/
theorem preTop24 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_24 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_24 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_24.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `425`. -/
theorem preRhs425 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_425 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_425 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `425`. -/
theorem preTop425 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_425 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_425 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_425.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `429`. -/
theorem preRhs429 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_429 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_429 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `429`. -/
theorem preTop429 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_429 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_429 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_429.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `513`. -/
theorem preRhs513 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_513 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_513 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `513`. -/
theorem preTop513 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_513 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_513 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_513.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `517`. -/
theorem preRhs517 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_517 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_517 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `517`. -/
theorem preTop517 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_517 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_517 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_517.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `469`. -/
theorem preRhs469 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_469 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_469 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `469`. -/
theorem preTop469 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_469 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_469 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_469.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `473`. -/
theorem preRhs473 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_473 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_473 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `473`. -/
theorem preTop473 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_473 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_473 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_473.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `557`. -/
theorem preRhs557 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_557 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((p4 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q5 N 1 8 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_557 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h33_0 hg'] at h34_0
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h46_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h65_0
  have hsp : w_32_0 = fun _ _ ↦ reducePoly Q5 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h32_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_59_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h59_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_42_0 = fun _ _ ↦ reducePoly Q5 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h42_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_41_0, w_39_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q5 N sp • w_17_0 + w_47_0 = _
    simp only [w_47_0, w_45_0, w_43_0, w_37_0, w_35_0, w_44_0, w_36_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q5 N sp • w_54_0 + w_66_0 = _
    simp only [w_66_0, w_64_0, w_62_0, w_61_0, w_60_0, w_63_0, w_36_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `557`. -/
theorem preTop557 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_557 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q5 8 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 48, 41, 47, 48, 47, 49] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_557 at h
  extract_lets -merge _params _backend w_0_0 w_4_0 w_10_0 w_21_0 w_24_0 w_31_0 w_38_0 w_40_0 w_58_0 w_36_0 w_39_0 w_41_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  obtain ⟨w_11_0, h⟩ := h
  extract_lets -merge w_12_0 w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  obtain ⟨w_17_0, h⟩ := h
  obtain ⟨w_18_0, h⟩ := h
  obtain ⟨w_19_0, h⟩ := h
  extract_lets -merge w_20_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 at h
  obtain ⟨w_28_0, h⟩ := h
  obtain ⟨w_29_0, h⟩ := h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  extract_lets -merge w_44_0 at h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  extract_lets -merge w_35_0 w_37_0 at h
  obtain ⟨w_42_0, h⟩ := h
  extract_lets -merge w_43_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  extract_lets -merge w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 w_51_0 at h
  obtain ⟨w_52_0, h⟩ := h
  obtain ⟨w_53_0, h⟩ := h
  obtain ⟨w_54_0, h⟩ := h
  extract_lets -merge w_63_0 at h
  obtain ⟨w_55_0, h⟩ := h
  obtain ⟨w_56_0, h⟩ := h
  extract_lets -merge w_57_0 at h
  obtain ⟨w_59_0, h⟩ := h
  extract_lets -merge w_60_0 w_61_0 w_62_0 w_64_0 at h
  obtain ⟨w_65_0, h⟩ := h
  extract_lets -merge w_66_0 at h
  obtain ⟨w_67_0, h⟩ := h
  obtain ⟨w_68_0, h⟩ := h
  extract_lets -merge w_69_0 w_70_0 at h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  obtain ⟨w_75_0, h⟩ := h
  obtain ⟨w_76_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_557.constraints_3] at h
  obtain ⟨-, -, -, h1_0, -, -, -, h2_0, h3_0, h5_0, -, -, -, h7_0, -, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h17_0
  · exact hashSample_eq h54_0

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS of cell `i` the preprocessing computes in `561`. -/
theorem preRhs561 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_561 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N sp)
    (hone : x.2.2.2.2.2.2.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly Q0 N sp • ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] j +
          ![outputs.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.1] j =
        reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1 *
          unitPair sp j) • g + reduceMatrix Q0 N 1 6 (eB j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_561 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h51_0 hg'] at h52_0
  have hgg := centeredRebaseRuns_unique h52_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h64_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h83_0
  have hsp : w_50_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h50_0 (fun _ _ ↦ sp)
      (by show x.2.2.2.2.2.1 = _; rw [hsec]; rfl)
  have hone' : w_77_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h77_0 (fun _ _ ↦ 1) (by
      show x.2.2.2.2.2.2.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : w_60_0 = fun _ _ ↦ reducePoly Q0 N (message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1) :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h60_0
      (fun _ _ ↦ message (n := N) x.2.2.2.2.2.2.1 x.2.2.2.2.2.2.2.1)
      (by simp only [w_59_0, w_57_0, multiplyMonomial_liftInteger]; rfl)
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N sp • w_33_0 + w_65_0 = _
    simp only [w_65_0, w_63_0, w_61_0, w_55_0, w_53_0, w_62_0, w_54_0, hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show reducePoly Q0 N sp • w_72_0 + w_84_0 = _
    simp only [w_84_0, w_82_0, w_80_0, w_79_0, w_78_0, w_81_0, w_54_0, hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The RHS tops of cell `i` the preprocessing samples in `561`. -/
theorem preTop561 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} {i : Nat} {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_561 Backend.backend model params i x outputs) :
    ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] = rhsTop model x.1 Q0 6 [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 99, 55, 51, 57, 49, 98, 98, 99, 50, 56, 97, 100, 57, 56, 99, 48, 52, 101, 57, 50, 52, 48, 97, 51, 100, 102, 52, 98, 48, 57, 55, 52, 102, 48, 97, 100, 57, 51, 52, 52, 102, 57, 55, 51, 54, 99, 100, 100, 100, 54, 51, 56, 55, 56, 98, 98, 98, 102, 56, 98, 50, 48, 51, 51, 47, 69, 120, 112, 111, 110, 101, 110, 116, 76, 117, 116, 80, 114, 111, 103, 114, 97, 109, 73, 100, 40, 91, 49, 54, 49, 44, 32, 49, 55, 57, 44, 32, 50, 48, 55, 44, 32, 49, 54, 54, 44, 32, 49, 53, 50, 44, 32, 49, 55, 51, 44, 32, 49, 56, 48, 44, 32, 53, 53, 44, 32, 49, 57, 48, 44, 32, 50, 49, 57, 44, 32, 50, 51, 57, 44, 32, 49, 55, 54, 44, 32, 53, 55, 44, 32, 51, 53, 44, 32, 50, 49, 56, 44, 32, 57, 53, 44, 32, 49, 53, 44, 32, 49, 53, 50, 44, 32, 50, 49, 50, 44, 32, 56, 50, 44, 32, 49, 48, 51, 44, 32, 52, 52, 44, 32, 49, 50, 53, 44, 32, 49, 55, 44, 32, 49, 50, 51, 44, 32, 57, 53, 44, 32, 49, 57, 50, 44, 32, 54, 53, 44, 32, 49, 56, 49, 44, 32, 50, 53, 53, 44, 32, 55, 55, 44, 32, 49, 53, 56, 93, 41, 47, 82, 104, 115, 73, 100, 40, 49, 41, 47, 49, 47, 50] i := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_561 at h
  extract_lets -merge _params _backend w_0_0 w_13_0 w_25_0 w_37_0 w_41_0 w_49_0 w_56_0 w_58_0 w_76_0 w_2_0 w_14_0 w_54_0 w_57_0 w_59_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_3_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 at h
  obtain ⟨w_8_0, h⟩ := h
  obtain ⟨w_9_0, h⟩ := h
  extract_lets -merge w_10_0 at h
  obtain ⟨w_11_0, h⟩ := h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  obtain ⟨w_21_0, h⟩ := h
  extract_lets -merge w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  obtain ⟨w_26_0, h⟩ := h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 w_29_0 at h
  obtain ⟨w_30_0, h⟩ := h
  obtain ⟨w_31_0, h⟩ := h
  obtain ⟨w_32_0, h⟩ := h
  obtain ⟨w_33_0, h⟩ := h
  obtain ⟨w_34_0, h⟩ := h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 at h
  obtain ⟨w_38_0, h⟩ := h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_42_0, h⟩ := h
  obtain ⟨w_43_0, h⟩ := h
  extract_lets -merge w_44_0 w_45_0 at h
  obtain ⟨w_46_0, h⟩ := h
  obtain ⟨w_47_0, h⟩ := h
  obtain ⟨w_48_0, h⟩ := h
  obtain ⟨w_50_0, h⟩ := h
  extract_lets -merge w_62_0 at h
  obtain ⟨w_51_0, h⟩ := h
  obtain ⟨w_52_0, h⟩ := h
  extract_lets -merge w_53_0 w_55_0 at h
  obtain ⟨w_60_0, h⟩ := h
  extract_lets -merge w_61_0 w_63_0 at h
  obtain ⟨w_64_0, h⟩ := h
  extract_lets -merge w_65_0 at h
  obtain ⟨w_66_0, h⟩ := h
  obtain ⟨w_67_0, h⟩ := h
  extract_lets -merge w_68_0 w_69_0 at h
  obtain ⟨w_70_0, h⟩ := h
  obtain ⟨w_71_0, h⟩ := h
  obtain ⟨w_72_0, h⟩ := h
  extract_lets -merge w_81_0 at h
  obtain ⟨w_73_0, h⟩ := h
  obtain ⟨w_74_0, h⟩ := h
  extract_lets -merge w_75_0 at h
  obtain ⟨w_77_0, h⟩ := h
  extract_lets -merge w_78_0 w_79_0 w_80_0 w_82_0 at h
  obtain ⟨w_83_0, h⟩ := h
  extract_lets -merge w_84_0 at h
  obtain ⟨w_85_0, h⟩ := h
  obtain ⟨w_86_0, h⟩ := h
  extract_lets -merge w_87_0 w_88_0 at h
  obtain ⟨w_89_0, h⟩ := h
  obtain ⟨w_90_0, h⟩ := h
  obtain ⟨w_91_0, h⟩ := h
  obtain ⟨w_92_0, h⟩ := h
  obtain ⟨w_93_0, h⟩ := h
  obtain ⟨w_94_0, h⟩ := h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_1, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_2, W_Stage_preprocess_0_0.parallel_generatedRoot_561.constraints_3] at h
  obtain ⟨-, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, -, h9_0, h11_0, h12_0, -, h14_0, h14_0_1, h15_0, h16_0, -, -, -, h18_0, h20_0, -, -, -, h21_0, h23_0, h24_0, h14_0_2, h14_0_3, h26_0, h27_0, -, -, -, -, -, -, h30_0, -, -, -, -, -, -, h31_0, h32_0, h33_0, h34_0, h35_0, h14_0_4, h14_0_5, h38_0, h39_0, h14_0_6, h14_0_7, h42_0, h43_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h47_0, h48_0, h50_0, h51_0, h52_0, h60_0, h64_0, h66_0, h67_0, -, -, -, -, -, -, h70_0, h71_0, h72_0, h73_0, h74_0, h77_0, h83_0, h85_0, h86_0, -, -, -, -, -, -, h89_0, h90_0, -, -, -, -, -, -, h91_0, h92_0, -, -, -, -, -, -, h93_0, h94_0, hout⟩ := h
  subst hout
  funext j
  fin_cases j
  · exact hashSample_eq h33_0
  · exact hashSample_eq h72_0

end RefreshTopProof.Part_0_0
