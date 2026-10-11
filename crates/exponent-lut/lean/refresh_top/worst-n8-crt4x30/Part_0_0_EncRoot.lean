import TopLemmas
import Stage_encrypt_0_0

/-! The encryptions of the encrypt stage's root (written by `generator/gen_enc_root.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `25_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_25_0 {model : HashModel}
    {w_0_0 : ExactMatrix Q6 8 1 10}
    {w_1_0 : ExactMatrix Q6 8 1 10}
    {w_2_0 : ExactMatrix Q6 8 1 10}
    {w_3_0 : ExactMatrix Q6 8 1 10}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_17_0 : Int}
    {w_19_0 : Int}
    {w_5_0 : ExactMatrix Q6 8 1 1}
    {w_6_0 : ExactMatrix Q6 8 1 10}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_8_0 : ExactMatrix Q6 8 1 1}
    {w_9_0 : ExactMatrix Q6 8 1 10}
    {w_10_0 : ExactMatrix Q6 8 1 10}
    {w_12_0 : ExactMatrix Q6 8 1 1}
    {w_13_0 : ExactMatrix Q5 8 1 10}
    {w_14_0 : ExactMatrix Q6 8 1 10}
    {w_15_0 : ExactMatrix Q6 8 1 10}
    {w_16_0 : ExactMatrix Q6 8 1 1}
    {w_18_0 : ExactMatrix Q6 8 1 1}
    {w_20_0 : ExactMatrix Q6 8 1 1}
    {w_21_0 : ExactMatrix Q6 8 1 1}
    {w_22_0 : ExactMatrix Q6 8 1 10}
    {w_23_0 : ExactMatrix Q6 8 1 10}
    {w_24_0 : ExactMatrix Q6 8 1 10}
    {w_25_0 : ExactMatrix Q6 8 1 10}
    {w_26_0 : ExactMatrix Q6 8 1 10}
    {w_27_0 : ExactMatrix Q6 8 1 10}
    {w_28_0 : ExactMatrix Q6 8 1 10}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_30_0 : ExactMatrix Q6 8 1 1}
    {w_31_0 : ExactMatrix Q6 8 1 10}
    {w_32_0 : ExactMatrix Q6 8 1 10}
    {w_33_0 : ExactMatrix Q6 8 1 10}
    {w_34_0 : ExactMatrix Q6 8 1 10}
    {w_35_0 : ExactMatrix Q6 8 1 10}
    (e_6_0 : w_6_0 = matrixMulScalarLeft w_5_0 w_0_0)
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_9_0 : w_9_0 = matrixMulScalarLeft w_8_0 w_2_0)
    (e_10_0 : w_10_0 = matrixAdd w_6_0 w_9_0)
    (e_15_0 : w_15_0 = matrixMulScalarLeft w_12_0 w_14_0)
    (e_16_0 : w_16_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_18_0 : w_18_0 = liftInteger w_17_0)
    (e_20_0 : w_20_0 = multiplyMonomial w_18_0 w_19_0)
    (e_21_0 : w_21_0 = matrixMulScalarLeft w_16_0 w_20_0)
    (e_22_0 : w_22_0 = matrixMulScalarRight w_15_0 w_21_0)
    (e_23_0 : w_23_0 = matrixSub w_10_0 w_22_0)
    (e_25_0 : w_25_0 = matrixAdd w_23_0 w_24_0)
    (e_26_0 : w_26_0 = matrixMulScalarLeft w_5_0 w_1_0)
    (e_27_0 : w_27_0 = matrixMulScalarLeft w_8_0 w_3_0)
    (e_28_0 : w_28_0 = matrixAdd w_26_0 w_27_0)
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_31_0 : w_31_0 = matrixMulScalarLeft w_30_0 w_14_0)
    (e_32_0 : w_32_0 = matrixMulScalarRight w_31_0 w_21_0)
    (e_33_0 : w_33_0 = matrixSub w_28_0 w_32_0)
    (e_35_0 : w_35_0 = matrixAdd w_33_0 w_34_0)
    (h5_0 : centeredRebaseRuns w_4_0 w_5_0)
    (h8_0 : centeredRebaseRuns w_7_0 w_8_0)
    (h12_0 : centeredRebaseRuns w_11_0 w_12_0)
    (h13_0 : gadgetMatrixRuns Backend.backend 32768 10 w_13_0)
    (h14_0 : centeredRebaseRuns w_13_0 w_14_0)
    (h24_0 : gaussianSample ((4/1)) (26) w_24_0)
    (h30_0 : centeredRebaseRuns w_29_0 w_30_0)
    (h34_0 : gaussianSample ((4/1)) (26) w_34_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_0_0, w_1_0], ![w_2_0, w_3_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_17_0 w_19_0 * unitPair sp j) • gg)
      ![w_25_0, w_35_0] 26 := by
  subst e_6_0 e_7_0 e_9_0 e_10_0 e_15_0 e_16_0 e_18_0 e_20_0 e_21_0 e_22_0 e_23_0 e_26_0 e_27_0 e_28_0 e_29_0 e_31_0 e_32_0 e_33_0
  rw [gadgetMatrixRuns_unique h13_0 hg'] at h14_0
  have hgg := centeredRebaseRuns_unique h14_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h24_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h34_0
  have rsm : w_5_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h5_0)
  have rsp : w_12_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h12_0)
  have rone0 : w_30_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h30_0)
  have rone1 : w_8_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h8_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_25_0 = _
    rw [e_25_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_35_0 = _
    rw [e_35_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `69_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_69_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_47_0 : ExactMatrix Q6 8 1 10}
    {w_48_0 : ExactMatrix Q6 8 1 10}
    {w_49_0 : ExactMatrix Q6 8 1 10}
    {w_50_0 : ExactMatrix Q6 8 1 10}
    {w_61_0 : Int}
    {w_63_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_51_0 : ExactMatrix Q6 8 1 1}
    {w_52_0 : ExactMatrix Q6 8 1 10}
    {w_53_0 : ExactMatrix Q6 8 1 1}
    {w_54_0 : ExactMatrix Q6 8 1 10}
    {w_55_0 : ExactMatrix Q6 8 1 10}
    {w_56_0 : ExactMatrix Q6 8 1 1}
    {w_57_0 : ExactMatrix Q5 8 1 10}
    {w_58_0 : ExactMatrix Q6 8 1 10}
    {w_59_0 : ExactMatrix Q6 8 1 10}
    {w_60_0 : ExactMatrix Q6 8 1 1}
    {w_62_0 : ExactMatrix Q6 8 1 1}
    {w_64_0 : ExactMatrix Q6 8 1 1}
    {w_65_0 : ExactMatrix Q6 8 1 1}
    {w_66_0 : ExactMatrix Q6 8 1 10}
    {w_67_0 : ExactMatrix Q6 8 1 10}
    {w_68_0 : ExactMatrix Q6 8 1 10}
    {w_69_0 : ExactMatrix Q6 8 1 10}
    {w_70_0 : ExactMatrix Q6 8 1 10}
    {w_71_0 : ExactMatrix Q6 8 1 10}
    {w_72_0 : ExactMatrix Q6 8 1 10}
    {w_73_0 : ExactMatrix Q6 8 1 1}
    {w_74_0 : ExactMatrix Q6 8 1 10}
    {w_75_0 : ExactMatrix Q6 8 1 10}
    {w_76_0 : ExactMatrix Q6 8 1 10}
    {w_77_0 : ExactMatrix Q6 8 1 10}
    {w_78_0 : ExactMatrix Q6 8 1 10}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_52_0 : w_52_0 = matrixMulScalarLeft w_51_0 w_47_0)
    (e_54_0 : w_54_0 = matrixMulScalarLeft w_53_0 w_49_0)
    (e_55_0 : w_55_0 = matrixAdd w_52_0 w_54_0)
    (e_59_0 : w_59_0 = matrixMulScalarLeft w_56_0 w_58_0)
    (e_60_0 : w_60_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_62_0 : w_62_0 = liftInteger w_61_0)
    (e_64_0 : w_64_0 = multiplyMonomial w_62_0 w_63_0)
    (e_65_0 : w_65_0 = matrixMulScalarLeft w_60_0 w_64_0)
    (e_66_0 : w_66_0 = matrixMulScalarRight w_59_0 w_65_0)
    (e_67_0 : w_67_0 = matrixSub w_55_0 w_66_0)
    (e_69_0 : w_69_0 = matrixAdd w_67_0 w_68_0)
    (e_70_0 : w_70_0 = matrixMulScalarLeft w_51_0 w_48_0)
    (e_71_0 : w_71_0 = matrixMulScalarLeft w_53_0 w_50_0)
    (e_72_0 : w_72_0 = matrixAdd w_70_0 w_71_0)
    (e_74_0 : w_74_0 = matrixMulScalarLeft w_73_0 w_58_0)
    (e_75_0 : w_75_0 = matrixMulScalarRight w_74_0 w_65_0)
    (e_76_0 : w_76_0 = matrixSub w_72_0 w_75_0)
    (e_78_0 : w_78_0 = matrixAdd w_76_0 w_77_0)
    (h51_0 : centeredRebaseRuns w_4_0 w_51_0)
    (h53_0 : centeredRebaseRuns w_7_0 w_53_0)
    (h56_0 : centeredRebaseRuns w_11_0 w_56_0)
    (h57_0 : gadgetMatrixRuns Backend.backend 32768 10 w_57_0)
    (h58_0 : centeredRebaseRuns w_57_0 w_58_0)
    (h68_0 : gaussianSample ((4/1)) (26) w_68_0)
    (h73_0 : centeredRebaseRuns w_29_0 w_73_0)
    (h77_0 : gaussianSample ((4/1)) (26) w_77_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_47_0, w_48_0], ![w_49_0, w_50_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_61_0 w_63_0 * unitPair sp j) • gg)
      ![w_69_0, w_78_0] 26 := by
  subst e_7_0 e_29_0 e_52_0 e_54_0 e_55_0 e_59_0 e_60_0 e_62_0 e_64_0 e_65_0 e_66_0 e_67_0 e_70_0 e_71_0 e_72_0 e_74_0 e_75_0 e_76_0
  rw [gadgetMatrixRuns_unique h57_0 hg'] at h58_0
  have hgg := centeredRebaseRuns_unique h58_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h68_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h77_0
  have rsm : w_51_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h51_0)
  have rsp : w_56_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h56_0)
  have rone0 : w_53_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h53_0)
  have rone1 : w_73_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h73_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_69_0 = _
    rw [e_69_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_78_0 = _
    rw [e_78_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `112_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_112_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_90_0 : ExactMatrix Q6 8 1 10}
    {w_91_0 : ExactMatrix Q6 8 1 10}
    {w_92_0 : ExactMatrix Q6 8 1 10}
    {w_93_0 : ExactMatrix Q6 8 1 10}
    {w_104_0 : Int}
    {w_106_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_94_0 : ExactMatrix Q6 8 1 1}
    {w_95_0 : ExactMatrix Q6 8 1 10}
    {w_96_0 : ExactMatrix Q6 8 1 1}
    {w_97_0 : ExactMatrix Q6 8 1 10}
    {w_98_0 : ExactMatrix Q6 8 1 10}
    {w_99_0 : ExactMatrix Q6 8 1 1}
    {w_100_0 : ExactMatrix Q5 8 1 10}
    {w_101_0 : ExactMatrix Q6 8 1 10}
    {w_102_0 : ExactMatrix Q6 8 1 10}
    {w_103_0 : ExactMatrix Q6 8 1 1}
    {w_105_0 : ExactMatrix Q6 8 1 1}
    {w_107_0 : ExactMatrix Q6 8 1 1}
    {w_108_0 : ExactMatrix Q6 8 1 1}
    {w_109_0 : ExactMatrix Q6 8 1 10}
    {w_110_0 : ExactMatrix Q6 8 1 10}
    {w_111_0 : ExactMatrix Q6 8 1 10}
    {w_112_0 : ExactMatrix Q6 8 1 10}
    {w_113_0 : ExactMatrix Q6 8 1 10}
    {w_114_0 : ExactMatrix Q6 8 1 10}
    {w_115_0 : ExactMatrix Q6 8 1 10}
    {w_116_0 : ExactMatrix Q6 8 1 1}
    {w_117_0 : ExactMatrix Q6 8 1 10}
    {w_118_0 : ExactMatrix Q6 8 1 10}
    {w_119_0 : ExactMatrix Q6 8 1 10}
    {w_120_0 : ExactMatrix Q6 8 1 10}
    {w_121_0 : ExactMatrix Q6 8 1 10}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_95_0 : w_95_0 = matrixMulScalarLeft w_94_0 w_90_0)
    (e_97_0 : w_97_0 = matrixMulScalarLeft w_96_0 w_92_0)
    (e_98_0 : w_98_0 = matrixAdd w_95_0 w_97_0)
    (e_102_0 : w_102_0 = matrixMulScalarLeft w_99_0 w_101_0)
    (e_103_0 : w_103_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_105_0 : w_105_0 = liftInteger w_104_0)
    (e_107_0 : w_107_0 = multiplyMonomial w_105_0 w_106_0)
    (e_108_0 : w_108_0 = matrixMulScalarLeft w_103_0 w_107_0)
    (e_109_0 : w_109_0 = matrixMulScalarRight w_102_0 w_108_0)
    (e_110_0 : w_110_0 = matrixSub w_98_0 w_109_0)
    (e_112_0 : w_112_0 = matrixAdd w_110_0 w_111_0)
    (e_113_0 : w_113_0 = matrixMulScalarLeft w_94_0 w_91_0)
    (e_114_0 : w_114_0 = matrixMulScalarLeft w_96_0 w_93_0)
    (e_115_0 : w_115_0 = matrixAdd w_113_0 w_114_0)
    (e_117_0 : w_117_0 = matrixMulScalarLeft w_116_0 w_101_0)
    (e_118_0 : w_118_0 = matrixMulScalarRight w_117_0 w_108_0)
    (e_119_0 : w_119_0 = matrixSub w_115_0 w_118_0)
    (e_121_0 : w_121_0 = matrixAdd w_119_0 w_120_0)
    (h94_0 : centeredRebaseRuns w_4_0 w_94_0)
    (h96_0 : centeredRebaseRuns w_7_0 w_96_0)
    (h99_0 : centeredRebaseRuns w_11_0 w_99_0)
    (h100_0 : gadgetMatrixRuns Backend.backend 32768 10 w_100_0)
    (h101_0 : centeredRebaseRuns w_100_0 w_101_0)
    (h111_0 : gaussianSample ((4/1)) (26) w_111_0)
    (h116_0 : centeredRebaseRuns w_29_0 w_116_0)
    (h120_0 : gaussianSample ((4/1)) (26) w_120_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_90_0, w_91_0], ![w_92_0, w_93_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_104_0 w_106_0 * unitPair sp j) • gg)
      ![w_112_0, w_121_0] 26 := by
  subst e_7_0 e_29_0 e_95_0 e_97_0 e_98_0 e_102_0 e_103_0 e_105_0 e_107_0 e_108_0 e_109_0 e_110_0 e_113_0 e_114_0 e_115_0 e_117_0 e_118_0 e_119_0
  rw [gadgetMatrixRuns_unique h100_0 hg'] at h101_0
  have hgg := centeredRebaseRuns_unique h101_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h111_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h120_0
  have rsm : w_94_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h94_0)
  have rsp : w_99_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h99_0)
  have rone0 : w_116_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h116_0)
  have rone1 : w_96_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h96_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_112_0 = _
    rw [e_112_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_121_0 = _
    rw [e_121_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `155_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_155_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_133_0 : ExactMatrix Q6 8 1 10}
    {w_134_0 : ExactMatrix Q6 8 1 10}
    {w_135_0 : ExactMatrix Q6 8 1 10}
    {w_136_0 : ExactMatrix Q6 8 1 10}
    {w_147_0 : Int}
    {w_149_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_137_0 : ExactMatrix Q6 8 1 1}
    {w_138_0 : ExactMatrix Q6 8 1 10}
    {w_139_0 : ExactMatrix Q6 8 1 1}
    {w_140_0 : ExactMatrix Q6 8 1 10}
    {w_141_0 : ExactMatrix Q6 8 1 10}
    {w_142_0 : ExactMatrix Q6 8 1 1}
    {w_143_0 : ExactMatrix Q5 8 1 10}
    {w_144_0 : ExactMatrix Q6 8 1 10}
    {w_145_0 : ExactMatrix Q6 8 1 10}
    {w_146_0 : ExactMatrix Q6 8 1 1}
    {w_148_0 : ExactMatrix Q6 8 1 1}
    {w_150_0 : ExactMatrix Q6 8 1 1}
    {w_151_0 : ExactMatrix Q6 8 1 1}
    {w_152_0 : ExactMatrix Q6 8 1 10}
    {w_153_0 : ExactMatrix Q6 8 1 10}
    {w_154_0 : ExactMatrix Q6 8 1 10}
    {w_155_0 : ExactMatrix Q6 8 1 10}
    {w_156_0 : ExactMatrix Q6 8 1 10}
    {w_157_0 : ExactMatrix Q6 8 1 10}
    {w_158_0 : ExactMatrix Q6 8 1 10}
    {w_159_0 : ExactMatrix Q6 8 1 1}
    {w_160_0 : ExactMatrix Q6 8 1 10}
    {w_161_0 : ExactMatrix Q6 8 1 10}
    {w_162_0 : ExactMatrix Q6 8 1 10}
    {w_163_0 : ExactMatrix Q6 8 1 10}
    {w_164_0 : ExactMatrix Q6 8 1 10}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_138_0 : w_138_0 = matrixMulScalarLeft w_137_0 w_133_0)
    (e_140_0 : w_140_0 = matrixMulScalarLeft w_139_0 w_135_0)
    (e_141_0 : w_141_0 = matrixAdd w_138_0 w_140_0)
    (e_145_0 : w_145_0 = matrixMulScalarLeft w_142_0 w_144_0)
    (e_146_0 : w_146_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_148_0 : w_148_0 = liftInteger w_147_0)
    (e_150_0 : w_150_0 = multiplyMonomial w_148_0 w_149_0)
    (e_151_0 : w_151_0 = matrixMulScalarLeft w_146_0 w_150_0)
    (e_152_0 : w_152_0 = matrixMulScalarRight w_145_0 w_151_0)
    (e_153_0 : w_153_0 = matrixSub w_141_0 w_152_0)
    (e_155_0 : w_155_0 = matrixAdd w_153_0 w_154_0)
    (e_156_0 : w_156_0 = matrixMulScalarLeft w_137_0 w_134_0)
    (e_157_0 : w_157_0 = matrixMulScalarLeft w_139_0 w_136_0)
    (e_158_0 : w_158_0 = matrixAdd w_156_0 w_157_0)
    (e_160_0 : w_160_0 = matrixMulScalarLeft w_159_0 w_144_0)
    (e_161_0 : w_161_0 = matrixMulScalarRight w_160_0 w_151_0)
    (e_162_0 : w_162_0 = matrixSub w_158_0 w_161_0)
    (e_164_0 : w_164_0 = matrixAdd w_162_0 w_163_0)
    (h137_0 : centeredRebaseRuns w_4_0 w_137_0)
    (h139_0 : centeredRebaseRuns w_7_0 w_139_0)
    (h142_0 : centeredRebaseRuns w_11_0 w_142_0)
    (h143_0 : gadgetMatrixRuns Backend.backend 32768 10 w_143_0)
    (h144_0 : centeredRebaseRuns w_143_0 w_144_0)
    (h154_0 : gaussianSample ((4/1)) (26) w_154_0)
    (h159_0 : centeredRebaseRuns w_29_0 w_159_0)
    (h163_0 : gaussianSample ((4/1)) (26) w_163_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_133_0, w_134_0], ![w_135_0, w_136_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_147_0 w_149_0 * unitPair sp j) • gg)
      ![w_155_0, w_164_0] 26 := by
  subst e_7_0 e_29_0 e_138_0 e_140_0 e_141_0 e_145_0 e_146_0 e_148_0 e_150_0 e_151_0 e_152_0 e_153_0 e_156_0 e_157_0 e_158_0 e_160_0 e_161_0 e_162_0
  rw [gadgetMatrixRuns_unique h143_0 hg'] at h144_0
  have hgg := centeredRebaseRuns_unique h144_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h154_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h163_0
  have rsm : w_137_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h137_0)
  have rsp : w_142_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h142_0)
  have rone0 : w_139_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h139_0)
  have rone1 : w_159_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h159_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_155_0 = _
    rw [e_155_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_164_0 = _
    rw [e_164_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `198_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_198_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_176_0 : ExactMatrix Q6 8 1 10}
    {w_177_0 : ExactMatrix Q6 8 1 10}
    {w_178_0 : ExactMatrix Q6 8 1 10}
    {w_179_0 : ExactMatrix Q6 8 1 10}
    {w_190_0 : Int}
    {w_192_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_180_0 : ExactMatrix Q6 8 1 1}
    {w_181_0 : ExactMatrix Q6 8 1 10}
    {w_182_0 : ExactMatrix Q6 8 1 1}
    {w_183_0 : ExactMatrix Q6 8 1 10}
    {w_184_0 : ExactMatrix Q6 8 1 10}
    {w_185_0 : ExactMatrix Q6 8 1 1}
    {w_186_0 : ExactMatrix Q5 8 1 10}
    {w_187_0 : ExactMatrix Q6 8 1 10}
    {w_188_0 : ExactMatrix Q6 8 1 10}
    {w_189_0 : ExactMatrix Q6 8 1 1}
    {w_191_0 : ExactMatrix Q6 8 1 1}
    {w_193_0 : ExactMatrix Q6 8 1 1}
    {w_194_0 : ExactMatrix Q6 8 1 1}
    {w_195_0 : ExactMatrix Q6 8 1 10}
    {w_196_0 : ExactMatrix Q6 8 1 10}
    {w_197_0 : ExactMatrix Q6 8 1 10}
    {w_198_0 : ExactMatrix Q6 8 1 10}
    {w_199_0 : ExactMatrix Q6 8 1 10}
    {w_200_0 : ExactMatrix Q6 8 1 10}
    {w_201_0 : ExactMatrix Q6 8 1 10}
    {w_202_0 : ExactMatrix Q6 8 1 1}
    {w_203_0 : ExactMatrix Q6 8 1 10}
    {w_204_0 : ExactMatrix Q6 8 1 10}
    {w_205_0 : ExactMatrix Q6 8 1 10}
    {w_206_0 : ExactMatrix Q6 8 1 10}
    {w_207_0 : ExactMatrix Q6 8 1 10}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_181_0 : w_181_0 = matrixMulScalarLeft w_180_0 w_176_0)
    (e_183_0 : w_183_0 = matrixMulScalarLeft w_182_0 w_178_0)
    (e_184_0 : w_184_0 = matrixAdd w_181_0 w_183_0)
    (e_188_0 : w_188_0 = matrixMulScalarLeft w_185_0 w_187_0)
    (e_189_0 : w_189_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_191_0 : w_191_0 = liftInteger w_190_0)
    (e_193_0 : w_193_0 = multiplyMonomial w_191_0 w_192_0)
    (e_194_0 : w_194_0 = matrixMulScalarLeft w_189_0 w_193_0)
    (e_195_0 : w_195_0 = matrixMulScalarRight w_188_0 w_194_0)
    (e_196_0 : w_196_0 = matrixSub w_184_0 w_195_0)
    (e_198_0 : w_198_0 = matrixAdd w_196_0 w_197_0)
    (e_199_0 : w_199_0 = matrixMulScalarLeft w_180_0 w_177_0)
    (e_200_0 : w_200_0 = matrixMulScalarLeft w_182_0 w_179_0)
    (e_201_0 : w_201_0 = matrixAdd w_199_0 w_200_0)
    (e_203_0 : w_203_0 = matrixMulScalarLeft w_202_0 w_187_0)
    (e_204_0 : w_204_0 = matrixMulScalarRight w_203_0 w_194_0)
    (e_205_0 : w_205_0 = matrixSub w_201_0 w_204_0)
    (e_207_0 : w_207_0 = matrixAdd w_205_0 w_206_0)
    (h180_0 : centeredRebaseRuns w_4_0 w_180_0)
    (h182_0 : centeredRebaseRuns w_7_0 w_182_0)
    (h185_0 : centeredRebaseRuns w_11_0 w_185_0)
    (h186_0 : gadgetMatrixRuns Backend.backend 32768 10 w_186_0)
    (h187_0 : centeredRebaseRuns w_186_0 w_187_0)
    (h197_0 : gaussianSample ((4/1)) (26) w_197_0)
    (h202_0 : centeredRebaseRuns w_29_0 w_202_0)
    (h206_0 : gaussianSample ((4/1)) (26) w_206_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_176_0, w_177_0], ![w_178_0, w_179_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_190_0 w_192_0 * unitPair sp j) • gg)
      ![w_198_0, w_207_0] 26 := by
  subst e_7_0 e_29_0 e_181_0 e_183_0 e_184_0 e_188_0 e_189_0 e_191_0 e_193_0 e_194_0 e_195_0 e_196_0 e_199_0 e_200_0 e_201_0 e_203_0 e_204_0 e_205_0
  rw [gadgetMatrixRuns_unique h186_0 hg'] at h187_0
  have hgg := centeredRebaseRuns_unique h187_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h197_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h206_0
  have rsm : w_180_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h180_0)
  have rsp : w_185_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h185_0)
  have rone0 : w_182_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h182_0)
  have rone1 : w_202_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h202_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_198_0 = _
    rw [e_198_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_207_0 = _
    rw [e_207_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `241_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_241_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_219_0 : ExactMatrix Q6 8 1 10}
    {w_220_0 : ExactMatrix Q6 8 1 10}
    {w_221_0 : ExactMatrix Q6 8 1 10}
    {w_222_0 : ExactMatrix Q6 8 1 10}
    {w_233_0 : Int}
    {w_235_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_223_0 : ExactMatrix Q6 8 1 1}
    {w_224_0 : ExactMatrix Q6 8 1 10}
    {w_225_0 : ExactMatrix Q6 8 1 1}
    {w_226_0 : ExactMatrix Q6 8 1 10}
    {w_227_0 : ExactMatrix Q6 8 1 10}
    {w_228_0 : ExactMatrix Q6 8 1 1}
    {w_229_0 : ExactMatrix Q5 8 1 10}
    {w_230_0 : ExactMatrix Q6 8 1 10}
    {w_231_0 : ExactMatrix Q6 8 1 10}
    {w_232_0 : ExactMatrix Q6 8 1 1}
    {w_234_0 : ExactMatrix Q6 8 1 1}
    {w_236_0 : ExactMatrix Q6 8 1 1}
    {w_237_0 : ExactMatrix Q6 8 1 1}
    {w_238_0 : ExactMatrix Q6 8 1 10}
    {w_239_0 : ExactMatrix Q6 8 1 10}
    {w_240_0 : ExactMatrix Q6 8 1 10}
    {w_241_0 : ExactMatrix Q6 8 1 10}
    {w_242_0 : ExactMatrix Q6 8 1 10}
    {w_243_0 : ExactMatrix Q6 8 1 10}
    {w_244_0 : ExactMatrix Q6 8 1 10}
    {w_245_0 : ExactMatrix Q6 8 1 1}
    {w_246_0 : ExactMatrix Q6 8 1 10}
    {w_247_0 : ExactMatrix Q6 8 1 10}
    {w_248_0 : ExactMatrix Q6 8 1 10}
    {w_249_0 : ExactMatrix Q6 8 1 10}
    {w_250_0 : ExactMatrix Q6 8 1 10}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_224_0 : w_224_0 = matrixMulScalarLeft w_223_0 w_219_0)
    (e_226_0 : w_226_0 = matrixMulScalarLeft w_225_0 w_221_0)
    (e_227_0 : w_227_0 = matrixAdd w_224_0 w_226_0)
    (e_231_0 : w_231_0 = matrixMulScalarLeft w_228_0 w_230_0)
    (e_232_0 : w_232_0 = (matrixPolynomial [(p5)] : ExactMatrix Q6 8 1 1))
    (e_234_0 : w_234_0 = liftInteger w_233_0)
    (e_236_0 : w_236_0 = multiplyMonomial w_234_0 w_235_0)
    (e_237_0 : w_237_0 = matrixMulScalarLeft w_232_0 w_236_0)
    (e_238_0 : w_238_0 = matrixMulScalarRight w_231_0 w_237_0)
    (e_239_0 : w_239_0 = matrixSub w_227_0 w_238_0)
    (e_241_0 : w_241_0 = matrixAdd w_239_0 w_240_0)
    (e_242_0 : w_242_0 = matrixMulScalarLeft w_223_0 w_220_0)
    (e_243_0 : w_243_0 = matrixMulScalarLeft w_225_0 w_222_0)
    (e_244_0 : w_244_0 = matrixAdd w_242_0 w_243_0)
    (e_246_0 : w_246_0 = matrixMulScalarLeft w_245_0 w_230_0)
    (e_247_0 : w_247_0 = matrixMulScalarRight w_246_0 w_237_0)
    (e_248_0 : w_248_0 = matrixSub w_244_0 w_247_0)
    (e_250_0 : w_250_0 = matrixAdd w_248_0 w_249_0)
    (h223_0 : centeredRebaseRuns w_4_0 w_223_0)
    (h225_0 : centeredRebaseRuns w_7_0 w_225_0)
    (h228_0 : centeredRebaseRuns w_11_0 w_228_0)
    (h229_0 : gadgetMatrixRuns Backend.backend 32768 10 w_229_0)
    (h230_0 : centeredRebaseRuns w_229_0 w_230_0)
    (h240_0 : gaussianSample ((4/1)) (26) w_240_0)
    (h245_0 : centeredRebaseRuns w_29_0 w_245_0)
    (h249_0 : gaussianSample ((4/1)) (26) w_249_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q5 N 1 10} {gg : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_219_0, w_220_0], ![w_221_0, w_222_0]]
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * message (n := N) w_233_0 w_235_0 * unitPair sp j) • gg)
      ![w_241_0, w_250_0] 26 := by
  subst e_7_0 e_29_0 e_224_0 e_226_0 e_227_0 e_231_0 e_232_0 e_234_0 e_236_0 e_237_0 e_238_0 e_239_0 e_242_0 e_243_0 e_244_0 e_246_0 e_247_0 e_248_0
  rw [gadgetMatrixRuns_unique h229_0 hg'] at h230_0
  have hgg := centeredRebaseRuns_unique h230_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h240_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h249_0
  have rsm : w_223_0 = fun _ _ ↦ reducePoly Q6 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h223_0)
  have rsp : w_228_0 = fun _ _ ↦ reducePoly Q6 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h228_0)
  have rone0 : w_225_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h225_0)
  have rone1 : w_245_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h245_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_241_0 = _
    rw [e_241_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_250_0 = _
    rw [e_250_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `280_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_280_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_272_0 : Int}
    {w_274_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_263_0 : ExactMatrix Q0 8 1 6}
    {w_264_0 : ExactMatrix Q0 8 1 6}
    {w_265_0 : ExactMatrix Q0 8 1 6}
    {w_266_0 : ExactMatrix Q0 8 1 6}
    {w_267_0 : ExactMatrix Q0 8 1 6}
    {w_268_0 : ExactMatrix Q1 8 1 6}
    {w_269_0 : ExactMatrix Q0 8 1 6}
    {w_270_0 : ExactMatrix Q0 8 1 6}
    {w_271_0 : ExactMatrix Q0 8 1 1}
    {w_273_0 : ExactMatrix Q0 8 1 1}
    {w_275_0 : ExactMatrix Q0 8 1 1}
    {w_276_0 : ExactMatrix Q0 8 1 1}
    {w_277_0 : ExactMatrix Q0 8 1 6}
    {w_278_0 : ExactMatrix Q0 8 1 6}
    {w_279_0 : ExactMatrix Q0 8 1 6}
    {w_280_0 : ExactMatrix Q0 8 1 6}
    {w_281_0 : ExactMatrix Q0 8 1 6}
    {w_282_0 : ExactMatrix Q0 8 1 6}
    {w_283_0 : ExactMatrix Q0 8 1 6}
    {w_284_0 : ExactMatrix Q0 8 1 6}
    {w_285_0 : ExactMatrix Q0 8 1 6}
    {w_286_0 : ExactMatrix Q0 8 1 6}
    {w_287_0 : ExactMatrix Q0 8 1 6}
    {w_288_0 : ExactMatrix Q0 8 1 6}
    {w_289_0 : ExactMatrix Q0 8 1 6}
    {w_290_0 : ExactMatrix Q0 8 1 6}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_264_0 : w_264_0 = matrixMulScalarLeft w_4_0 w_263_0)
    (e_266_0 : w_266_0 = matrixMulScalarLeft w_7_0 w_265_0)
    (e_267_0 : w_267_0 = matrixAdd w_264_0 w_266_0)
    (e_270_0 : w_270_0 = matrixMulScalarLeft w_11_0 w_269_0)
    (e_271_0 : w_271_0 = (matrixPolynomial [(q3)] : ExactMatrix Q0 8 1 1))
    (e_273_0 : w_273_0 = liftInteger w_272_0)
    (e_275_0 : w_275_0 = multiplyMonomial w_273_0 w_274_0)
    (e_276_0 : w_276_0 = matrixMulScalarLeft w_271_0 w_275_0)
    (e_277_0 : w_277_0 = matrixMulScalarRight w_270_0 w_276_0)
    (e_278_0 : w_278_0 = matrixSub w_267_0 w_277_0)
    (e_280_0 : w_280_0 = matrixAdd w_278_0 w_279_0)
    (e_282_0 : w_282_0 = matrixMulScalarLeft w_4_0 w_281_0)
    (e_284_0 : w_284_0 = matrixMulScalarLeft w_7_0 w_283_0)
    (e_285_0 : w_285_0 = matrixAdd w_282_0 w_284_0)
    (e_286_0 : w_286_0 = matrixMulScalarLeft w_29_0 w_269_0)
    (e_287_0 : w_287_0 = matrixMulScalarRight w_286_0 w_276_0)
    (e_288_0 : w_288_0 = matrixSub w_285_0 w_287_0)
    (e_290_0 : w_290_0 = matrixAdd w_288_0 w_289_0)
    (h268_0 : gadgetMatrixRuns Backend.backend 32768 6 w_268_0)
    (h269_0 : centeredRebaseRuns w_268_0 w_269_0)
    (h279_0 : gaussianSample ((4/1)) (26) w_279_0)
    (h289_0 : gaussianSample ((4/1)) (26) w_289_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    
    {g' : ExactMatrix Q1 N 1 6} {gg : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_263_0, w_281_0], ![w_265_0, w_283_0]]
      (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * message (n := N) w_272_0 w_274_0 * unitPair sp j) • gg)
      ![w_280_0, w_290_0] 26 := by
  subst e_7_0 e_29_0 e_264_0 e_266_0 e_267_0 e_270_0 e_271_0 e_273_0 e_275_0 e_276_0 e_277_0 e_278_0 e_282_0 e_284_0 e_285_0 e_286_0 e_287_0 e_288_0
  rw [gadgetMatrixRuns_unique h268_0 hg'] at h269_0
  have hgg := centeredRebaseRuns_unique h269_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h279_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h289_0
  have rsm : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm := hsmIn
  have rsp : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp := hspIn
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_280_0 = _
    rw [e_280_0]
    simp only [rsm, rsp, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_290_0 = _
    rw [e_290_0]
    simp only [rsm, rsp, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `315_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_315_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_293_0 : ExactMatrix Q7 8 1 12}
    {w_294_0 : ExactMatrix Q7 8 1 12}
    {w_295_0 : ExactMatrix Q7 8 1 12}
    {w_296_0 : ExactMatrix Q7 8 1 12}
    {w_307_0 : Int}
    {w_309_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_297_0 : ExactMatrix Q7 8 1 1}
    {w_298_0 : ExactMatrix Q7 8 1 12}
    {w_299_0 : ExactMatrix Q7 8 1 1}
    {w_300_0 : ExactMatrix Q7 8 1 12}
    {w_301_0 : ExactMatrix Q7 8 1 12}
    {w_302_0 : ExactMatrix Q7 8 1 1}
    {w_303_0 : ExactMatrix Q6 8 1 12}
    {w_304_0 : ExactMatrix Q7 8 1 12}
    {w_305_0 : ExactMatrix Q7 8 1 12}
    {w_306_0 : ExactMatrix Q7 8 1 1}
    {w_308_0 : ExactMatrix Q7 8 1 1}
    {w_310_0 : ExactMatrix Q7 8 1 1}
    {w_311_0 : ExactMatrix Q7 8 1 1}
    {w_312_0 : ExactMatrix Q7 8 1 12}
    {w_313_0 : ExactMatrix Q7 8 1 12}
    {w_314_0 : ExactMatrix Q7 8 1 12}
    {w_315_0 : ExactMatrix Q7 8 1 12}
    {w_316_0 : ExactMatrix Q7 8 1 12}
    {w_317_0 : ExactMatrix Q7 8 1 12}
    {w_318_0 : ExactMatrix Q7 8 1 12}
    {w_319_0 : ExactMatrix Q7 8 1 1}
    {w_320_0 : ExactMatrix Q7 8 1 12}
    {w_321_0 : ExactMatrix Q7 8 1 12}
    {w_322_0 : ExactMatrix Q7 8 1 12}
    {w_323_0 : ExactMatrix Q7 8 1 12}
    {w_324_0 : ExactMatrix Q7 8 1 12}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_298_0 : w_298_0 = matrixMulScalarLeft w_297_0 w_293_0)
    (e_300_0 : w_300_0 = matrixMulScalarLeft w_299_0 w_295_0)
    (e_301_0 : w_301_0 = matrixAdd w_298_0 w_300_0)
    (e_305_0 : w_305_0 = matrixMulScalarLeft w_302_0 w_304_0)
    (e_306_0 : w_306_0 = (matrixPolynomial [(p6)] : ExactMatrix Q7 8 1 1))
    (e_308_0 : w_308_0 = liftInteger w_307_0)
    (e_310_0 : w_310_0 = multiplyMonomial w_308_0 w_309_0)
    (e_311_0 : w_311_0 = matrixMulScalarLeft w_306_0 w_310_0)
    (e_312_0 : w_312_0 = matrixMulScalarRight w_305_0 w_311_0)
    (e_313_0 : w_313_0 = matrixSub w_301_0 w_312_0)
    (e_315_0 : w_315_0 = matrixAdd w_313_0 w_314_0)
    (e_316_0 : w_316_0 = matrixMulScalarLeft w_297_0 w_294_0)
    (e_317_0 : w_317_0 = matrixMulScalarLeft w_299_0 w_296_0)
    (e_318_0 : w_318_0 = matrixAdd w_316_0 w_317_0)
    (e_320_0 : w_320_0 = matrixMulScalarLeft w_319_0 w_304_0)
    (e_321_0 : w_321_0 = matrixMulScalarRight w_320_0 w_311_0)
    (e_322_0 : w_322_0 = matrixSub w_318_0 w_321_0)
    (e_324_0 : w_324_0 = matrixAdd w_322_0 w_323_0)
    (h297_0 : centeredRebaseRuns w_4_0 w_297_0)
    (h299_0 : centeredRebaseRuns w_7_0 w_299_0)
    (h302_0 : centeredRebaseRuns w_11_0 w_302_0)
    (h303_0 : gadgetMatrixRuns Backend.backend 32768 12 w_303_0)
    (h304_0 : centeredRebaseRuns w_303_0 w_304_0)
    (h314_0 : gaussianSample ((4/1)) (26) w_314_0)
    (h319_0 : centeredRebaseRuns w_29_0 w_319_0)
    (h323_0 : gaussianSample ((4/1)) (26) w_323_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q6 N 1 12} {gg : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_293_0, w_294_0], ![w_295_0, w_296_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * message (n := N) w_307_0 w_309_0 * unitPair sp j) • gg)
      ![w_315_0, w_324_0] 26 := by
  subst e_7_0 e_29_0 e_298_0 e_300_0 e_301_0 e_305_0 e_306_0 e_308_0 e_310_0 e_311_0 e_312_0 e_313_0 e_316_0 e_317_0 e_318_0 e_320_0 e_321_0 e_322_0
  rw [gadgetMatrixRuns_unique h303_0 hg'] at h304_0
  have hgg := centeredRebaseRuns_unique h304_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h314_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h323_0
  have rsm : w_297_0 = fun _ _ ↦ reducePoly Q7 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h297_0)
  have rsp : w_302_0 = fun _ _ ↦ reducePoly Q7 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h302_0)
  have rone0 : w_299_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h299_0)
  have rone1 : w_319_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h319_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_315_0 = _
    rw [e_315_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_324_0 = _
    rw [e_324_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The encryption `359_0`: `p μ (s_p, 1) g` under the mask `(s_m, 1)`. -/
theorem encOne_359_0 {model : HashModel}
    {w_4_0 : ExactMatrix Q0 8 1 1}
    {w_11_0 : ExactMatrix Q0 8 1 1}
    {w_337_0 : ExactMatrix Q7 8 1 12}
    {w_338_0 : ExactMatrix Q7 8 1 12}
    {w_339_0 : ExactMatrix Q7 8 1 12}
    {w_340_0 : ExactMatrix Q7 8 1 12}
    {w_351_0 : Int}
    {w_353_0 : Int}
    {w_7_0 : ExactMatrix Q0 8 1 1}
    {w_29_0 : ExactMatrix Q0 8 1 1}
    {w_341_0 : ExactMatrix Q7 8 1 1}
    {w_342_0 : ExactMatrix Q7 8 1 12}
    {w_343_0 : ExactMatrix Q7 8 1 1}
    {w_344_0 : ExactMatrix Q7 8 1 12}
    {w_345_0 : ExactMatrix Q7 8 1 12}
    {w_346_0 : ExactMatrix Q7 8 1 1}
    {w_347_0 : ExactMatrix Q6 8 1 12}
    {w_348_0 : ExactMatrix Q7 8 1 12}
    {w_349_0 : ExactMatrix Q7 8 1 12}
    {w_350_0 : ExactMatrix Q7 8 1 1}
    {w_352_0 : ExactMatrix Q7 8 1 1}
    {w_354_0 : ExactMatrix Q7 8 1 1}
    {w_355_0 : ExactMatrix Q7 8 1 1}
    {w_356_0 : ExactMatrix Q7 8 1 12}
    {w_357_0 : ExactMatrix Q7 8 1 12}
    {w_358_0 : ExactMatrix Q7 8 1 12}
    {w_359_0 : ExactMatrix Q7 8 1 12}
    {w_360_0 : ExactMatrix Q7 8 1 12}
    {w_361_0 : ExactMatrix Q7 8 1 12}
    {w_362_0 : ExactMatrix Q7 8 1 12}
    {w_363_0 : ExactMatrix Q7 8 1 1}
    {w_364_0 : ExactMatrix Q7 8 1 12}
    {w_365_0 : ExactMatrix Q7 8 1 12}
    {w_366_0 : ExactMatrix Q7 8 1 12}
    {w_367_0 : ExactMatrix Q7 8 1 12}
    {w_368_0 : ExactMatrix Q7 8 1 12}
    (e_7_0 : w_7_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_29_0 : w_29_0 = (1 : ExactMatrix Q0 8 1 1))
    (e_342_0 : w_342_0 = matrixMulScalarLeft w_341_0 w_337_0)
    (e_344_0 : w_344_0 = matrixMulScalarLeft w_343_0 w_339_0)
    (e_345_0 : w_345_0 = matrixAdd w_342_0 w_344_0)
    (e_349_0 : w_349_0 = matrixMulScalarLeft w_346_0 w_348_0)
    (e_350_0 : w_350_0 = (matrixPolynomial [(p6)] : ExactMatrix Q7 8 1 1))
    (e_352_0 : w_352_0 = liftInteger w_351_0)
    (e_354_0 : w_354_0 = multiplyMonomial w_352_0 w_353_0)
    (e_355_0 : w_355_0 = matrixMulScalarLeft w_350_0 w_354_0)
    (e_356_0 : w_356_0 = matrixMulScalarRight w_349_0 w_355_0)
    (e_357_0 : w_357_0 = matrixSub w_345_0 w_356_0)
    (e_359_0 : w_359_0 = matrixAdd w_357_0 w_358_0)
    (e_360_0 : w_360_0 = matrixMulScalarLeft w_341_0 w_338_0)
    (e_361_0 : w_361_0 = matrixMulScalarLeft w_343_0 w_340_0)
    (e_362_0 : w_362_0 = matrixAdd w_360_0 w_361_0)
    (e_364_0 : w_364_0 = matrixMulScalarLeft w_363_0 w_348_0)
    (e_365_0 : w_365_0 = matrixMulScalarRight w_364_0 w_355_0)
    (e_366_0 : w_366_0 = matrixSub w_362_0 w_365_0)
    (e_368_0 : w_368_0 = matrixAdd w_366_0 w_367_0)
    (h341_0 : centeredRebaseRuns w_4_0 w_341_0)
    (h343_0 : centeredRebaseRuns w_7_0 w_343_0)
    (h346_0 : centeredRebaseRuns w_11_0 w_346_0)
    (h347_0 : gadgetMatrixRuns Backend.backend 32768 12 w_347_0)
    (h348_0 : centeredRebaseRuns w_347_0 w_348_0)
    (h358_0 : gaussianSample ((4/1)) (26) w_358_0)
    (h363_0 : centeredRebaseRuns w_29_0 w_363_0)
    (h367_0 : gaussianSample ((4/1)) (26) w_367_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hsmIn : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm) (hspIn : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp)
    (hone : w_29_0 = 1)
    {g' : ExactMatrix Q6 N 1 12} {gg : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' gg) :
    Rel sm ![![w_337_0, w_338_0], ![w_339_0, w_340_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * message (n := N) w_351_0 w_353_0 * unitPair sp j) • gg)
      ![w_359_0, w_368_0] 26 := by
  subst e_7_0 e_29_0 e_342_0 e_344_0 e_345_0 e_349_0 e_350_0 e_352_0 e_354_0 e_355_0 e_356_0 e_357_0 e_360_0 e_361_0 e_362_0 e_364_0 e_365_0 e_366_0
  rw [gadgetMatrixRuns_unique h347_0 hg'] at h348_0
  have hgg := centeredRebaseRuns_unique h348_0 hg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h358_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h367_0
  have rsm : w_341_0 = fun _ _ ↦ reducePoly Q7 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hsmIn]; exact h341_0)
  have rsp : w_346_0 = fun _ _ ↦ reducePoly Q7 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hspIn]; exact h346_0)
  have rone0 : w_343_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h343_0)
  have rone1 : w_363_0 = 1 := rebase_one (q := Q0) (by norm_num) (by rw [← hone]; exact h363_0)
  subst hgg
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_359_0 = _
    rw [e_359_0]
    simp only [rsm, rsp, rone0, rone1, hE0, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_368_0 = _
    rw [e_368_0]
    simp only [rsm, rsp, rone0, rone1, hE1, multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

end RefreshTopProof.Part_0_0
