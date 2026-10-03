import LwrSetup

/-!
# The preprocessing stage

The selector of cell `i` of block `j` is preprocessed into the RHS ciphertext
`s top_j + bot_j = (P ν r_j) g + e_j` at the modulus of the block's private multiplication, and
the readout LUT's switching ciphertexts at `Q1` satisfy
`σ_i(s) B_{0j} + B_{1j} = (C_i ⌊Q1/2⌋ r_j) · 1 + e_j`.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `0` at `Q6`: `s top_j + bot_j = (P6 ν r_j) g + e_j`. -/
theorem preprocess_selector0 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g) (i : Fin 14) :
    ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ jj, CoeffBound (e jj) 26) ∧ ∀ jj,
      reducePoly Q6 N sp • rhsTop model key Q6 10 rhsPrefix0 i jj + ![outputs.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.1 i] jj =
        reducePoly Q6 N (((P6 : Nat) : Int) * message (n := N) (b0 i) (x0 i) * unitPair sp jj) • g +
          reduceMatrix Q6 N 1 10 (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have hpar := hb.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_3_0, w_5_0, w_7_0, w_8_0, w_9_0, w_11_0, w_14_0, w_15_0, w_16_0, w_17_0, w_18_0, w_19_0, w_22_0, w_25_0, w_28_0, w_29_0, w_30_0, w_32_0, w_33_0, w_34_0, w_42_0, w_46_0, w_48_0, w_49_0, w_52_0, w_53_0, w_54_0, w_55_0, w_56_0, w_59_0, w_65_0, w_67_0, w_68_0, w_71_0, w_72_0, w_73_0, w_74_0, w_75_0, w_76_0, -, -, h1_0, -, -, h2_0, h3_0, h5_0, -, -, h7_0, -, -, h8_0, h9_0, h11_0, -, -, -, -, -, -, h14_0, -, -, -, -, -, -, h15_0, h16_0, h17_0, h18_0, h19_0, h22_0, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h30_0, h32_0, h33_0, h34_0, h42_0, h46_0, h48_0, h49_0, -, -, -, -, -, -, h52_0, h53_0, h54_0, h55_0, h56_0, h59_0, h65_0, h67_0, h68_0, -, -, -, -, -, -, h71_0, h72_0, -, -, -, -, -, -, h73_0, h74_0, -, -, -, -, -, -, h75_0, h76_0, hout⟩ := hpar i
  have hpay := modulusReduce_poly hQ7 hQ6 hN h32_0
  have hone := modulusReduce_one hQ7 hQ6 hN h59_0
  rw [multiplyMonomial_liftInteger] at h42_0
  have hnu := modulusReduce_poly hQ7 hQ6 hN h42_0
  have hgg' := gadgetMatrixRuns_unique h33_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h34_0 hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h46_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h65_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q6 10 rhsPrefix0 i jj = ![w_17_0, w_54_0] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq h17_0).symm
    · exact (hashSample_eq h54_0).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `1` at `Q5`: `s top_j + bot_j = (P5 ν r_j) g + e_j`. -/
theorem preprocess_selector1 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g) (i : Fin 14) :
    ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ jj, CoeffBound (e jj) 26) ∧ ∀ jj,
      reducePoly Q5 N sp • rhsTop model key Q5 8 rhsPrefix1 i jj + ![outputs.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.1 i] jj =
        reducePoly Q5 N (((P5 : Nat) : Int) * message (n := N) (b1 i) (x1 i) * unitPair sp jj) • g +
          reduceMatrix Q5 N 1 8 (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have hpar := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_11_0, w_12_0, w_14_0, w_16_0, w_18_0, w_19_0, w_21_0, w_22_0, w_24_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, w_32_0, w_35_0, w_38_0, w_41_0, w_42_0, w_43_0, w_45_0, w_46_0, w_47_0, w_55_0, w_59_0, w_61_0, w_62_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_72_0, w_78_0, w_80_0, w_81_0, w_84_0, w_85_0, w_86_0, w_87_0, w_88_0, w_89_0, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, h9_0, h11_0, h12_0, h14_0, -, -, h16_0, h18_0, -, -, h19_0, h21_0, h22_0, h24_0, -, -, -, -, -, -, h27_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h32_0, h35_0, h38_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h43_0, h45_0, h46_0, h47_0, h55_0, h59_0, h61_0, h62_0, -, -, -, -, -, -, h65_0, h66_0, h67_0, h68_0, h69_0, h72_0, h78_0, h80_0, h81_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, -, -, -, -, -, -, h88_0, h89_0, hout⟩ := hpar i
  have hpay := modulusReduce_poly hQ7 hQ5 hN h45_0
  have hone := modulusReduce_one hQ7 hQ5 hN h72_0
  rw [multiplyMonomial_liftInteger] at h55_0
  have hnu := modulusReduce_poly hQ7 hQ5 hN h55_0
  have hgg' := gadgetMatrixRuns_unique h46_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h47_0 hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h59_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h78_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q5 8 rhsPrefix1 i jj = ![w_30_0, w_67_0] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq h30_0).symm
    · exact (hashSample_eq h67_0).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `2` at `Q4`: `s top_j + bot_j = (P4 ν r_j) g + e_j`. -/
theorem preprocess_selector2 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g) (i : Fin 14) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ jj, CoeffBound (e jj) 26) ∧ ∀ jj,
      reducePoly Q4 N sp • rhsTop model key Q4 6 rhsPrefix2 i jj + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] jj =
        reducePoly Q4 N (((P4 : Nat) : Int) * message (n := N) (b2 i) (x2 i) * unitPair sp jj) • g +
          reduceMatrix Q4 N 1 6 (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have hpar := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_11_0, w_12_0, w_14_0, w_16_0, w_18_0, w_19_0, w_21_0, w_22_0, w_24_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, w_32_0, w_35_0, w_38_0, w_41_0, w_42_0, w_43_0, w_45_0, w_46_0, w_47_0, w_55_0, w_59_0, w_61_0, w_62_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_72_0, w_78_0, w_80_0, w_81_0, w_84_0, w_85_0, w_86_0, w_87_0, w_88_0, w_89_0, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, h9_0, h11_0, h12_0, h14_0, -, -, h16_0, h18_0, -, -, h19_0, h21_0, h22_0, h24_0, -, -, -, -, -, -, h27_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h32_0, h35_0, h38_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h43_0, h45_0, h46_0, h47_0, h55_0, h59_0, h61_0, h62_0, -, -, -, -, -, -, h65_0, h66_0, h67_0, h68_0, h69_0, h72_0, h78_0, h80_0, h81_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, -, -, -, -, -, -, h88_0, h89_0, hout⟩ := hpar i
  have hpay := modulusReduce_poly hQ7 hQ4 hN h45_0
  have hone := modulusReduce_one hQ7 hQ4 hN h72_0
  rw [multiplyMonomial_liftInteger] at h55_0
  have hnu := modulusReduce_poly hQ7 hQ4 hN h55_0
  have hgg' := gadgetMatrixRuns_unique h46_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h47_0 hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h59_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h78_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q4 6 rhsPrefix2 i jj = ![w_30_0, w_67_0] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq h30_0).symm
    · exact (hashSample_eq h67_0).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `3` at `Q3`: `s top_j + bot_j = (P3 ν r_j) g + e_j`. -/
theorem preprocess_selector3 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q2 N 1 4} {g : ExactMatrix Q3 N 1 4}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 4 g') (hg : centeredRebaseRuns g' g) (i : Fin 14) :
    ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ jj, CoeffBound (e jj) 26) ∧ ∀ jj,
      reducePoly Q3 N sp • rhsTop model key Q3 4 rhsPrefix3 i jj + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] jj =
        reducePoly Q3 N (((P3 : Nat) : Int) * message (n := N) (b3 i) (x3 i) * unitPair sp jj) • g +
          reduceMatrix Q3 N 1 4 (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have hpar := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_11_0, w_12_0, w_14_0, w_16_0, w_18_0, w_19_0, w_21_0, w_22_0, w_24_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, w_32_0, w_35_0, w_38_0, w_41_0, w_42_0, w_43_0, w_45_0, w_46_0, w_47_0, w_55_0, w_59_0, w_61_0, w_62_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_72_0, w_78_0, w_80_0, w_81_0, w_84_0, w_85_0, w_86_0, w_87_0, w_88_0, w_89_0, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, h9_0, h11_0, h12_0, h14_0, -, -, h16_0, h18_0, -, -, h19_0, h21_0, h22_0, h24_0, -, -, -, -, -, -, h27_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h32_0, h35_0, h38_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h43_0, h45_0, h46_0, h47_0, h55_0, h59_0, h61_0, h62_0, -, -, -, -, -, -, h65_0, h66_0, h67_0, h68_0, h69_0, h72_0, h78_0, h80_0, h81_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, -, -, -, -, -, -, h88_0, h89_0, hout⟩ := hpar i
  have hpay := modulusReduce_poly hQ7 hQ3 hN h45_0
  have hone := modulusReduce_one hQ7 hQ3 hN h72_0
  rw [multiplyMonomial_liftInteger] at h55_0
  have hnu := modulusReduce_poly hQ7 hQ3 hN h55_0
  have hgg' := gadgetMatrixRuns_unique h46_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h47_0 hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h59_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h78_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q3 4 rhsPrefix3 i jj = ![w_30_0, w_67_0] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq h30_0).symm
    · exact (hashSample_eq h67_0).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The selector RHS of member `i` of block `4` at `Q2`: `s top_j + bot_j = (P2 ν r_j) g + e_j`. -/
theorem preprocess_selector4 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q1 N 1 2} {g : ExactMatrix Q2 N 1 2}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 2 g') (hg : centeredRebaseRuns g' g) (i : Fin 14) :
    ∃ e : Fin 2 → ErrorMatrix N 1 2, (∀ jj, CoeffBound (e jj) 26) ∧ ∀ jj,
      reducePoly Q2 N sp • rhsTop model key Q2 2 rhsPrefix4 i jj + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] jj =
        reducePoly Q2 N (((P2 : Nat) : Int) * message (n := N) (b4 i) (x4 i) * unitPair sp jj) • g +
          reduceMatrix Q2 N 1 2 (e jj) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have hpar := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_3_0, w_4_0, w_6_0, w_8_0, w_9_0, w_11_0, w_12_0, w_14_0, w_16_0, w_18_0, w_19_0, w_21_0, w_22_0, w_24_0, w_27_0, w_28_0, w_29_0, w_30_0, w_31_0, w_32_0, w_35_0, w_38_0, w_41_0, w_42_0, w_43_0, w_45_0, w_46_0, w_47_0, w_55_0, w_59_0, w_61_0, w_62_0, w_65_0, w_66_0, w_67_0, w_68_0, w_69_0, w_72_0, w_78_0, w_80_0, w_81_0, w_84_0, w_85_0, w_86_0, w_87_0, w_88_0, w_89_0, -, -, h1_0, h3_0, h4_0, h6_0, h8_0, -, -, h9_0, h11_0, h12_0, h14_0, -, -, h16_0, h18_0, -, -, h19_0, h21_0, h22_0, h24_0, -, -, -, -, -, -, h27_0, -, -, -, -, -, -, h28_0, h29_0, h30_0, h31_0, h32_0, h35_0, h38_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h43_0, h45_0, h46_0, h47_0, h55_0, h59_0, h61_0, h62_0, -, -, -, -, -, -, h65_0, h66_0, h67_0, h68_0, h69_0, h72_0, h78_0, h80_0, h81_0, -, -, -, -, -, -, h84_0, h85_0, -, -, -, -, -, -, h86_0, h87_0, -, -, -, -, -, -, h88_0, h89_0, hout⟩ := hpar i
  have hpay := modulusReduce_poly hQ7 hQ2 hN h45_0
  have hone := modulusReduce_one hQ7 hQ2 hN h72_0
  rw [multiplyMonomial_liftInteger] at h55_0
  have hnu := modulusReduce_poly hQ7 hQ2 hN h55_0
  have hgg' := gadgetMatrixRuns_unique h46_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h47_0 hg
  subst hgg
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h59_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h78_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, ho6, ho7, -⟩ := hout
  refine ⟨![e0', e1'], fun jj ↦ by fin_cases jj <;> assumption, fun jj ↦ ?_⟩
  have htop : ∀ jj, rhsTop model key Q2 2 rhsPrefix4 i jj = ![w_30_0, w_67_0] jj := by
    intro jj
    fin_cases jj
    · exact (hashSample_eq h30_0).symm
    · exact (hashSample_eq h67_0).symm
  simp only [htop]
  fin_cases jj <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [ho6, ho7, matrixAdd, matrixSub, matrixMulScalarRight_smul,
      Matrix.mul_apply, unitPair, hb0, hb1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The readout LUT's switching ciphertexts at `Q1`:
`σ_i(s) B_{0j} + B_{1j} = (C_i ⌊Q1/2⌋ r_j) · 1 + e_j`. -/
theorem preprocess_switch {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {b0 x0 b1 x1 b2 x2 b3 x3 b4 x4 : Fin 14 → Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model GeneratedClaim.stage_2_params
      (key, payload, b0, x0, b1, x1, b2, x2, b3, x3, b4, x4, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 1, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.1 i, outputs.2.1 i] j + ![outputs.2.2.1 i, outputs.2.2.2.1 i] j =
        (lutCoefficient Q1 inverse1 (sigma i).toNat *
          reducePoly Q1 N ((((Q1 / 2 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) •
          (1 : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h3_3 := hb.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_110_0, w_116_0, w_120_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, -, -, -, -, h35_0, -, -, -, -, h40_0, -, -, -, -, h46_0, -, -, -, -, h51_0, -, -, -, -, h59_0, -, -, -, -, h64_0, -, -, -, -, h70_0, -, -, -, -, h75_0, -, -, -, -, h82_0, -, -, -, -, h87_0, -, -, -, -, h93_0, -, -, -, -, h98_0, h110_0, h116_0, h120_0, hout⟩ := h3_3 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ1 hN h5_0
  have hw4 := modulusReduce_poly hQ7 hQ1 hN h4_0
  obtain ⟨e0', hb0, he0⟩ := gaussian_bound h110_0
  obtain ⟨e1', hb1, he1⟩ := gaussian_bound h120_0
  have hw116 := modulusReduce_one hQ7 hQ1 hN h116_0
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ1 hN h12_0
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ1 hN h17_0
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ1 hN h23_0
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ1 hN h28_0
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ1 hN h35_0
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ1 hN h40_0
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ1 hN h46_0
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ1 hN h51_0
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ1 hN h59_0
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ1 hN h64_0
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ1 hN h70_0
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ1 hN h75_0
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ1 hN h82_0
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ1 hN h87_0
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ1 hN h93_0
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ1 hN h98_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, hb0, hb1, hw4, hw5,
      hw116, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ, readoutPoly,
      MxxExponentLutLwr.readout, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14,
      a15, rotationPolynomial, substitution_root_pow, map_ofNat, reducePoly_substitution, sigma,
      Matrix.one_apply_eq] <;>
    ring

end LwrProof
