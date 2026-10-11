import TopSwitch
import W_Stage_preprocess_0_0
import ExponentScale
import ExponentAutomorphism

/-! The LUT switching ciphertexts the preprocessing computes (written by
`generator/gen_pre_switch.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `13`. -/
theorem preSwitch13 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_13 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_13 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_13.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `41`. -/
theorem preSwitch41 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_41 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q7 N sp) (hone : x.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k)) (sigma i).toNat *
          reducePoly Q0 N (((q3 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q0 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_41 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_44_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_41_0 w_42_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_43_0 at h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  extract_lets -merge w_37_0 w_38_0 at h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_45_0, h⟩ := h
  extract_lets -merge w_46_0 w_47_0 w_48_0 at h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_41.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, h35_0, h36_0, h39_0, h45_0, h49_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h35_0 hg'] at h36_0
  have hgg := centeredRebaseRuns_unique h36_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h39_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h49_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_45_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h45_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N _ • w_1_0 + w_40_0 = _
    simp only [w_40_0, w_38_0, w_7_0, w_37_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q0 N _ • w_2_0 + w_50_0 = _
    simp only [w_50_0, w_48_0, w_42_0, w_47_0, w_46_0, w_43_0, w_8_0, w_41_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `43`. -/
theorem preSwitch43 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_43 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q7 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_43 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_43.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((536870721 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `182`. -/
theorem preSwitch182 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 2) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_182 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q0 N sp) (hone : x.2.2.1 = 1)
    {g' : ExactMatrix M01 N 1 4} {g : ExactMatrix Q1 N 1 4}
    (hg' : gadgetMatrixRuns Backend.backend 32768 4 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 4, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma2 i).toNat (sigma2_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient2 Q1 618968813111010066723674601 (fun k ↦ reducePoly Q1 N (![Xe (n := N) 1, Xe (n := N) 0] k)) (sigma2 i).toNat *
          reducePoly Q1 N (((q2 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 4 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_182 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_32_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_29_0 w_30_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 w_20_0 w_21_0 w_22_0 w_31_0 at h
  obtain ⟨w_23_0, h⟩ := h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  obtain ⟨w_33_0, h⟩ := h
  extract_lets -merge w_34_0 w_35_0 w_36_0 at h
  obtain ⟨w_37_0, h⟩ := h
  extract_lets -merge w_38_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_182.constraints_0] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, h23_0, h24_0, h27_0, h33_0, h37_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h23_0 hg'] at h24_0
  have hgg := centeredRebaseRuns_unique h24_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h27_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h37_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_33_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h33_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma2 i).toNat (sigma2_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  have hC : w_20_0 0 0 = lutCoefficient2 Q1 618968813111010066723674601 (fun k ↦ reducePoly Q1 N (![Xe (n := N) 1, Xe (n := N) 0] k)) (sigma2 i).toNat := by
    rw [lutCoefficient2, ← switch_coeff2 (sigma2 i).toNat (sigma2_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, hy12_0, hy17_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma2, sigmas2, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_28_0 = _
    simp only [w_28_0, w_26_0, w_7_0, w_25_0, w_22_0, w_21_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_38_0 = _
    simp only [w_38_0, w_36_0, w_30_0, w_35_0, w_34_0, w_31_0, w_8_0, w_29_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `216`. -/
theorem preSwitch216 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_216 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q7 N sp) (hone : x.2.2.1 = 1)
    {g' : ExactMatrix Q1 N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k)) (sigma i).toNat *
          reducePoly Q0 N (((q3 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q0 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_216 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_44_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_41_0 w_42_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_43_0 at h
  obtain ⟨w_35_0, h⟩ := h
  obtain ⟨w_36_0, h⟩ := h
  extract_lets -merge w_37_0 w_38_0 at h
  obtain ⟨w_39_0, h⟩ := h
  extract_lets -merge w_40_0 at h
  obtain ⟨w_45_0, h⟩ := h
  extract_lets -merge w_46_0 w_47_0 w_48_0 at h
  obtain ⟨w_49_0, h⟩ := h
  extract_lets -merge w_50_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_216.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_216.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, h35_0, h36_0, h39_0, h45_0, h49_0, hout⟩ := h
  rw [gadgetMatrixRuns_unique h35_0 hg'] at h36_0
  have hgg := centeredRebaseRuns_unique h36_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h39_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h49_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q0 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_45_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h45_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q0 996917851236477127152305493547100497 (fun k ↦ reducePoly Q0 N (![Xe (n := N) 2, Xe (n := N) 2, Xe (n := N) 6, Xe (n := N) 6] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q0 N _ • w_1_0 + w_40_0 = _
    simp only [w_40_0, w_38_0, w_7_0, w_37_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q0 N _ • w_2_0 + w_50_0 = _
    simp only [w_50_0, w_48_0, w_42_0, w_47_0, w_46_0, w_43_0, w_8_0, w_41_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `218`. -/
theorem preSwitch218 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_218 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q7 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_218 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_218.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_218.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((0 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `374`. -/
theorem preSwitch374 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_374 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_374 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_374.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_374.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `418`. -/
theorem preSwitch418 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_418 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_418 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_418.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_418.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `462`. -/
theorem preSwitch462 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_462 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_462 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_462.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_462.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `506`. -/
theorem preSwitch506 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_506 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_506 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_506.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_506.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The switching ciphertexts of branch `i` the preprocessing computes in `550`. -/
theorem preSwitch550 {model : HashModel} {params : W_Stage_preprocess_0_0.Params} (i : Fin 4) {x : _} {outputs : _}
    (h : W_Stage_preprocess_0_0.parallel_generatedRoot_550 Backend.backend model params i x outputs)
    {sp : ErrorPoly N} (hsec : x.2.1 = fun _ _ ↦ reducePoly Q6 N sp) (hone : x.2.2.1 = 1)
    {g : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 0 j +
        ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] 1 j =
        (lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat *
          reducePoly Q1 N (((1 : Int) : ErrorPoly N) * unitPair sp j)) • g +
          reduceMatrix Q1 N 1 6 (e j) := by
  unfold W_Stage_preprocess_0_0.parallel_generatedRoot_550 at h
  extract_lets -merge _params _backend w_0_0 w_3_0 w_43_0 w_8_0 w_9_0 w_10_0 w_11_0 w_14_0 w_15_0 w_16_0 w_20_0 w_21_0 w_22_0 w_25_0 w_26_0 w_27_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_2_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  obtain ⟨w_5_0, h⟩ := h
  extract_lets -merge w_6_0 w_7_0 w_40_0 w_41_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 w_19_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 at h
  obtain ⟨w_28_0, h⟩ := h
  extract_lets -merge w_29_0 w_30_0 w_31_0 w_32_0 w_33_0 w_34_0 w_42_0 at h
  obtain ⟨w_35_0, h⟩ := h
  extract_lets -merge w_36_0 w_37_0 at h
  obtain ⟨w_38_0, h⟩ := h
  extract_lets -merge w_39_0 at h
  obtain ⟨w_44_0, h⟩ := h
  extract_lets -merge w_45_0 w_46_0 w_47_0 at h
  obtain ⟨w_48_0, h⟩ := h
  extract_lets -merge w_49_0 at h
  simp only [W_Stage_preprocess_0_0.parallel_generatedRoot_550.constraints_0, W_Stage_preprocess_0_0.parallel_generatedRoot_550.constraints_1] at h
  obtain ⟨-, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, h12_0, -, -, -, -, h17_0, -, -, -, -, h23_0, -, -, -, -, h28_0, h35_0, h38_0, h44_0, h48_0, hout⟩ := h
  have hgg := gadgetMatrixRuns_unique h35_0 hg
  subst hgg
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound h38_0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound h48_0
  have hsp : w_4_0 = fun _ _ ↦ reducePoly Q1 N sp :=
    modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h4_0 (fun _ _ ↦ sp)
      (by show x.2.1 = _; rw [hsec]; rfl)
  have hone' : w_44_0 = 1 := by
    rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN h44_0 (fun _ _ ↦ 1) (by
      show x.2.2.1 = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  obtain ⟨_, hsw⟩ := ringAutomorphismRuns_eq (by norm_num) hN h5_0
  have hsw' : w_5_0 = fun _ _ ↦ reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) := by
    rw [hsw, hsp]
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  obtain ⟨_, hy12_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h12_0
  obtain ⟨_, hy17_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h17_0
  obtain ⟨_, hy23_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h23_0
  obtain ⟨_, hy28_0⟩ := ringAutomorphismRuns_eq (by norm_num) hN h28_0
  have hC : w_32_0 0 0 = lutCoefficient Q1 928453219666515100085511901 (fun k ↦ reducePoly Q1 N (![((1 : Int) : ErrorPoly N), ((1 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N), ((3 : Int) : ErrorPoly N)] k)) (sigma i).toNat := by
    rw [lutCoefficient, ← switch_coeff4 (sigma i).toNat (sigma_odd i)]
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.cons_val_three,
      Matrix.head_cons, reducePoly_Xe, map_intCast]
    simp only [w_9_0, w_10_0, w_11_0, w_13_0, w_14_0, w_15_0, w_16_0, w_18_0, w_19_0, w_20_0, w_21_0, w_22_0, w_24_0, w_25_0, w_26_0, w_27_0, w_29_0, w_30_0, w_31_0, w_32_0, hy12_0, hy17_0, hy23_0, hy28_0,
      matrixMulScalarLeft, matrixAdd, matrixNeg, rotationPolynomial, Matrix.add_apply, Matrix.neg_apply]
    simp [matrixPolynomial_single, reducePoly_Xe, Fin.sum_univ_four, X, sigma, sigmas, map_ofNat, map_one, map_zero]
    try ring
  subst hout
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show reducePoly Q1 N _ • w_1_0 + w_39_0 = _
    simp only [w_39_0, w_37_0, w_7_0, w_36_0, w_34_0, w_33_0, w_8_0, w_6_0, hsw', hsp, hE0]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring
  · show reducePoly Q1 N _ • w_2_0 + w_49_0 = _
    simp only [w_49_0, w_47_0, w_41_0, w_46_0, w_45_0, w_42_0, w_8_0, w_40_0, hsw', hone', hE1]
    simp only [matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixNeg, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat, hC]
    try ring

end RefreshTopProof.Part_0_0
