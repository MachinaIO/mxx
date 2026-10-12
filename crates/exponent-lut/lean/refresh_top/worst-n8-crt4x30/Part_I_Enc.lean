import TopInst
import TopLemmas
import Stage_encrypt
import TopLut2

/-! The encryptions of the encrypt stage's parallels (written by `generator/gen_enc.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `42`. -/
theorem encLabel42 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_42 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_42 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_42.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `85`. -/
theorem encLabel85 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_85 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_85 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_85.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `128`. -/
theorem encLabel128 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_128 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_128 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_128.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `171`. -/
theorem encLabel171 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_171 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_171 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_171.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `214`. -/
theorem encLabel214 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_214 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_214 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_214.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `257`. -/
theorem encLabel257 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_257 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q6 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q6 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_257 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_257.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `331`. -/
theorem encLabel331 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_331 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q7 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_331 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_331.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- Label member `o` of the encryption parallel `375`. -/
theorem encLabel375 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} {o : Nat}
    {A00 A01 A10 A11 : ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1} {c e : Int} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_375 Backend.backend model params i0 o
      (A00, A01, A10, A11, smM, oneM, c, e, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G) :
    (outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1) = (A00, A01, A10, A11) ∧
      Rel sm ![![A00, A01], ![A10, A11]]
        (fun j ↦ reducePoly Q7 N (message (n := N) c e * unitPair sm j) • G)
        ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1] 26 ∧
      outputs.2.2.2.2.2.2.1 = fun _ _ ↦ reducePoly Q7 N (message (n := N) c e) := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_375 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_2_0 w_3_0 w_4_0 w_6_0 w_9_0 w_11_0 w_5_0 w_7_0 w_8_0 w_10_0 w_12_0 w_19_0 w_20_0 w_21_0 at h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 w_16_0 w_22_0 w_23_0 at h
  obtain ⟨w_17_0, h⟩ := h
  extract_lets -merge w_18_0 at h
  obtain ⟨w_24_0, h⟩ := h
  extract_lets -merge w_25_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_375.constraints_0] at h
  obtain ⟨h13_0, h17_0, h24_0, hout⟩ := h
  have hg13 := gadgetMatrixRuns_unique h13_0 hG
  subst hg13
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h17_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h24_0
  subst hout
  refine ⟨rfl, ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩, ?_⟩
  · fin_cases j
    · show w_18_0 = _
      simp only [w_18_0, w_16_0, w_15_0, w_14_0, w_8_0, w_5_0, w_7_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_0_0, w_2_0, w_9_0, w_11_0, hsm, hone, he0, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
    · show w_25_0 = _
      simp only [w_25_0, w_23_0, w_22_0, w_21_0, w_19_0, w_20_0, w_14_0, w_12_0, w_10_0, w_4_0, w_6_0,
        w_1_0, w_3_0, w_9_0, w_11_0, hsm, hone, he1, multiplyMonomial_liftInteger]
      simp only [matrixMulScalarLeft_eq]
      funext r c'
      obtain rfl : r = 0 := Subsingleton.elim _ _
      simp [matrixAdd, matrixSub, unitPair, reduceMatrix]
      try ring
    try exact rfl
  · show w_12_0 = _
    simp only [w_12_0, w_10_0, w_9_0, w_11_0, multiplyMonomial_liftInteger]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `44`. -/
theorem encHelper44 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_44 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_44 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_44.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `87`. -/
theorem encHelper87 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_87 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_87 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_87.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `130`. -/
theorem encHelper130 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_130 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_130 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_130.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `173`. -/
theorem encHelper173 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_173 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_173 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_173.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `216`. -/
theorem encHelper216 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_216 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_216 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_216.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `259`. -/
theorem encHelper259 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_259 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 0 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_259 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_259.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q6 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `291`. -/
theorem encHelper291 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 2)
    {smM oneM : ExactMatrix Q0 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_291 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q0 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 0 (sigma2 o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma2 o).toNat (sigma2_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_291 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_291.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q0 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma2 o).toNat (sigma2_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 0 (sigma2 o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `333`. -/
theorem encHelper333 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_333 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q7 12 0 (sigma o))
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_333 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_333.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q7 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q7 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q7 12 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `377`. -/
theorem encHelper377 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_377 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q7 12 0 (sigma o))
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_377 at h
  extract_lets -merge _params _backend w_0_0 w_1_0 w_5_0 w_2_0 w_9_0 at h
  obtain ⟨w_3_0, h⟩ := h
  extract_lets -merge w_4_0 at h
  obtain ⟨w_6_0, h⟩ := h
  extract_lets -merge w_7_0 w_8_0 at h
  obtain ⟨w_10_0, h⟩ := h
  extract_lets -merge w_11_0 at h
  obtain ⟨w_12_0, h⟩ := h
  obtain ⟨w_13_0, h⟩ := h
  extract_lets -merge w_14_0 w_15_0 at h
  obtain ⟨w_16_0, h⟩ := h
  extract_lets -merge w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 w_22_0 at h
  obtain ⟨w_23_0, h⟩ := h
  extract_lets -merge w_24_0 w_25_0 w_26_0 at h
  obtain ⟨w_27_0, h⟩ := h
  extract_lets -merge w_28_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_377.constraints_0] at h
  obtain ⟨-, -, h3_0, h6_0, -, -, h10_0, h12_0, h13_0, h16_0, h18_0, h20_0, -, -, h23_0, h27_0, hout⟩ := h
  have hsm' : w_0_0 = fun _ _ ↦ reducePoly Q7 N sm := hsm
  have hone' : w_5_0 = 1 := hone
  rw [gadgetMatrixRuns_unique h12_0 hg'] at h13_0
  have hgg := centeredRebaseRuns_unique h13_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h16_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h27_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h10_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h23_0
  have hsub' : w_10_0 = fun _ _ ↦ reducePoly Q7 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_23_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q7 12 0 (sigma o) =
      ![![w_3_0, w_18_0], ![w_6_0, w_20_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h3_0).symm
    · exact (hashSample_eq h18_0).symm
    · exact (hashSample_eq h6_0).symm
    · exact (hashSample_eq h20_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_17_0 = _
    simp only [w_17_0, w_15_0, w_14_0, w_11_0, w_8_0, w_4_0, w_7_0, w_9_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_28_0 = _
    simp only [w_28_0, w_26_0, w_25_0, w_24_0, w_22_0, w_19_0, w_21_0, w_9_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `45`. -/
theorem encHelper45 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_45 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_45 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_45.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `46`. -/
theorem encHelper46 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_46 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_46 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_46.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `88`. -/
theorem encHelper88 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_88 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_88 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_88.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `89`. -/
theorem encHelper89 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_89 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_89 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_89.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `131`. -/
theorem encHelper131 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_131 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_131 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_131.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `132`. -/
theorem encHelper132 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_132 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_132 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_132.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `174`. -/
theorem encHelper174 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_174 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_174 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_174.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `175`. -/
theorem encHelper175 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_175 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_175 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_175.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `217`. -/
theorem encHelper217 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_217 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_217 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_217.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `218`. -/
theorem encHelper218 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_218 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_218 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_218.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `260`. -/
theorem encHelper260 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_260 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 1 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_260 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_260.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `261`. -/
theorem encHelper261 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q6 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_261 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 2 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_261 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_261.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `292`. -/
theorem encHelper292 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 2)
    {smM oneM : ExactMatrix Q0 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_292 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q0 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (Mx i0) N 1 4} {g : ExactMatrix (R i0) N 1 4}
    (hg' : gadgetMatrixRuns Backend.backend 32768 4 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key (R i0) 4 1 (sigma2 o))
      (fun j ↦ reducePoly (R i0) N ((((pp i0) : Nat) : Int) *
        unitPair (substitution (sigma2 o).toNat (sigma2_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_292 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_292.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly (R i0) N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly (R i0) N (substitution (sigma2 o).toNat (sigma2_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key (R i0) 4 1 (sigma2 o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `334`. -/
theorem encHelper334 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_334 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 1 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_334 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_334.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `335`. -/
theorem encHelper335 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_335 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 2 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_335 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_335.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `336`. -/
theorem encHelper336 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_336 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 3 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_336 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_336.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 3 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `378`. -/
theorem encHelper378 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_378 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 32768 10 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q6 10 1 (sigma o))
      (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_378 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_378.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q6 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q6 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q6 10 1 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `379`. -/
theorem encHelper379 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_379 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix Q0 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 32768 8 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q5 8 2 (sigma o))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_379 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_379.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q5 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q5 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q5 8 2 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- LUT helper `o` of the encryption parallel `380`. -/
theorem encHelper380 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params} (o : Fin 4)
    {smM oneM : ExactMatrix Q7 N 1 1} {key : ByteArray} {outputs}
    (h : Stage_encrypt.parallel_parallel_generatedRoot_454_380 Backend.backend model params i0 o (smM, key, oneM, ()) outputs)
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {g' : ExactMatrix (R i0) N 1 6} {g : ExactMatrix Q0 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') (hg : centeredRebaseRuns g' g) :
    Rel sm (lutA model key Q0 6 3 (sigma o))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma o).toNat (sigma_odd o) sm) j) • g)
      ![outputs.1, outputs.2.1] 26 := by
  unfold Stage_encrypt.parallel_parallel_generatedRoot_454_380 at h
  extract_lets -merge _params _backend w_0_0 w_2_0 w_6_0 w_3_0 w_11_0 at h
  obtain ⟨w_1_0, h⟩ := h
  obtain ⟨w_4_0, h⟩ := h
  extract_lets -merge w_5_0 at h
  obtain ⟨w_7_0, h⟩ := h
  obtain ⟨w_8_0, h⟩ := h
  extract_lets -merge w_9_0 w_10_0 at h
  obtain ⟨w_12_0, h⟩ := h
  extract_lets -merge w_13_0 at h
  obtain ⟨w_14_0, h⟩ := h
  obtain ⟨w_15_0, h⟩ := h
  extract_lets -merge w_16_0 w_17_0 at h
  obtain ⟨w_18_0, h⟩ := h
  extract_lets -merge w_19_0 at h
  obtain ⟨w_20_0, h⟩ := h
  extract_lets -merge w_21_0 at h
  obtain ⟨w_22_0, h⟩ := h
  extract_lets -merge w_23_0 w_24_0 at h
  obtain ⟨w_25_0, h⟩ := h
  extract_lets -merge w_26_0 w_27_0 w_28_0 at h
  obtain ⟨w_29_0, h⟩ := h
  extract_lets -merge w_30_0 at h
  simp only [Stage_encrypt.parallel_parallel_generatedRoot_454_380.constraints_0] at h
  obtain ⟨h1_0, -, -, h4_0, h7_0, h8_0, -, -, -, -, h12_0, h14_0, h15_0, h18_0, h20_0, h22_0, -, -, h25_0, h29_0, hout⟩ := h
  have hsm' : w_1_0 = fun _ _ ↦ reducePoly Q0 N sm :=
    modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h1_0 (fun _ _ ↦ sm) (by show smM = _; rw [hsm]; rfl)
  have hone' : w_7_0 = 1 := by
    rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN h7_0 (fun _ _ ↦ 1) (by
      show oneM = _; rw [hone]; funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  rw [gadgetMatrixRuns_unique h14_0 hg'] at h15_0
  have hgg := centeredRebaseRuns_unique h15_0 hg
  subst hgg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h18_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h29_0
  obtain ⟨hodd, hsub⟩ := ringAutomorphismRuns_eq (by inst_num) hN h12_0
  obtain ⟨_, hsub1⟩ := ringAutomorphismRuns_eq (by inst_num) hN h25_0
  have hsub' : w_12_0 = fun _ _ ↦ reducePoly Q0 N (substitution (sigma o).toNat (sigma_odd o) sm) := by
    rw [hsub, hsm']
    funext r c
    exact (reducePoly_substitution _ _ _).symm
  have hsub1' : w_25_0 = 1 := by
    rw [hsub1, hone']
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp
  have hAd : lutA model key Q0 6 3 (sigma o) =
      ![![w_4_0, w_20_0], ![w_8_0, w_22_0]] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact (hashSample_eq h4_0).symm
    · exact (hashSample_eq h20_0).symm
    · exact (hashSample_eq h8_0).symm
    · exact (hashSample_eq h22_0).symm
  rw [hAd]
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_19_0 = _
    simp only [w_19_0, w_17_0, w_16_0, w_13_0, w_10_0, w_5_0, w_9_0, w_11_0, hsm', hone', he0, hsub']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl
  · show w_30_0 = _
    simp only [w_30_0, w_28_0, w_27_0, w_26_0, w_24_0, w_21_0, w_23_0, w_11_0, hsm', hone', he1, hsub1']
    simp only [matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
    try exact rfl

end RefreshTopProof.Part_I
