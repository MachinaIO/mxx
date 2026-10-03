import NandSetup

/-!
# The preprocessing stage

Preprocessing publishes, for every round, the bottom row of the RHS ciphertext of `X^(8 b)`, whose
top row is sampled from the hash key, and the switching ciphertexts `B_σ` of the round's LUT.
-/

namespace NandProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path loopPath : List Nat}

/-- The top row of an RHS ciphertext, sampled from the hash key. -/
noncomputable def rhsTop (model : HashModel) (key : ByteArray) (q m : Nat) (tagPrefix : Blob) :
    Fin 2 → ExactMatrix q N 1 m :=
  fun j ↦ model.sample q N 1 m key (completeHashTag tagPrefix
    [.bytes componentBytes, .u64Le (j : Int)])

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `0` at level `1`: `s top_j + bot_j = (P7 ν r_j) g + e_j`. -/
theorem preprocess_rhs0 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 12, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q7 N sp • rhsTop model key Q7 12 rhsPrefix0 j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1,
            outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q7 N (((P7 : Nat) : Int) * message c0 e0 * unitPair sp j) • g +
          reduceMatrix Q7 N 1 12 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h14_0 := hb.2.2.2.2.2.2.2.2.2.2.2.1
  have h21_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h22_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h33_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h41_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h50_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  have hgg' := gadgetMatrixRuns_unique h21_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h22_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h33_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h50_0
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have htop : ∀ j, rhsTop model key Q7 12 rhsPrefix0 j = ![w.w_14_0, w.w_41_0] j := by
    intro j
    fin_cases j
    · exact (hashSample_eq h14_0).symm
    · exact (hashSample_eq h41_0).symm
  simp only [htop]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq, Matrix.mul_apply, unitPair, he0, he1,
      reduceMatrix, matrixPolynomial_single, multiplyMonomial_liftInteger, map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `1` at level `3`. -/
theorem preprocess_rhs1 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q5 N 1 10} {g : ExactMatrix Q6 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N sp • rhsTop model key Q6 10 rhsPrefix1 j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q6 N (((P6 : Nat) : Int) * message c1 e1 * unitPair sp j) • g +
          reduceMatrix Q6 N 1 10 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h69_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h95_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h70_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h71_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h79_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h62_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h91_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h83_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h101_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  dsimp (config := { zetaDelta := true }) only at h69_0 h95_0 h79_0
  have hpay := modulusReduce_poly hQ7 hQ6 h69_0
  have hone := modulusReduce_one hQ7 hQ6 h95_0
  rw [multiplyMonomial_liftInteger] at h79_0
  have hnu := modulusReduce_poly hQ7 hQ6 h79_0
  have hgg' := gadgetMatrixRuns_unique h70_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h71_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h83_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h101_0
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have htop : ∀ j, rhsTop model key Q6 10 rhsPrefix1 j = ![w.w_62_0, w.w_91_0] j := by
    intro j
    fin_cases j
    · exact (hashSample_eq h62_0).symm
    · exact (hashSample_eq h91_0).symm
  simp only [htop]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `2` at level `5`. -/
theorem preprocess_rhs2 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q4 N 1 8} {g : ExactMatrix Q5 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N sp • rhsTop model key Q5 8 rhsPrefix2 j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q5 N (((P5 : Nat) : Int) * message c2 e2 * unitPair sp j) • g +
          reduceMatrix Q5 N 1 8 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h120_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h146_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h121_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h122_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h130_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h113_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h142_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h134_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h152_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  dsimp (config := { zetaDelta := true }) only at h120_0 h146_0 h130_0
  have hpay := modulusReduce_poly hQ7 hQ5 h120_0
  have hone := modulusReduce_one hQ7 hQ5 h146_0
  rw [multiplyMonomial_liftInteger] at h130_0
  have hnu := modulusReduce_poly hQ7 hQ5 h130_0
  have hgg' := gadgetMatrixRuns_unique h121_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h122_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h134_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h152_0
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have htop : ∀ j, rhsTop model key Q5 8 rhsPrefix2 j = ![w.w_113_0, w.w_142_0] j := by
    intro j
    fin_cases j
    · exact (hashSample_eq h113_0).symm
    · exact (hashSample_eq h142_0).symm
  simp only [htop]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The RHS ciphertext of round `3` at level `7`. -/
theorem preprocess_rhs3 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' : ExactMatrix Q3 N 1 6} {g : ExactMatrix Q4 N 1 6}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 6 g') (hg : centeredRebaseRuns g' g) :
    ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q4 N sp • rhsTop model key Q4 6 rhsPrefix3 j +
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1] j =
        reducePoly Q4 N (((P4 : Nat) : Int) * message c3 e3 * unitPair sp j) • g +
          reduceMatrix Q4 N 1 6 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h171_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h197_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h172_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h173_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h181_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h164_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h193_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h185_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have h203_0 := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  dsimp (config := { zetaDelta := true }) only at h171_0 h197_0 h181_0
  have hpay := modulusReduce_poly hQ7 hQ4 h171_0
  have hone := modulusReduce_one hQ7 hQ4 h197_0
  rw [multiplyMonomial_liftInteger] at h181_0
  have hnu := modulusReduce_poly hQ7 hQ4 h181_0
  have hgg' := gadgetMatrixRuns_unique h172_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h173_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h185_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h203_0
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  have htop : ∀ j, rhsTop model key Q4 6 rhsPrefix3 j = ![w.w_164_0, w.w_193_0] j := by
    intro j
    fin_cases j
    · exact (hashSample_eq h164_0).symm
    · exact (hashSample_eq h193_0).symm
  simp only [htop]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp (config := { zetaDelta := true }) [matrixAdd, matrixSub, matrixMulScalarRight_eq,
      Matrix.mul_apply, unitPair, he0, he1, hpay, hone, hnu, reduceMatrix, matrixPolynomial_single,
      map_ofNat] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `0` at level `2`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem preprocess_switch0 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q6 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 12 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 12, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q6 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.1 i, outputs.2.1 i] j + ![outputs.2.2.1 i, outputs.2.2.2.1 i] j =
        (lutCoefficient Q6 inverse6 (MxxExponentLutNand.table 0) (sigma i).toNat *
          reducePoly Q6 N (1 * unitPair sp j)) • g + reduceMatrix Q6 N 1 12 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h3_3 := hb.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_107_0, w_108_0, w_111_0, w_117_0, w_121_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h35_0, -, -, -, -, -, -, h40_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h51_0, -, -, -, -, -, -, h59_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h75_0, -, -, -, -, -, -, h82_0, -, -, -, -, -, -, h87_0, -, -, -, -, -, -, h93_0, -, -, -, -, -, -, h98_0, h107_0, h108_0, h111_0, h117_0, h121_0, hout⟩ := h3_3 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ6 hN h5_0
  have hw4 := modulusReduce_poly hQ7 hQ6 h4_0
  have hgg' := gadgetMatrixRuns_unique h107_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h108_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h111_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h121_0
  have hw117 := modulusReduce_one hQ7 hQ6 h117_0
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ6 hN h12_0
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ6 hN h17_0
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ6 hN h23_0
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ6 hN h28_0
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ6 hN h35_0
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ6 hN h40_0
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ6 hN h46_0
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ6 hN h51_0
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ6 hN h59_0
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ6 hN h64_0
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ6 hN h70_0
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ6 hN h75_0
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ6 hN h82_0
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ6 hN h87_0
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ6 hN h93_0
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ6 hN h98_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `1` at level `4`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem preprocess_switch1 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q5 N 1 10}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 10 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 10, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q5 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.1 i, outputs.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q5 inverse5 (MxxExponentLutNand.table 1) (sigma i).toNat *
          reducePoly Q5 N (1 * unitPair sp j)) • g + reduceMatrix Q5 N 1 10 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h4_3 := hb.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_107_0, w_108_0, w_111_0, w_117_0, w_121_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h35_0, -, -, -, -, -, -, h40_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h51_0, -, -, -, -, -, -, h59_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h75_0, -, -, -, -, -, -, h82_0, -, -, -, -, -, -, h87_0, -, -, -, -, -, -, h93_0, -, -, -, -, -, -, h98_0, h107_0, h108_0, h111_0, h117_0, h121_0, hout⟩ := h4_3 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ5 hN h5_0
  have hw4 := modulusReduce_poly hQ7 hQ5 h4_0
  have hgg' := gadgetMatrixRuns_unique h107_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h108_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h111_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h121_0
  have hw117 := modulusReduce_one hQ7 hQ5 h117_0
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ5 hN h12_0
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ5 hN h17_0
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ5 hN h23_0
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ5 hN h28_0
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ5 hN h35_0
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ5 hN h40_0
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ5 hN h46_0
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ5 hN h51_0
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ5 hN h59_0
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ5 hN h64_0
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ5 hN h70_0
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ5 hN h75_0
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ5 hN h82_0
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ5 hN h87_0
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ5 hN h93_0
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ5 hN h98_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `2` at level `6`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem preprocess_switch2 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    {g' g : ExactMatrix Q4 N 1 8}
    (hg' : gadgetMatrixRuns Backend.backend 1048576 8 g') (hg : centeredRebaseRuns g' g)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 8, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q4 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q4 inverse4 (MxxExponentLutNand.table 2) (sigma i).toNat *
          reducePoly Q4 N (1 * unitPair sp j)) • g + reduceMatrix Q4 N 1 8 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h5_3 := hb.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_107_0, w_108_0, w_111_0, w_117_0, w_121_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h35_0, -, -, -, -, -, -, h40_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h51_0, -, -, -, -, -, -, h59_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h75_0, -, -, -, -, -, -, h82_0, -, -, -, -, -, -, h87_0, -, -, -, -, -, -, h93_0, -, -, -, -, -, -, h98_0, h107_0, h108_0, h111_0, h117_0, h121_0, hout⟩ := h5_3 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ4 hN h5_0
  have hw4 := modulusReduce_poly hQ7 hQ4 h4_0
  have hgg' := gadgetMatrixRuns_unique h107_0 hg'
  subst hgg'
  have hgg := centeredRebaseRuns_unique h108_0 hg
  subst hgg
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h111_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h121_0
  have hw117 := modulusReduce_one hQ7 hQ4 h117_0
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ4 hN h12_0
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ4 hN h17_0
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ4 hN h23_0
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ4 hN h28_0
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ4 hN h35_0
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ4 hN h40_0
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ4 hN h46_0
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ4 hN h51_0
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ4 hN h59_0
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ4 hN h64_0
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ4 hN h70_0
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ4 hN h75_0
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ4 hN h82_0
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ4 hN h87_0
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ4 hN h93_0
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ4 hN h98_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw117, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, reducePoly_substitution, sigma] <;>
    ring

set_option maxHeartbeats 8000000 in
/-- The switching ciphertexts of round `3` at level `8`: `σ_i(s) B_0j + B_1j = (C_i r_j) g + e`. -/
theorem preprocess_switch3 {model : HashModel} {key : ByteArray} {payload : ExactMatrix Q7 N 1 1}
    {c0 e0 c1 e1 c2 e2 c3 e3 : Int} {outputs}
    (h : Stage_preprocess.generatedRoot Backend.backend model tape path GeneratedClaim.stage_2_params
      (key, payload, c0, e0, c1, e1, c2, e2, c3, e3, ()) outputs)
    {sp : ErrorPoly N} (hpayload : payload = fun _ _ ↦ reducePoly Q7 N sp)
    (i : Fin 16) :
    ∃ e : Fin 2 → ErrorMatrix N 1 1, (∀ j, CoeffBound (e j) 26) ∧ ∀ j,
      reducePoly Q3 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j + ![outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i, outputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 i] j =
        (lutCoefficient Q3 inverse3 (MxxExponentLutNand.table 3) (sigma i).toNat *
          reducePoly Q3 N ((((Q3 / 2 : Nat) : Int) : ErrorPoly N) * unitPair sp j)) •
          (1 : ExactMatrix Q3 N 1 1) + reduceMatrix Q3 N 1 1 (e j) := by
  obtain ⟨w, hb⟩ := h
  unfold Stage_preprocess.generatedRoot.body at hb
  extract_lets at hb
  have h6_3 := hb.2.2.2.2.2.2.2.1
  have hout := hb.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2
  subst hout hpayload
  obtain ⟨w_1_0, w_2_0, w_4_0, w_5_0, w_12_0, w_17_0, w_23_0, w_28_0, w_35_0, w_40_0, w_46_0, w_51_0, w_59_0, w_64_0, w_70_0, w_75_0, w_82_0, w_87_0, w_93_0, w_98_0, w_110_0, w_116_0, w_120_0, -, -, h1_0, -, -, h2_0, h4_0, -, -, h5_0, -, -, -, -, -, -, h12_0, -, -, -, -, -, -, h17_0, -, -, -, -, -, -, h23_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h35_0, -, -, -, -, -, -, h40_0, -, -, -, -, -, -, h46_0, -, -, -, -, -, -, h51_0, -, -, -, -, -, -, h59_0, -, -, -, -, -, -, h64_0, -, -, -, -, -, -, h70_0, -, -, -, -, -, -, h75_0, -, -, -, -, -, -, h82_0, -, -, -, -, -, -, h87_0, -, -, -, -, -, -, h93_0, -, -, -, -, -, -, h98_0, h110_0, h116_0, h120_0, hout⟩ := h6_3 i
  obtain ⟨_, hw5⟩ := ringAutomorphismRuns_eq hQ3 hN h5_0
  have hw4 := modulusReduce_poly hQ7 hQ3 h4_0
  obtain ⟨e0', he0, hb0⟩ := gaussian_bound h110_0
  obtain ⟨e1', he1, hb1⟩ := gaussian_bound h120_0
  have hw116 := modulusReduce_one hQ7 hQ3 h116_0
  obtain ⟨_, a0⟩ := ringAutomorphismRuns_eq hQ3 hN h12_0
  obtain ⟨_, a1⟩ := ringAutomorphismRuns_eq hQ3 hN h17_0
  obtain ⟨_, a2⟩ := ringAutomorphismRuns_eq hQ3 hN h23_0
  obtain ⟨_, a3⟩ := ringAutomorphismRuns_eq hQ3 hN h28_0
  obtain ⟨_, a4⟩ := ringAutomorphismRuns_eq hQ3 hN h35_0
  obtain ⟨_, a5⟩ := ringAutomorphismRuns_eq hQ3 hN h40_0
  obtain ⟨_, a6⟩ := ringAutomorphismRuns_eq hQ3 hN h46_0
  obtain ⟨_, a7⟩ := ringAutomorphismRuns_eq hQ3 hN h51_0
  obtain ⟨_, a8⟩ := ringAutomorphismRuns_eq hQ3 hN h59_0
  obtain ⟨_, a9⟩ := ringAutomorphismRuns_eq hQ3 hN h64_0
  obtain ⟨_, a10⟩ := ringAutomorphismRuns_eq hQ3 hN h70_0
  obtain ⟨_, a11⟩ := ringAutomorphismRuns_eq hQ3 hN h75_0
  obtain ⟨_, a12⟩ := ringAutomorphismRuns_eq hQ3 hN h82_0
  obtain ⟨_, a13⟩ := ringAutomorphismRuns_eq hQ3 hN h87_0
  obtain ⟨_, a14⟩ := ringAutomorphismRuns_eq hQ3 hN h93_0
  obtain ⟨_, a15⟩ := ringAutomorphismRuns_eq hQ3 hN h98_0
  simp only [Prod.mk.injEq] at hout
  obtain ⟨ho0, ho1, ho2, ho3, -⟩ := hout
  refine ⟨![e0', e1'], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  rw [lutCoefficient_eq (sigma_odd i)]
  fin_cases j <;> funext r c <;> obtain rfl : r = 0 := Subsingleton.elim _ _ <;>
    obtain rfl : c = 0 := Subsingleton.elim _ _ <;>
    simp [ho0, ho1, ho2, ho3, matrixAdd, matrixNeg, Matrix.mul_apply, unitPair, he0, he1, hw4, hw5,
      hw116, reduceMatrix, matrixPolynomial_single, Fin.sum_univ_succ,
      MxxExponentLutNand.table, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15,
      rotationPolynomial, substitution_root_pow, map_ofNat, reducePoly_substitution, sigma,
      Matrix.one_apply_eq] <;>
    ring


end NandProof
