import TopAction

/-!
# The hint and the one encoding of the refresh

The encryption stage encrypts the hint `b = [s, 1] A + e` under the trapdoor's public matrix `A`
and the one encoding's second block `c₁ = s A₀₁ + A₁₁ - G + e`, for the mask `s`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

theorem mask_reduce {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a : ExactMatrix Q1 N 1 1} (ha : modulusReduceRuns w4 a) :
    a = fun _ _ ↦ reducePoly Q1 N sm :=
  modulusReduceRuns_reduce (by norm_num) (by norm_num) hN ha (fun _ _ ↦ sm) (by rw [hw4]; rfl)

theorem one_reduce {b : ExactMatrix Q1 N 1 1} (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b) :
    b = 1 := by
  rw [modulusReduceRuns_reduce (by norm_num) (by norm_num) hN hb (fun _ _ ↦ 1) (by
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
  funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]

/-- The hint: `[s, 1] A + e`. -/
theorem hint_value {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a b : ExactMatrix Q1 N 1 1}
    (ha : modulusReduceRuns w4 a) (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b)
    {r : ExactMatrix Q1 N 1 (1 + 1)} (hr : concatColumns a b r) {A : ExactMatrix Q1 N (1 + 1) 16}
    {e : ExactMatrix Q1 N 1 16} (he : gaussianSample (4 / 1) 26 e) :
    ∃ eI : ErrorMatrix N 1 16, CoeffBound eI 26 ∧
      matrixAdd (matrixMul r A) e = maskPart (reducePoly Q1 N sm) (fun i ↦ rowOf A i) +
        reduceMatrix Q1 N 1 16 eI := by
  obtain ⟨eI, heI, hb'⟩ := gaussian_bound he
  refine ⟨eI, hb', ?_⟩
  rw [mask_reduce hw4 ha] at hr
  rw [one_reduce hb] at hr
  rw [heI]
  congr 1
  funext i c
  obtain rfl : i = 0 := Subsingleton.elim _ _
  simp only [matrixMul, Matrix.mul_apply, hr 0, maskPart, rowOf,
    Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
  simp [Fin.sum_univ_two, Matrix.one_apply]

/-- The one encoding's second block: `s A₀₁ + A₁₁ - G + e`. -/
theorem one_value {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a b : ExactMatrix Q1 N 1 1}
    (ha : modulusReduceRuns w4 a) (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b)
    {A01 A11 G : ExactMatrix Q1 N 1 6} {e : ExactMatrix Q1 N 1 6} (he : gaussianSample (4 / 1) 26 e) :
    ∃ eI : ErrorMatrix N 1 6, CoeffBound eI 26 ∧
      matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft a A01) (matrixMulScalarLeft b A11))
        (matrixMulScalarLeft b (matrixMulScalarLeft (matrixPolynomial [1]) G))) e =
        maskPart (reducePoly Q1 N sm) ![A01, A11] - G + reduceMatrix Q1 N 1 6 eI := by
  obtain ⟨eI, heI, hb'⟩ := gaussian_bound he
  refine ⟨eI, hb', ?_⟩
  rw [mask_reduce hw4 ha, one_reduce hb, heI]
  funext i c
  simp only [matrixAdd, matrixSub, matrixMulScalarLeft, maskPart, Matrix.add_apply, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single]
  simp

end RefreshTopProof

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 4000000 in
/-- The right-hand side of the offset's private multiplication the preprocessing computes:
`s top_j + bot_j = p m (s, 1)_j G + e_j`. -/
theorem offset_rhs {q qs m p : Nat} (hq : 1 < q) (hqs : 1 < qs) {sp : ErrorPoly N}
    {sec : ExactMatrix qs N 1 1} (hsec : sec = fun _ _ ↦ reducePoly qs N sp)
    {s : ExactMatrix q N 1 1} (hs : modulusReduceRuns sec s)
    {one : ExactMatrix q N 1 1} (hone : modulusReduceRuns (1 : ExactMatrix qs N 1 1) one)
    {G : ExactMatrix q N 1 m} {c e : Int} {msg : ExactMatrix q N 1 1}
    (hmsg : modulusReduceRuns (multiplyMonomial (liftInteger c : ExactMatrix qs N 1 1) e) msg)
    {top0 top1 e0 e1 : ExactMatrix q N 1 m} (he0 : gaussianSample (4 / 1) 26 e0)
    (he1 : gaussianSample (4 / 1) 26 e1) :
    ∃ eB : Fin 2 → ErrorMatrix N 1 m, (∀ j, CoeffBound (eB j) 26) ∧ ∀ j,
      reducePoly q N sp • ![top0, top1] j +
        ![matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight (matrixMulScalarLeft s G)
            (matrixPolynomial [(p : Int)])) msg) (matrixMulScalarLeft s top0)) e0,
          matrixAdd (matrixSub (matrixMulScalarRight (matrixMulScalarRight (matrixMulScalarLeft one G)
            (matrixPolynomial [(p : Int)])) msg) (matrixMulScalarLeft s top1)) e1] j =
        reducePoly q N (((p : Nat) : Int) * message (n := N) c e * unitPair sp j) • G +
          reduceMatrix q N 1 m (eB j) := by
  obtain ⟨E0, hE0, hb0⟩ := gaussian_bound he0
  obtain ⟨E1, hE1, hb1⟩ := gaussian_bound he1
  have hsp : s = fun _ _ ↦ reducePoly q N sp :=
    modulusReduceRuns_reduce hqs hq hN hs (fun _ _ ↦ sp) (by rw [hsec]; rfl)
  have hone' : one = 1 := by
    rw [modulusReduceRuns_reduce hqs hq hN hone (fun _ _ ↦ 1) (by
      funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
      obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]
  have hnu : msg = fun _ _ ↦ reducePoly q N (message (n := N) c e) :=
    modulusReduceRuns_reduce hqs hq hN hmsg (fun _ _ ↦ message (n := N) c e)
      (by rw [multiplyMonomial_liftInteger]; rfl)
  refine ⟨![E0, E1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero]
    rw [hsp, hnu, hE0]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, Matrix.head_cons]
    rw [hsp, hnu, hone', hE1]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c'
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

end RefreshTopProof

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

/-- A preimage under a sampled trapdoor solves the target within the cutoff. -/
theorem preimage_sampled {q n rows inner cols : Nat} {Token : Type} {backend : BackendContext}
    {sigma : Rat} {base digits cutoff : Int} {A : ExactMatrix q n rows inner}
    {T : TrapdoorValue (ExactMatrix q n rows inner) Token}
    (hT : trapdoorSample backend sigma base digits cutoff A T)
    {target : ExactMatrix q n rows cols} {cut : Nat} {P : ExactMatrix q n inner cols}
    (hP : preimageRunsDispatched backend A T target cut P) :
    A * P = target ∧ PreimageWithin P cut := by
  rcases hP.2 with h | h
  · exact ⟨preimageRuns_equation h, preimageRuns_bounded h⟩
  · exact absurd (h.1.symm.trans (trapdoorSample_sampled hT)) (by decide)

end RefreshTopProof
