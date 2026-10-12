import TopDecode

/-! The recombination of the four slots' residues modulo `Q₀ = q₀ q₁ q₂ q₃` (the decode stage of
the refresh of every CRT slot). -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem coeff_intCast_mul {q : Nat} (K : Int) (p : ExactPoly q N) (i : Fin N) :
    ((K : ExactPoly q N) * p).coeff i = (K : ZMod q) * p.coeff i := by
  rw [← map_intCast (algebraMap (ZMod q) (ExactPoly q N)) K, Negacyclic.coeff_smul]

/-- The CRT basis of `Q₀`: `K_c ≡ [c = c']` modulo `q_c'`. -/
abbrev K0 : Int := 890511653275736298117428745327756462
abbrev K1 : Int := 673373482644736804462909142109450233
abbrev K2 : Int := 676801805491843546832002660143739331
abbrev K3 : Int := 417760661884955689660474101877988633

/-- Integers congruent to `V` modulo each slot's prime recombine to `V` modulo `Q₀`. -/
theorem crt_int {x0 x1 x2 x3 V : Int} (h0 : (x0 : ZMod q0) = (V : ZMod q0))
    (h1 : (x1 : ZMod q1) = (V : ZMod q1)) (h2 : (x2 : ZMod q2) = (V : ZMod q2))
    (h3 : (x3 : ZMod q3) = (V : ZMod q3)) :
    ((x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 : Int) : ZMod Q0) = (V : ZMod Q0) := by
  rw [ZMod.intCast_eq_intCast_iff] at h0 h1 h2 h3 ⊢
  have e0 : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ V [ZMOD (q0 : Int)] := by
    have : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ x0 * 1 + x1 * 0 + x2 * 0 + x3 * 0 [ZMOD (q0 : Int)] :=
      (((Int.ModEq.refl x0).mul (by decide)).add ((Int.ModEq.refl x1).mul (by decide))).add
        ((Int.ModEq.refl x2).mul (by decide)) |>.add ((Int.ModEq.refl x3).mul (by decide))
    exact this.trans (by simpa using h0)
  have e1 : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ V [ZMOD (q1 : Int)] := by
    have : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ x0 * 0 + x1 * 1 + x2 * 0 + x3 * 0 [ZMOD (q1 : Int)] :=
      (((Int.ModEq.refl x0).mul (by decide)).add ((Int.ModEq.refl x1).mul (by decide))).add
        ((Int.ModEq.refl x2).mul (by decide)) |>.add ((Int.ModEq.refl x3).mul (by decide))
    exact this.trans (by simpa using h1)
  have e2 : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ V [ZMOD (q2 : Int)] := by
    have : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ x0 * 0 + x1 * 0 + x2 * 1 + x3 * 0 [ZMOD (q2 : Int)] :=
      (((Int.ModEq.refl x0).mul (by decide)).add ((Int.ModEq.refl x1).mul (by decide))).add
        ((Int.ModEq.refl x2).mul (by decide)) |>.add ((Int.ModEq.refl x3).mul (by decide))
    exact this.trans (by simpa using h2)
  have e3 : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ V [ZMOD (q3 : Int)] := by
    have : x0 * K0 + x1 * K1 + x2 * K2 + x3 * K3 ≡ x0 * 0 + x1 * 0 + x2 * 0 + x3 * 1 [ZMOD (q3 : Int)] :=
      (((Int.ModEq.refl x0).mul (by decide)).add ((Int.ModEq.refl x1).mul (by decide))).add
        ((Int.ModEq.refl x2).mul (by decide)) |>.add ((Int.ModEq.refl x3).mul (by decide))
    exact this.trans (by simpa using h3)
  have e01 := (Int.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨e0, e1⟩
  have e012 := (Int.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨e01, e2⟩
  have e0123 := (Int.modEq_and_modEq_iff_modEq_mul (by decide)).mp ⟨e012, e3⟩
  exact e0123

/-- A residue's centered lift is congruent to the value modulo the residue's prime. -/
theorem lift_residue {q : Nat} (hq : 1 < q) {r : ExactMatrix q N 1 6} {v : ExactMatrix Q0 N 1 6}
    (h : Residue r v) (c : Fin 6) (i : Fin N) :
    (((centeredIntLift (r 0 c)).coeff i : Int) : ZMod q) = ((((v 0 c).coeff i).val : Int) : ZMod q) := by
  have e := congrArg (fun p : ExactPoly q N ↦ p.coeff i) (reducePoly_centeredIntLift hq hN (r 0 c))
  simp only at e
  rw [reducePoly_coeff_eq hq hN] at e
  rw [e, h c i, Int.cast_natCast]

/-- The decode stage's recombination of the four slots' residues of `v` is `v`. -/
theorem crt_matrix {v : ExactMatrix Q0 N 1 6} {r0 : ExactMatrix q0 N 1 6} {r1 : ExactMatrix q1 N 1 6}
    {r2 : ExactMatrix q2 N 1 6} {r3 : ExactMatrix q3 N 1 6} (h0 : Residue r0 v) (h1 : Residue r1 v)
    (h2 : Residue r2 v) (h3 : Residue r3 v) {l0 l1 l2 l3 : ExactMatrix Q0 N 1 6}
    (hl0 : centeredRebaseRuns r0 l0) (hl1 : centeredRebaseRuns r1 l1) (hl2 : centeredRebaseRuns r2 l2)
    (hl3 : centeredRebaseRuns r3 l3) :
    matrixAdd (matrixAdd (matrixAdd (matrixMulScalarRight l0 (matrixPolynomial [K0]))
      (matrixMulScalarRight l1 (matrixPolynomial [K1]))) (matrixMulScalarRight l2 (matrixPolynomial [K2])))
      (matrixMulScalarRight l3 (matrixPolynomial [K3])) = v := by
  haveI : Fact (1 < Q0) := ⟨by decide⟩
  rw [centeredRebaseRuns_reduce hl0, centeredRebaseRuns_reduce hl1, centeredRebaseRuns_reduce hl2,
    centeredRebaseRuns_reduce hl3]
  funext row c
  obtain rfl : row = 0 := Subsingleton.elim _ _
  apply negacyclic_ext hN
  intro i
  simp only [matrixAdd, matrixMulScalarRight_eq, matrixPolynomial_single, Matrix.add_apply,
    Matrix.smul_apply, smul_eq_mul, Negacyclic.coeff_add, coeff_intCast_mul,
    reducePoly_coeff_eq (by decide : 1 < Q0) hN]
  have e := crt_int (lift_residue (by decide) h0 c i) (lift_residue (by decide) h1 c i)
    (lift_residue (by decide) h2 c i) (lift_residue (by decide) h3 c i)
  push_cast at e
  simp only [ZMod.natCast_zmod_val] at e
  rw [← e]
  ring

theorem matrixSub_eq {q n r c : Nat} (a b : ExactMatrix q n r c) : matrixSub a b = a - b := rfl

/-- A column of a small difference at `Q₀` decodes to zero. -/
theorem column_zeroQ0 (E : ErrorMatrix N 1 6) (hE : ∀ c i, 4 * |(E 0 c).coeff i| + 4 < (Q0 : Int))
    {x : ExactMatrix Q0 N 1 1} {a b : Int}
    (h : sliceMatrix (reduceMatrix Q0 N 1 6 E) 0 1 a b x) (ha : a.toNat < 6) {position out : Int}
    (hd : thresholdDecode 2 8 position x out) : out = 0 :=
  decode_small (by decide) (by decide) (slice_entry h ha) (fun i ↦ hE _ i) hd

end RefreshTopProof
