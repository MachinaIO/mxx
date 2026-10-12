import TopAction

/-!
# Assembling the decoder columns

The six decoder columns are computed one by one and concatenated; the smudge minus the half
constant is small.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The entries of five nested concatenations of six `1 × 1` matrices. -/
theorem concat_six {q : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix q N 1 1} {z1 : ExactMatrix q N 1 2}
    {z2 : ExactMatrix q N 1 3} {z3 : ExactMatrix q N 1 4} {z4 : ExactMatrix q N 1 5}
    {z : ExactMatrix q N 1 6} (c1 : concatColumns x0 x1 z1) (c2 : concatColumns z1 x2 z2)
    (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4) (c5 : concatColumns z4 x5 z) :
    ∀ k : Fin 6, z 0 k = ![x0 0 0, x1 0 0, x2 0 0, x3 0 0, x4 0 0, x5 0 0] k := by
  intro k
  fin_cases k <;> simp [c5 0, c4 0, c3 0, c2 0, c1 0]

/-- The half constant `(Q₁ + 1) / 2`. -/
abbrev halfQ1 : Int := 618968813111010066723674601

/-- The half constant on every coefficient. -/
theorem half_poly : (matrixPolynomial [halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1] :
    ExactMatrix Q1 N 1 1) 0 0 = reducePoly Q1 N (∑ c : Fin 8, ((halfQ1 : Int) : ErrorPoly N) * Xe (n := N) c) := by
  simp only [matrixPolynomial, List.foldr, Fin.sum_univ_eight, map_add, map_mul, reducePoly_Xe,
    reducePoly_intCast]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two, X, pow_zero, pow_one, mul_one, mul_zero, add_zero]
  norm_num
  ring

/-- One coefficient of the smudge minus the half constant: the large first weight only shifts an
odd digit `t` by `(t - 1) / 2`. -/
theorem half_digit (t : Int) (ht : t % 2 = 1) :
    reducePoly Q1 N (((halfQ1 * t - halfQ1 : Int) : ErrorPoly N)) =
      reducePoly Q1 N (((t - 1) / 2 : Int) : ErrorPoly N) := by
  obtain ⟨k, hk⟩ : ∃ k, t = 2 * k + 1 := ⟨t / 2, by omega⟩
  subst hk
  have h1 : (2 * k + 1 - 1) / 2 = k := by omega
  rw [h1, show halfQ1 * (2 * k + 1) - halfQ1 = (Q1 : Int) * k + k by simp only [halfQ1, Q1]; ring,
    Int.cast_add, Int.cast_mul, map_add, reducePoly_modulus_mul, zero_add]

/-- Member `(c, r)` of a smudge: digit `r` of coefficient `c`. -/
def fin160 : Fin 8 × Fin 20 ≃ Fin 160 := finProdFinEquiv.trans (finCongr (by norm_num))

theorem fin160_val (c : Fin 8) (r : Fin 20) : (fin160 (c, r) : Nat) = r + 20 * c := rfl

/-- The smudge minus the half constant: digit `0` of each coefficient contributes `(d - 1) / 2`. -/
noncomputable def smudgeSmall (w : Fin 20 → Int) (d : Fin 160 → Int) : ErrorPoly N :=
  ∑ c : Fin 8, Xe (n := N) c * (((d (fin160 (c, 0)) - 1) / 2 +
    ∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ)) : Int) : ErrorPoly N)

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 65536 in
theorem smudge_minus_half (w : Fin 20 → Int) (hw : w 0 = halfQ1) (d : Fin 160 → Int)
    (hd : ∀ o, d o % 2 = 1) :
    reducePoly Q1 N (∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) * ((d o : Int) : ErrorPoly N))) -
      (matrixPolynomial [halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1] :
        ExactMatrix Q1 N 1 1) 0 0 = reducePoly Q1 N (smudgeSmall w d) := by
  rw [half_poly, ← map_sub]
  have hsum : (∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) * ((d o : Int) : ErrorPoly N))) =
      ∑ c : Fin 8, ∑ r : Fin 20, Xe (n := N) c *
        (((w r : Int) : ErrorPoly N) * ((d (fin160 (c, r)) : Int) : ErrorPoly N)) := by
    rw [← Equiv.sum_comp fin160, Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro c _
    apply Finset.sum_congr rfl
    intro r _
    have h1 : (fin160 (c, r) : Nat) / 20 = c := by rw [fin160_val]; omega
    have h2 : (⟨(fin160 (c, r) : Nat) % 20, Nat.mod_lt _ (by norm_num)⟩ : Fin 20) = r := by
      ext; simp only [fin160_val]; omega
    rw [h1, h2]
  rw [hsum, ← Finset.sum_sub_distrib]
  unfold smudgeSmall
  rw [map_sum, map_sum]
  apply Finset.sum_congr rfl
  intro c _
  rw [Fin.sum_univ_succ, hw]
  have hsplit : Xe (n := N) c * (((halfQ1 : Int) : ErrorPoly N) *
        ((d (fin160 (c, 0)) : Int) : ErrorPoly N)) +
      ∑ r : Fin 19, Xe (n := N) c * (((w r.succ : Int) : ErrorPoly N) *
        ((d (fin160 (c, r.succ)) : Int) : ErrorPoly N)) -
      ((halfQ1 : Int) : ErrorPoly N) * Xe (n := N) c =
      Xe (n := N) c * ((((halfQ1 * d (fin160 (c, 0)) - halfQ1 : Int) : ErrorPoly N)) +
        ((∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ)) : Int) : ErrorPoly N)) := by
    rw [mul_add, Int.cast_sum, Finset.mul_sum]
    simp only [Int.cast_mul, Int.cast_sub]
    ring
  rw [hsplit, map_mul, map_add, half_digit _ (hd _)]
  rw [← map_add, ← map_mul, Int.cast_add]

/-- The integer unit column. -/
noncomputable def unitColI {m : Nat} (k : Fin m) : ErrorMatrix N m 1 := fun i _ ↦ if i = k then 1 else 0

theorem unitCol_reduce {m : Nat} (k : Fin m) :
    unitCol (q := Q1) k = reduceMatrix Q1 N m 1 (unitColI k) := by
  funext i c
  simp only [unitCol, unitColI, reduceMatrix]
  split <;> simp

theorem mul_unitColI {rows m : Nat} (E : ErrorMatrix N rows m) (k : Fin m) :
    E * unitColI k = fun r _ ↦ E r k := by
  funext r c
  rw [Matrix.mul_apply, Finset.sum_eq_single k]
  · simp [unitColI]
  · intro i _ hi
    simp [unitColI, hi]
  · simp

theorem coeffBound_col {rows m : Nat} {E : ErrorMatrix N rows m} {B : Nat} (h : CoeffBound E B)
    (k : Fin m) : CoeffBound (E * unitColI k) B := by
  rw [mul_unitColI]
  intro r c i
  exact h r k i

set_option maxHeartbeats 4000000 in
/-- A decoder column is `Q_q y` plus a small integer polynomial. -/
theorem column_noise {y : ExactMatrix Q1 N 1 6} (k : Fin 6) {M : ErrorMatrix N 1 1} {BM : Nat}
    (hM : CoeffBound M BM) {ebI : ErrorMatrix N 1 16} {Beb : Nat} (heb : CoeffBound ebI Beb)
    {KI : ErrorMatrix N 16 1} {Bpre : Nat} (hK : CoeffBound KI Bpre)
    {egI e1I ewI : ErrorMatrix N 1 6} {Beg Be1 Bew : Nat} (heg : CoeffBound egI Beg)
    (he1 : CoeffBound e1I Be1) (hew : CoeffBound ewI Bew)
    {X0I X1I : ErrorMatrix N 6 6} (hX0 : CoeffBound X0I 16384) (hX1 : CoeffBound X1I 16384)
    {exI : ErrorMatrix N 1 4} {Bex : Nat} (hex : CoeffBound exI Bex)
    {YI : ErrorMatrix N 4 6} (hY : CoeffBound YI 16384) {esmI : ErrorMatrix N 1 1} {Besm : Nat}
    (hesm : CoeffBound esmI Besm) {out : ExactMatrix Q1 N 1 1}
    (hout : out = QqPoly • (y * unitCol (q := Q1) k) + reduceMatrix Q1 N 1 1 M +
      (reduceMatrix Q1 N 1 16 ebI * reduceMatrix Q1 N 16 1 KI -
        (reduceMatrix Q1 N 1 6 egI * reduceMatrix Q1 N 6 6 X0I +
          reduceMatrix Q1 N 1 6 e1I * reduceMatrix Q1 N 6 6 X1I -
          reduceMatrix Q1 N 1 4 exI * reduceMatrix Q1 N 4 6 YI + reduceMatrix Q1 N 1 6 ewI) *
          unitCol (q := Q1) k - reduceMatrix Q1 N 1 1 esmI)) :
    ∃ ν : ErrorMatrix N 1 1, CoeffBound ν (BM + 16 * N * Beb * Bpre +
        (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 + 4 * N * Bex * 16384 + Bew) + Besm) ∧
      out 0 0 = ((1152919668508929361 : Int) : ExactPoly Q1 N) * y 0 k + reducePoly Q1 N (ν 0 0) := by
  set ν : ErrorMatrix N 1 1 := M + ebI * KI - (egI * X0I + e1I * X1I - exI * YI + ewI) * unitColI k -
    esmI with hν
  refine ⟨ν, coeffBound_sub (coeffBound_sub (coeffBound_add hM (coeffBound_mul hN heb hK)) ?_) hesm, ?_⟩
  · exact coeffBound_col (coeffBound_add (coeffBound_sub (coeffBound_add
      (coeffBound_mul hN heg hX0) (coeffBound_mul hN he1 hX1)) (coeffBound_mul hN hex hY)) hew) k
  · have hr : reduceMatrix Q1 N 1 1 ν = reduceMatrix Q1 N 1 1 M +
        (reduceMatrix Q1 N 1 16 ebI * reduceMatrix Q1 N 16 1 KI -
          (reduceMatrix Q1 N 1 6 egI * reduceMatrix Q1 N 6 6 X0I +
            reduceMatrix Q1 N 1 6 e1I * reduceMatrix Q1 N 6 6 X1I -
            reduceMatrix Q1 N 1 4 exI * reduceMatrix Q1 N 4 6 YI + reduceMatrix Q1 N 1 6 ewI) *
            unitCol (q := Q1) k - reduceMatrix Q1 N 1 1 esmI) := by
      rw [hν, unitCol_reduce]
      simp only [reduceMatrix_add, reduceMatrix_sub, reduceMatrix_mul]
      abel
    rw [hout, add_assoc, ← hr, Matrix.add_apply, Matrix.smul_apply, mul_unitCol, smul_eq_mul]
    rfl

/-- A row from its entries. -/
noncomputable def rowMat {q m : Nat} (f : Fin m → ExactPoly q N) : ExactMatrix q N 1 m := fun _ k ↦ f k

/-- Six `1 × 1` rows of target `t_k Q_q` side by side: a row of target `Q_q t`. -/
theorem six_rows {s : ExactPoly Q1 N} {z : ExactMatrix Q1 N 1 6} {x : Fin 6 → ExactMatrix Q1 N 1 1}
    (hz : ∀ k, z 0 k = x k 0 0) {K : Fin 6 → Fin 2 → ExactMatrix Q1 N 1 1} {t : Fin 6 → ErrorPoly N}
    {e : Fin 6 → ErrorMatrix N 1 1}
    (hx : ∀ k, x k = maskPart s (K k) - reducePoly Q1 N (t k) •
      (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 (e k)) :
    z = maskPart s (fun i ↦ rowMat (fun k ↦ K k i 0 0)) - QqPoly • rowMat (fun k ↦ reducePoly Q1 N (t k)) +
      reduceMatrix Q1 N 1 6 (fun _ k ↦ e k 0 0) := by
  funext r k
  obtain rfl : r = 0 := Subsingleton.elim _ _
  rw [hz, hx]
  simp [maskPart, matrixPolynomial_single, QqPoly, reduceMatrix, rowMat, mul_comm]

theorem polyNorm_Xe_int (c : Nat) (C : Int) : polyNorm (Xe (n := N) c * (C : ErrorPoly N)) ≤ C.natAbs := by
  rw [mul_comm]
  refine (polyNorm_intCast_mul_le C _).trans ?_
  have := polyNorm_Xe hN1 c
  calc C.natAbs * polyNorm (Xe (n := N) c) ≤ C.natAbs * 1 := Nat.mul_le_mul_left _ this
    _ = C.natAbs := mul_one _

/-- The smudge remainder is bounded by the bound of its digit sums. -/
theorem smudgeSmall_bound (w : Fin 20 → Int) (d : Fin 160 → Int) {A : Nat}
    (hA : ∀ c : Fin 8, ((d (fin160 (c, 0)) - 1) / 2 +
      ∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ))).natAbs ≤ A) :
    CoeffBound (fun _ _ ↦ smudgeSmall w d : ErrorMatrix N 1 1) (8 * A) := by
  intro r c i
  refine (coeff_natAbs_le_polyNorm _ i).trans ?_
  unfold smudgeSmall
  refine (polyNorm_sum_le _ _).trans ?_
  calc ∑ c : Fin 8, polyNorm (Xe (n := N) c * ((((d (fin160 (c, 0)) - 1) / 2 +
        ∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ)) : Int)) : ErrorPoly N))
      ≤ ∑ _c : Fin 8, A := Finset.sum_le_sum fun c _ ↦ (polyNorm_Xe_int _ _).trans (hA c)
    _ = 8 * A := by simp

/-- A digit sum of odd digits `|d| ≤ 3`. -/
theorem digit_sum_bound (w : Fin 20 → Int) (d : Fin 160 → Int) (hd : ∀ o, |d o| ≤ 3) (c : Fin 8) :
    ((d (fin160 (c, 0)) - 1) / 2 + ∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ))).natAbs ≤
      2 + 3 * ∑ r : Fin 19, (w r.succ).natAbs := by
  have h0 := hd (fin160 (c, 0))
  have hs : |∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ))| ≤
      ∑ r : Fin 19, |w r.succ| * 3 := by
    refine (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun r _ ↦ ?_)
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left (hd _) (abs_nonneg _)
  zify
  have hsum : ∑ r : Fin 19, |w r.succ| * 3 = 3 * ∑ r : Fin 19, |w r.succ| := by
    rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun r _ ↦ mul_comm _ _
  rw [hsum] at hs
  have hh : |(d (fin160 (c, 0)) - 1) / 2| ≤ 2 := by
    rw [abs_le] at h0 ⊢; constructor <;> omega
  calc |(d (fin160 (c, 0)) - 1) / 2 + ∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ))|
      ≤ |(d (fin160 (c, 0)) - 1) / 2| + |∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ))| :=
        abs_add_le _ _
    _ ≤ 2 + 3 * ∑ r : Fin 19, |w r.succ| := add_le_add hh hs

end RefreshTopProof
