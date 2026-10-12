import TopChain
import TopLinkC
import TopFinal
import TopInst

/-! The lemmas of the decoding ring of instance `i0`: those of the shared `Top*` files that
mention `Q₁`, at `R i0` (written by `generator/gen_local.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof

/-- The half constant on every coefficient. -/
theorem half_poly {i0 : Nat} [Fact (i0 < 8)] : (matrixPolynomial [(halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int))] :
    ExactMatrix (R i0) N 1 1) 0 0 = reducePoly (R i0) N (∑ c : Fin 8, (((halfZ (i0 : Int)) : Int) : ErrorPoly N) * Xe (n := N) c) := by
  simp only [matrixPolynomial, List.foldr, Fin.sum_univ_eight, map_add, map_mul, reducePoly_Xe,
    reducePoly_intCast]
  simp only [Fin.val_zero, Fin.val_one, Fin.val_two, X, pow_zero, pow_one, mul_one, mul_zero, add_zero]
  norm_num
  ring

/-- One coefficient of the smudge minus the half constant: the large first weight only shifts an
odd digit `t` by `(t - 1) / 2`. -/
theorem half_digit {i0 : Nat} [Fact (i0 < 8)] (t : Int) (ht : t % 2 = 1) :
    reducePoly (R i0) N ((((halfZ (i0 : Int)) * t - (halfZ (i0 : Int)) : Int) : ErrorPoly N)) =
      reducePoly (R i0) N (((t - 1) / 2 : Int) : ErrorPoly N) := by
  obtain ⟨k, hk⟩ : ∃ k, t = 2 * k + 1 := ⟨t / 2, by omega⟩
  subst hk
  have h1 : (2 * k + 1 - 1) / 2 = k := by omega
  rw [h1, show (halfZ (i0 : Int)) * (2 * k + 1) - (halfZ (i0 : Int)) = ((R i0) : Int) * k + k by
      linear_combination k * half_two i0 hi0,
    Int.cast_add, Int.cast_mul, map_add, reducePoly_modulus_mul, zero_add]

set_option maxHeartbeats 4000000 in
set_option maxRecDepth 65536 in
theorem smudge_minus_half {i0 : Nat} [Fact (i0 < 8)] (w : Fin 20 → Int) (hw : w 0 = (halfZ (i0 : Int))) (d : Fin 160 → Int)
    (hd : ∀ o, d o % 2 = 1) :
    reducePoly (R i0) N (∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) * ((d o : Int) : ErrorPoly N))) -
      (matrixPolynomial [(halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int))] :
        ExactMatrix (R i0) N 1 1) 0 0 = reducePoly (R i0) N (smudgeSmall w d) := by
  rw [half_poly, ← map_sub]
  have hsum : (∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) * ((d o : Int) : ErrorPoly N))) =
      ∑ c : Fin 8, ∑ r : Fin 20, Xe (n := N) c *
        (((w r : Int) : ErrorPoly N) * ((d (fin160 (c, r)) : Int) : ErrorPoly N)) := by
    rw [← Equiv.sum_comp fin160, Fintype.sum_prod_type]
    apply Finset.sum_congr rfl
    intro c _
    apply Finset.sum_congr rfl
    intro r _
    have h1 : (fin160 (c, r) : Nat) / 20 = c := by rw [fin160_val]; omega
    have h2 : (⟨(fin160 (c, r) : Nat) % 20, Nat.mod_lt _ (by inst_num)⟩ : Fin 20) = r := by
      ext; simp only [fin160_val]; omega
    rw [h1, h2]
  rw [hsum, ← Finset.sum_sub_distrib]
  unfold smudgeSmall
  rw [map_sum, map_sum]
  apply Finset.sum_congr rfl
  intro c _
  rw [Fin.sum_univ_succ, hw]
  have hsplit : Xe (n := N) c * ((((halfZ (i0 : Int)) : Int) : ErrorPoly N) *
        ((d (fin160 (c, 0)) : Int) : ErrorPoly N)) +
      ∑ r : Fin 19, Xe (n := N) c * (((w r.succ : Int) : ErrorPoly N) *
        ((d (fin160 (c, r.succ)) : Int) : ErrorPoly N)) -
      (((halfZ (i0 : Int)) : Int) : ErrorPoly N) * Xe (n := N) c =
      Xe (n := N) c * (((((halfZ (i0 : Int)) * d (fin160 (c, 0)) - (halfZ (i0 : Int)) : Int) : ErrorPoly N)) +
        ((∑ r : Fin 19, w r.succ * d (fin160 (c, r.succ)) : Int) : ErrorPoly N)) := by
    rw [mul_add, Int.cast_sum, Finset.mul_sum]
    simp only [Int.cast_mul, Int.cast_sub]
    ring
  rw [hsplit, map_mul, map_add, half_digit _ (hd _)]
  rw [← map_add, ← map_mul, Int.cast_add]

theorem unitCol_reduce {i0 : Nat} [Fact (i0 < 8)] {m : Nat} (k : Fin m) :
    unitCol (q := (R i0)) k = reduceMatrix (R i0) N m 1 (unitColI k) := by
  funext i c
  simp only [unitCol, unitColI, reduceMatrix]
  split <;> simp



theorem mask_reduce {i0 : Nat} [Fact (i0 < 8)] {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a : ExactMatrix (R i0) N 1 1} (ha : modulusReduceRuns w4 a) :
    a = fun _ _ ↦ reducePoly (R i0) N sm :=
  modulusReduceRuns_reduce (by inst_num) (by inst_num) hN ha (fun _ _ ↦ sm) (by rw [hw4]; rfl)

theorem one_reduce {i0 : Nat} [Fact (i0 < 8)] {b : ExactMatrix (R i0) N 1 1} (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b) :
    b = 1 := by
  rw [modulusReduceRuns_reduce (by inst_num) (by inst_num) hN hb (fun _ _ ↦ 1) (by
    funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix])]
  funext r c; obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _; simp [reduceMatrix]

/-- The hint: `[s, 1] A + e`. -/
theorem hint_value {i0 : Nat} [Fact (i0 < 8)] {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a b : ExactMatrix (R i0) N 1 1}
    (ha : modulusReduceRuns w4 a) (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b)
    {r : ExactMatrix (R i0) N 1 (1 + 1)} (hr : concatColumns a b r) {A : ExactMatrix (R i0) N (1 + 1) 16}
    {e : ExactMatrix (R i0) N 1 16} (he : gaussianSample (4 / 1) 26 e) :
    ∃ eI : ErrorMatrix N 1 16, CoeffBound eI 26 ∧
      matrixAdd (matrixMul r A) e = maskPart (reducePoly (R i0) N sm) (fun i ↦ rowOf A i) +
        reduceMatrix (R i0) N 1 16 eI := by
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
theorem one_value {i0 : Nat} [Fact (i0 < 8)] {sm : ErrorPoly N} {w4 : ExactMatrix Q0 N 1 1}
    (hw4 : w4 = fun _ _ ↦ reducePoly Q0 N sm) {a b : ExactMatrix (R i0) N 1 1}
    (ha : modulusReduceRuns w4 a) (hb : modulusReduceRuns (1 : ExactMatrix Q0 N 1 1) b)
    {A01 A11 G : ExactMatrix (R i0) N 1 6} {e : ExactMatrix (R i0) N 1 6} (he : gaussianSample (4 / 1) 26 e) :
    ∃ eI : ErrorMatrix N 1 6, CoeffBound eI 26 ∧
      matrixAdd (matrixSub (matrixAdd (matrixMulScalarLeft a A01) (matrixMulScalarLeft b A11))
        (matrixMulScalarLeft b (matrixMulScalarLeft (matrixPolynomial [1]) G))) e =
        maskPart (reducePoly (R i0) N sm) ![A01, A11] - G + reduceMatrix (R i0) N 1 6 eI := by
  obtain ⟨eI, heI, hb'⟩ := gaussian_bound he
  refine ⟨eI, hb', ?_⟩
  rw [mask_reduce hw4 ha, one_reduce hb, heI]
  funext i c
  simp only [matrixAdd, matrixSub, matrixMulScalarLeft, maskPart, Matrix.add_apply, Matrix.sub_apply,
    Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single]
  simp



/-- A row of the preprocessing's combination at column `a`, plus the smudge keys. -/
theorem pre_row {i0 : Nat} [Fact (i0 < 8)] {kg k1 : ExactMatrix (R i0) N 1 6} {gc : ExactMatrix (R i0) N 1 (6 + 6)}
    (hgc : concatColumns kg k1 gc) {D : ExactMatrix (R i0) N (6 + 6) 6} {kx : ExactMatrix (R i0) N 1 4}
    {Y : ExactMatrix (R i0) N 4 6} {kw Rm : ExactMatrix (R i0) N 1 6}
    (hR : Rm = matrixAdd (matrixSub (matrixMul gc D) (matrixMul kx Y)) kw)
    {a b : Int} (ha : a.toNat < 6) {s : ExactMatrix (R i0) N 1 1} (hs : sliceMatrix Rm 0 1 a b s)
    {sm r : ExactMatrix (R i0) N 1 1} (hr : r = matrixAdd s sm) :
    r = (kg * topRows D + k1 * bottomRows D - kx * Y + kw) * unitCol (q := (R i0)) ⟨a.toNat, ha⟩ + sm := by
  rw [hr, slice_col hs ha, hR]
  simp only [matrixAdd, matrixSub, matrixMul, concatColumns_mul hgc]

/-- The rows of a preimage: row `i` of `A`, times the preimage, is target row `i`. -/
theorem target_rows {i0 : Nat} [Fact (i0 < 8)] {A : ExactMatrix (R i0) N (1 + 1) 16} {P : ExactMatrix (R i0) N 16 1}
    {r0 r1 : ExactMatrix (R i0) N 1 1} {T : ExactMatrix (R i0) N (1 + 1) 1} (hT : concatRows r0 r1 T)
    (hP : A * P = T) : ∀ i : Fin 2, rowOf A i * P = ![r0, r1] i := by
  intro i
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  have := congrFun (congrFun hP i) 0
  rw [hT i 0] at this
  fin_cases i <;> simpa [rowOf, Matrix.mul_apply] using this



/-- The preimage equation of a decoder column, with the preimage within its cutoff. -/
theorem preimage_column {i0 : Nat} [Fact (i0 < 8)] {Token : Type} {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix (R i0) N (1 + 1) 16} {Tk : TrapdoorValue (ExactMatrix (R i0) N (1 + 1) 16) Token}
    (hT : trapdoorSample Backend.backend sigma base digits cutoff A Tk)
    {T : ExactMatrix (R i0) N (1 + 1) 1} {cut : Nat} {P : ExactMatrix (R i0) N 16 1}
    (hP : preimageRunsDispatched Backend.backend A Tk T cut P)
    {r0 r1 : ExactMatrix (R i0) N 1 1} (hrows : concatRows r0 r1 T)
    {D : ExactMatrix (R i0) N (6 + 6) 6} {Y : ExactMatrix (R i0) N 4 6} {a b : Int} (ha : a.toNat < 6)
    {kg0 k10 kg1 k11 : ExactMatrix (R i0) N 1 6} {gc0 gc1 : ExactMatrix (R i0) N 1 (6 + 6)}
    (hgc0 : concatColumns kg0 k10 gc0) (hgc1 : concatColumns kg1 k11 gc1)
    {kx0 kx1 : ExactMatrix (R i0) N 1 4} {kw0 kw1 R0 R1 : ExactMatrix (R i0) N 1 6}
    (hR0 : R0 = matrixAdd (matrixSub (matrixMul gc0 D) (matrixMul kx0 Y)) kw0)
    (hR1 : R1 = matrixAdd (matrixSub (matrixMul gc1 D) (matrixMul kx1 Y)) kw1)
    {s0 s1 sm0 sm1 : ExactMatrix (R i0) N 1 1} (hs0 : sliceMatrix R0 0 1 a b s0)
    (hs1 : sliceMatrix R1 0 1 a b s1) (hr0 : r0 = matrixAdd s0 sm0) (hr1 : r1 = matrixAdd s1 sm1)
    {B Kg K₁ Kw Ksm : Fin 2 → _} {Kx : Fin 2 → ExactMatrix (R i0) N 1 4}
    (hB : ∀ i, B i = rowOf A i) (hKg : ∀ i, Kg i = ![kg0, kg1] i) (hK₁ : ∀ i, K₁ i = ![k10, k11] i)
    (hKx : ∀ i, Kx i = ![kx0, kx1] i) (hKw : ∀ i, Kw i = ![kw0, kw1] i)
    (hKsm : ∀ i, Ksm i = ![sm0, sm1] i) :
    (∀ i, B i * P = (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) *
      unitCol (q := (R i0)) ⟨a.toNat, ha⟩ + Ksm i) ∧
    ∃ KI : ErrorMatrix N 16 1, P = reduceMatrix (R i0) N 16 1 KI ∧ CoeffBound KI cut := by
  obtain ⟨hAP, hW⟩ := preimage_sampled hT hP
  have hr := target_rows hrows hAP
  refine ⟨fun i ↦ ?_, hW⟩
  rw [hB i, hr i, hKg i, hK₁ i, hKx i, hKw i, hKsm i]
  fin_cases i
  · exact pre_row hgc0 hR0 ha hs0 hr0
  · exact pre_row hgc1 hR1 ha hs1 hr1

/-- The digits `D` of `Q_q A'` that two stages compute from one hashed `A` agree. -/
theorem d_unique {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {tag : Blob} {comps : List HashTagComponent}
    {key key' : ByteArray} (hk : key = key') {a a' : ExactMatrix Q0 N (1 + 1) 6}
    (ha : hashSample model tag comps key a) (ha' : hashSample model tag comps key' a')
    {b b' : ExactMatrix (R i0) N (1 + 1) 6} (hb : modulusReduceRuns a b) (hb' : modulusReduceRuns a' b')
    {c : ExactMatrix (R i0) N 1 1} {z z' : ExactMatrix (R i0) N (1 + 1) 6} (hz : z = matrixMulScalarLeft c b)
    (hz' : z' = matrixMulScalarLeft c b') {Dz Dz' : ExactMatrix (R i0) N (6 + 6) 6}
    (hd : gadgetDecomposeRuns Backend.backend 32768 6 z Dz)
    (hd' : gadgetDecomposeRuns Backend.backend 32768 6 z' Dz') {D D' : ExactMatrix (R i0) N (6 + 6) 6}
    (hD : compactCenteredRebaseRuns 16384 Dz D) (hD' : compactCenteredRebaseRuns 16384 Dz' D') :
    D = D' := by
  subst hk
  obtain rfl : a = a' := (hashSample_eq ha).trans (hashSample_eq ha').symm
  obtain rfl := modulusReduceRuns_unique hb hb'
  subst hz hz'
  exact dec_rebase_unique hd hd' hD hD'

set_option maxRecDepth 65536 in
/-- A smudge minus the half constant is small. -/
theorem smudge_glue {i0 : Nat} [Fact (i0 < 8)] (w : Fin 20 → Int) (hw : w 0 = (halfZ (i0 : Int))) (hwb : ∑ r : Fin 19, (w r.succ).natAbs ≤ 274877906942)
    (z : Fin 160 → Nat) (sp : ErrorPoly N) {H : ExactMatrix (R i0) N 1 1}
    (hH : H = matrixPolynomial [(halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int)), (halfZ (i0 : Int))]) :
    ∃ M : ErrorPoly N, CoeffBound (fun _ _ ↦ M : ErrorMatrix N 1 1) (8 * (2 + 3 * 274877906942)) ∧
      reducePoly (R i0) N ((∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by inst_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
            Part_I.smudgeTable ⟨z o % 4, Nat.mod_lt _ (by inst_num)⟩)))) * unitPair sp 1) - H 0 0 =
        reducePoly (R i0) N M := by
  set d : Fin 160 → Int := fun o ↦ (-1 : Int) ^ (z o / 4) * (![1, 1, 3, 3] ⟨z o % 4, Nat.mod_lt _ (by inst_num)⟩ : Int)
    with hd
  have hv : ∀ f : Fin 4, ((![1, 1, 3, 3] f : Int) % 2 = 1) ∧ |(![1, 1, 3, 3] f : Int)| ≤ 3 := by decide
  have hodd : ∀ o, d o % 2 = 1 := by
    intro o
    obtain ⟨h1, -⟩ := hv ⟨z o % 4, Nat.mod_lt _ (by inst_num)⟩
    rcases neg_one_pow_eq_or ℤ (z o / 4) with h | h <;> simp only [hd, h] <;> omega
  have habs : ∀ o, |d o| ≤ 3 := by
    intro o
    obtain ⟨-, h2⟩ := hv ⟨z o % 4, Nat.mod_lt _ (by inst_num)⟩
    rcases neg_one_pow_eq_or ℤ (z o / 4) with h | h <;> simp only [hd, h] <;> simpa [abs_neg] using h2
  refine ⟨smudgeSmall w d, ?_, ?_⟩
  · refine (smudgeSmall_bound w d (A := 2 + 3 * 274877906942) fun c ↦ ?_)
    exact (digit_sum_bound w d habs c).trans (by omega)
  · rw [← smudge_minus_half w hw d hodd, hH, show unitPair sp 1 = 1 from rfl, mul_one]
    congr 2
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    simp only [hd, Part_I.smudgeTable, Int.cast_mul, one_mul]

/-- An entry of a fresh key: the online key, switched down and times the digits of `Q_q`, is the
preprocessing's. -/
theorem key_entry {i0 : Nat} [Fact (i0 < 8)] {q : Nat} {a b : ExactMatrix q N 1 6} (hab : a = b) {y : ExactMatrix (R i0) N 1 6}
    (hy : blockModSwitchRuns [q0, q1, q2, q3] 1 b y) {d d' : ExactMatrix (R i0) N 6 1} (hd : d = d') :
    ((blockModSwitch [q0, q1, q2, q3] 1 a : ExactMatrix (R i0) N 1 6) * d) 0 0 = (y * d') 0 0 := by
  rw [hab, hd, blockModSwitchRuns_eq hy]

/-- A row whose entries are those of six columns side by side. -/
theorem row_of_entries {i0 : Nat} [Fact (i0 < 8)] {u : Fin 6 → ExactPoly (R i0) N} {x0 x1 x2 x3 x4 x5 : ExactMatrix (R i0) N 1 1}
    {z1 : ExactMatrix (R i0) N 1 2} {z2 : ExactMatrix (R i0) N 1 3} {z3 : ExactMatrix (R i0) N 1 4}
    {z4 : ExactMatrix (R i0) N 1 5} {z : ExactMatrix (R i0) N 1 6} (c1 : concatColumns x0 x1 z1)
    (c2 : concatColumns z1 x2 z2) (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4)
    (c5 : concatColumns z4 x5 z) (h0 : u 0 = x0 0 0) (h1 : u 1 = x1 0 0) (h2 : u 2 = x2 0 0)
    (h3 : u 3 = x3 0 0) (h4 : u 4 = x4 0 0) (h5 : u 5 = x5 0 0) : rowMat u = z := by
  rw [rowMat_concat c1 c2 c3 c4 c5]
  congr 1
  funext k
  fin_cases k
  · exact h0
  · exact h1
  · exact h2
  · exact h3
  · exact h4
  · exact h5

set_option maxHeartbeats 4000000 in
/-- A decoder column is `Q_q y` plus a small integer polynomial. -/
theorem column_noiseQ {i0 : Nat} [Fact (i0 < 8)] {Qq : Int} {y : ExactMatrix (R i0) N 1 6} (k : Fin 6) {M : ErrorMatrix N 1 1} {BM : Nat}
    (hM : CoeffBound M BM) {ebI : ErrorMatrix N 1 16} {Beb : Nat} (heb : CoeffBound ebI Beb)
    {KI : ErrorMatrix N 16 1} {Bpre : Nat} (hK : CoeffBound KI Bpre)
    {egI e1I ewI : ErrorMatrix N 1 6} {Beg Be1 Bew : Nat} (heg : CoeffBound egI Beg)
    (he1 : CoeffBound e1I Be1) (hew : CoeffBound ewI Bew)
    {X0I X1I : ErrorMatrix N 6 6} (hX0 : CoeffBound X0I 16384) (hX1 : CoeffBound X1I 16384)
    {exI : ErrorMatrix N 1 4} {Bex : Nat} (hex : CoeffBound exI Bex)
    {YI : ErrorMatrix N 4 6} (hY : CoeffBound YI 16384) {esmI : ErrorMatrix N 1 1} {Besm : Nat}
    (hesm : CoeffBound esmI Besm) {out : ExactMatrix (R i0) N 1 1}
    (hout : out = ((Qq : Int) : ExactPoly (R i0) N) • (y * unitCol (q := (R i0)) k) + reduceMatrix (R i0) N 1 1 M +
      (reduceMatrix (R i0) N 1 16 ebI * reduceMatrix (R i0) N 16 1 KI -
        (reduceMatrix (R i0) N 1 6 egI * reduceMatrix (R i0) N 6 6 X0I +
          reduceMatrix (R i0) N 1 6 e1I * reduceMatrix (R i0) N 6 6 X1I -
          reduceMatrix (R i0) N 1 4 exI * reduceMatrix (R i0) N 4 6 YI + reduceMatrix (R i0) N 1 6 ewI) *
          unitCol (q := (R i0)) k - reduceMatrix (R i0) N 1 1 esmI)) :
    ∃ ν : ErrorMatrix N 1 1, CoeffBound ν (BM + 16 * N * Beb * Bpre +
        (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 + 4 * N * Bex * 16384 + Bew) + Besm) ∧
      out 0 0 = ((Qq : Int) : ExactPoly (R i0) N) * y 0 k + reducePoly (R i0) N (ν 0 0) := by
  set ν : ErrorMatrix N 1 1 := M + ebI * KI - (egI * X0I + e1I * X1I - exI * YI + ewI) * unitColI k -
    esmI with hν
  refine ⟨ν, coeffBound_sub (coeffBound_sub (coeffBound_add hM (coeffBound_mul hN heb hK)) ?_) hesm, ?_⟩
  · exact coeffBound_col (coeffBound_add (coeffBound_sub (coeffBound_add
      (coeffBound_mul hN heg hX0) (coeffBound_mul hN he1 hX1)) (coeffBound_mul hN hex hY)) hew) k
  · have hr : reduceMatrix (R i0) N 1 1 ν = reduceMatrix (R i0) N 1 1 M +
        (reduceMatrix (R i0) N 1 16 ebI * reduceMatrix (R i0) N 16 1 KI -
          (reduceMatrix (R i0) N 1 6 egI * reduceMatrix (R i0) N 6 6 X0I +
            reduceMatrix (R i0) N 1 6 e1I * reduceMatrix (R i0) N 6 6 X1I -
            reduceMatrix (R i0) N 1 4 exI * reduceMatrix (R i0) N 4 6 YI + reduceMatrix (R i0) N 1 6 ewI) *
            unitCol (q := (R i0)) k - reduceMatrix (R i0) N 1 1 esmI) := by
      rw [hν, unitCol_reduce]
      simp only [reduceMatrix_add, reduceMatrix_sub, reduceMatrix_mul]
      abel
    rw [hout, add_assoc, ← hr, Matrix.add_apply, Matrix.smul_apply, mul_unitCol, smul_eq_mul]
    rfl

/-- Six `1 × 1` rows of target `t_k Q_q` side by side: a row of target `Q_q t`. -/
theorem six_rowsQ {i0 : Nat} [Fact (i0 < 8)] {Qq : Int} {s : ExactPoly (R i0) N} {z : ExactMatrix (R i0) N 1 6} {x : Fin 6 → ExactMatrix (R i0) N 1 1}
    (hz : ∀ k, z 0 k = x k 0 0) {K : Fin 6 → Fin 2 → ExactMatrix (R i0) N 1 1} {t : Fin 6 → ErrorPoly N}
    {e : Fin 6 → ErrorMatrix N 1 1}
    (hx : ∀ k, x k = maskPart s (K k) - reducePoly (R i0) N (t k) •
      (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 (e k)) :
    z = maskPart s (fun i ↦ rowMat (fun k ↦ K k i 0 0)) - ((Qq : Int) : ExactPoly (R i0) N) • rowMat (fun k ↦ reducePoly (R i0) N (t k)) +
      reduceMatrix (R i0) N 1 6 (fun _ k ↦ e k 0 0) := by
  funext r k
  obtain rfl : r = 0 := Subsingleton.elim _ _
  rw [hz, hx]
  simp [maskPart, matrixPolynomial_single, reduceMatrix, rowMat, mul_comm]

/-- A row of target `t G` times the digits of `Q_q` has target `Q_q t`. -/
theorem row_mul_qqQ {i0 : Nat} [Fact (i0 < 8)] {Qq : Int} {sm : ErrorPoly N} {K : Fin 2 → ExactMatrix (R i0) N 1 6} {t : ErrorPoly N}
    {gq : ExactMatrix (R i0) N 1 6} (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {e : ErrorMatrix N 1 6} {B : Nat} (he : CoeffBound e B) {c : ExactMatrix (R i0) N 1 6}
    (hc : c = maskPart (reducePoly (R i0) N sm) K - reducePoly (R i0) N t • gq + reduceMatrix (R i0) N 1 6 e)
    {Dq' Dq : ExactMatrix (R i0) N 6 1}
    (hD' : gadgetDecomposeRuns Backend.backend 32768 6
      (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) Dq')
    (hD : compactCenteredRebaseRuns 16384 Dq' Dq) :
    ∃ e' : ErrorMatrix N 1 1, CoeffBound e' (6 * N * B * 16384) ∧
      c * Dq = maskPart (reducePoly (R i0) N sm) (fun k ↦ K k * Dq) -
        reducePoly (R i0) N t • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) +
          reduceMatrix (R i0) N 1 1 e' := by
  obtain ⟨L, hL, hLx, -, -⟩ := layoutR i0 hi0
  obtain ⟨W, hW, hWb⟩ := hD.2.2.2
  have hg := gadget_mul_decompose_one (by inst_num) L hL hLx hgq hD'
  rw [← compactRebase_self (by inst_num) hD] at hg
  refine ⟨e * W, coeffBound_mul hN he hWb, ?_⟩
  rw [hc, Matrix.add_mul, Matrix.sub_mul, maskPart_mul, Matrix.smul_mul, hg, hW,
    ← reduceMatrix_mul]

/-- The fresh columns side by side: a row of target `Q_q e'`. -/
theorem fresh_glueQ {i0 : Nat} [Fact (i0 < 8)] {Qq : Int} {sm : ErrorPoly N} {B : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix (R i0) N 1 1}
    {K0 K1 K2 K3 K4 K5 : Fin 2 → ExactMatrix (R i0) N 1 1} {t0 t1 t2 t3 t4 t5 : ErrorPoly N}
    (h0 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x0 = maskPart (reducePoly (R i0) N sm) K0 -
      reducePoly (R i0) N t0 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    (h1 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x1 = maskPart (reducePoly (R i0) N sm) K1 -
      reducePoly (R i0) N t1 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    (h2 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x2 = maskPart (reducePoly (R i0) N sm) K2 -
      reducePoly (R i0) N t2 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    (h3 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x3 = maskPart (reducePoly (R i0) N sm) K3 -
      reducePoly (R i0) N t3 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    (h4 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x4 = maskPart (reducePoly (R i0) N sm) K4 -
      reducePoly (R i0) N t4 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    (h5 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x5 = maskPart (reducePoly (R i0) N sm) K5 -
      reducePoly (R i0) N t5 • (matrixPolynomial [Qq] : ExactMatrix (R i0) N 1 1) + reduceMatrix (R i0) N 1 1 e)
    {z1 : ExactMatrix (R i0) N 1 2} {z2 : ExactMatrix (R i0) N 1 3} {z3 : ExactMatrix (R i0) N 1 4}
    {z4 : ExactMatrix (R i0) N 1 5} {z : ExactMatrix (R i0) N 1 6} (c1 : concatColumns x0 x1 z1)
    (c2 : concatColumns z1 x2 z2) (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4)
    (c5 : concatColumns z4 x5 z) :
    ∃ ewI : ErrorMatrix N 1 6, CoeffBound ewI B ∧
      z = maskPart (reducePoly (R i0) N sm) (fun i ↦ rowMat (fun k ↦ ![K0, K1, K2, K3, K4, K5] k i 0 0)) -
        ((Qq : Int) : ExactPoly (R i0) N) • rowMat (fun k ↦ reducePoly (R i0) N (![t0, t1, t2, t3, t4, t5] k)) +
          reduceMatrix (R i0) N 1 6 ewI := by
  obtain ⟨e0, b0, h0⟩ := h0
  obtain ⟨e1, b1, h1⟩ := h1
  obtain ⟨e2, b2, h2⟩ := h2
  obtain ⟨e3, b3, h3⟩ := h3
  obtain ⟨e4, b4, h4⟩ := h4
  obtain ⟨e5, b5, h5⟩ := h5
  refine ⟨fun _ k ↦ ![e0, e1, e2, e3, e4, e5] k 0 0, fun r k i ↦ ?_, ?_⟩
  · fin_cases k
    · exact b0 0 0 i
    · exact b1 0 0 i
    · exact b2 0 0 i
    · exact b3 0 0 i
    · exact b4 0 0 i
    · exact b5 0 0 i
  · refine six_rowsQ (x := ![x0, x1, x2, x3, x4, x5]) (fun k ↦ ?_) (K := ![K0, K1, K2, K3, K4, K5])
      (t := ![t0, t1, t2, t3, t4, t5]) (e := ![e0, e1, e2, e3, e4, e5]) (fun k ↦ ?_)
    · exact (concat_six c1 c2 c3 c4 c5 k).trans (by fin_cases k <;> rfl)
    · fin_cases k
      · exact h0
      · exact h1
      · exact h2
      · exact h3
      · exact h4
      · exact h5

/-- The action digits `Y` that two stages compute from the gadget agree. -/
theorem y_uniqueC {i0 : Nat} [Fact (i0 < 8)] {qc Mx : Nat} {g g' : ExactMatrix (R i0) N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g)
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') {r r' : ExactMatrix Q0 N 1 6}
    (hr : centeredRebaseRuns g r) (hr' : centeredRebaseRuns g' r') {m m' : ExactMatrix qc N 1 6}
    (hm : modulusReduceRuns r m) (hm' : modulusReduceRuns r' m') {c3 c1 : ExactMatrix qc N 1 1}
    {s s' t t' : ExactMatrix qc N 1 6} (hs : s = matrixMulScalarRight m c3)
    (hs' : s' = matrixMulScalarRight m' c3) (ht : t = matrixMulScalarRight s c1)
    (ht' : t' = matrixMulScalarRight s' c1) {u u' : ExactMatrix Mx N 1 6}
    (hu : centeredRebaseRuns t u) (hu' : centeredRebaseRuns t' u') {c : ExactMatrix Mx N 1 1}
    {v v' : ExactMatrix Mx N 1 6} (hv : v = matrixMulScalarRight u c)
    (hv' : v' = matrixMulScalarRight u' c) {Yz Yz' : ExactMatrix Mx N 4 6}
    (hy : gadgetDecomposeRuns Backend.backend 32768 4 v Yz)
    (hy' : gadgetDecomposeRuns Backend.backend 32768 4 v' Yz') {Y Y' : ExactMatrix (R i0) N 4 6}
    (hY : compactCenteredRebaseRuns 16384 Yz Y) (hY' : compactCenteredRebaseRuns 16384 Yz' Y') :
    Y = Y' := by
  obtain rfl := gadgetMatrixRuns_unique hg hg'
  obtain rfl := centeredRebaseRuns_unique hr hr'
  obtain rfl := modulusReduceRuns_unique hm hm'
  subst hs hs' ht ht'
  obtain rfl := centeredRebaseRuns_unique hu hu'
  subst hv hv'
  exact dec_rebase_unique hy hy' hY hY'

end RefreshTopProof.Part_I
