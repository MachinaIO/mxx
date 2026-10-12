import TopTarget
import TopHint
import TopColumn
import Part_I_Smudge
import ExponentFunctional

/-!
# The preimage equation of a decoder column

The preprocessing samples a trapdoor `A` and solves `A P = [r₀; r₁]`, where row `i` of the target
is its combination of keys at the column plus the smudge key of row `i`. The keys the online
evaluation sees are those keys (by the key links), so `B i P` is the online combination's key at
the column plus the smudge key, for the hint keys `B i = rowOf A i`.
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh

/-- The preimage equation of a decoder column, with the preimage within its cutoff. -/
theorem preimage_column {Token : Type} {sigma : Rat} {base digits cutoff : Int}
    {A : ExactMatrix Q1 N (1 + 1) 16} {Tk : TrapdoorValue (ExactMatrix Q1 N (1 + 1) 16) Token}
    (hT : trapdoorSample Backend.backend sigma base digits cutoff A Tk)
    {T : ExactMatrix Q1 N (1 + 1) 1} {cut : Nat} {P : ExactMatrix Q1 N 16 1}
    (hP : preimageRunsDispatched Backend.backend A Tk T cut P)
    {r0 r1 : ExactMatrix Q1 N 1 1} (hrows : concatRows r0 r1 T)
    {D : ExactMatrix Q1 N (6 + 6) 6} {Y : ExactMatrix Q1 N 4 6} {a b : Int} (ha : a.toNat < 6)
    {kg0 k10 kg1 k11 : ExactMatrix Q1 N 1 6} {gc0 gc1 : ExactMatrix Q1 N 1 (6 + 6)}
    (hgc0 : concatColumns kg0 k10 gc0) (hgc1 : concatColumns kg1 k11 gc1)
    {kx0 kx1 : ExactMatrix Q1 N 1 4} {kw0 kw1 R0 R1 : ExactMatrix Q1 N 1 6}
    (hR0 : R0 = matrixAdd (matrixSub (matrixMul gc0 D) (matrixMul kx0 Y)) kw0)
    (hR1 : R1 = matrixAdd (matrixSub (matrixMul gc1 D) (matrixMul kx1 Y)) kw1)
    {s0 s1 sm0 sm1 : ExactMatrix Q1 N 1 1} (hs0 : sliceMatrix R0 0 1 a b s0)
    (hs1 : sliceMatrix R1 0 1 a b s1) (hr0 : r0 = matrixAdd s0 sm0) (hr1 : r1 = matrixAdd s1 sm1)
    {B Kg K₁ Kw Ksm : Fin 2 → _} {Kx : Fin 2 → ExactMatrix Q1 N 1 4}
    (hB : ∀ i, B i = rowOf A i) (hKg : ∀ i, Kg i = ![kg0, kg1] i) (hK₁ : ∀ i, K₁ i = ![k10, k11] i)
    (hKx : ∀ i, Kx i = ![kx0, kx1] i) (hKw : ∀ i, Kw i = ![kw0, kw1] i)
    (hKsm : ∀ i, Ksm i = ![sm0, sm1] i) :
    (∀ i, B i * P = (Kg i * topRows D + K₁ i * bottomRows D - Kx i * Y + Kw i) *
      unitCol (q := Q1) ⟨a.toNat, ha⟩ + Ksm i) ∧
    ∃ KI : ErrorMatrix N 16 1, P = reduceMatrix Q1 N 16 1 KI ∧ CoeffBound KI cut := by
  obtain ⟨hAP, hW⟩ := preimage_sampled hT hP
  have hr := target_rows hrows hAP
  refine ⟨fun i ↦ ?_, hW⟩
  rw [hB i, hr i, hKg i, hK₁ i, hKx i, hKw i, hKsm i]
  fin_cases i
  · exact pre_row hgc0 hR0 ha hs0 hr0
  · exact pre_row hgc1 hR1 ha hs1 hr1

/-- Two decompositions of one matrix, re-encoded compactly, agree. -/
theorem dec_rebase_unique {q p rows columns inner : Nat} {base digits : Int} {bound : Nat}
    {X : ExactMatrix q N rows columns} {D D' : ExactMatrix q N inner columns}
    (hd : gadgetDecomposeRuns Backend.backend base digits X D)
    (hd' : gadgetDecomposeRuns Backend.backend base digits X D')
    {Y Y' : ExactMatrix p N inner columns} (hY : compactCenteredRebaseRuns bound D Y)
    (hY' : compactCenteredRebaseRuns bound D' Y') : Y = Y' := by
  obtain rfl := gadgetDecomposeRuns_unique hd hd'
  exact compactCenteredRebaseRuns_unique hY hY'

/-- The digits `D` of `Q_q A'` that two stages compute from one hashed `A` agree. -/
theorem d_unique {model : HashModel} {tag : Blob} {comps : List HashTagComponent}
    {key key' : ByteArray} (hk : key = key') {a a' : ExactMatrix Q0 N (1 + 1) 6}
    (ha : hashSample model tag comps key a) (ha' : hashSample model tag comps key' a')
    {b b' : ExactMatrix Q1 N (1 + 1) 6} (hb : modulusReduceRuns a b) (hb' : modulusReduceRuns a' b')
    {c : ExactMatrix Q1 N 1 1} {z z' : ExactMatrix Q1 N (1 + 1) 6} (hz : z = matrixMulScalarLeft c b)
    (hz' : z' = matrixMulScalarLeft c b') {Dz Dz' : ExactMatrix Q1 N (6 + 6) 6}
    (hd : gadgetDecomposeRuns Backend.backend 32768 6 z Dz)
    (hd' : gadgetDecomposeRuns Backend.backend 32768 6 z' Dz') {D D' : ExactMatrix Q1 N (6 + 6) 6}
    (hD : compactCenteredRebaseRuns 16384 Dz D) (hD' : compactCenteredRebaseRuns 16384 Dz' D') :
    D = D' := by
  subst hk
  obtain rfl : a = a' := (hashSample_eq ha).trans (hashSample_eq ha').symm
  obtain rfl := modulusReduceRuns_unique hb hb'
  subst hz hz'
  exact dec_rebase_unique hd hd' hD hD'

/-- The action digits `Y` that two stages compute from the gadget agree. -/
theorem y_unique {g g' : ExactMatrix Q1 N 1 6} (hg : gadgetMatrixRuns Backend.backend 32768 6 g)
    (hg' : gadgetMatrixRuns Backend.backend 32768 6 g') {r r' : ExactMatrix Q0 N 1 6}
    (hr : centeredRebaseRuns g r) (hr' : centeredRebaseRuns g' r') {m m' : ExactMatrix q0 N 1 6}
    (hm : modulusReduceRuns r m) (hm' : modulusReduceRuns r' m') {c3 c1 : ExactMatrix q0 N 1 1}
    {s s' t t' : ExactMatrix q0 N 1 6} (hs : s = matrixMulScalarRight m c3)
    (hs' : s' = matrixMulScalarRight m' c3) (ht : t = matrixMulScalarRight s c1)
    (ht' : t' = matrixMulScalarRight s' c1) {u u' : ExactMatrix M01 N 1 6}
    (hu : centeredRebaseRuns t u) (hu' : centeredRebaseRuns t' u') {c : ExactMatrix M01 N 1 1}
    {v v' : ExactMatrix M01 N 1 6} (hv : v = matrixMulScalarRight u c)
    (hv' : v' = matrixMulScalarRight u' c) {Yz Yz' : ExactMatrix M01 N 4 6}
    (hy : gadgetDecomposeRuns Backend.backend 32768 4 v Yz)
    (hy' : gadgetDecomposeRuns Backend.backend 32768 4 v' Yz') {Y Y' : ExactMatrix Q1 N 4 6}
    (hY : compactCenteredRebaseRuns 16384 Yz Y) (hY' : compactCenteredRebaseRuns 16384 Yz' Y') :
    Y = Y' := by
  obtain rfl := gadgetMatrixRuns_unique hg hg'
  obtain rfl := centeredRebaseRuns_unique hr hr'
  obtain rfl := modulusReduceRuns_unique hm hm'
  subst hs hs' ht ht'
  obtain rfl := centeredRebaseRuns_unique hu hu'
  subst hv hv'
  exact dec_rebase_unique hy hy' hY hY'

/-- Six `1 × 1` columns side by side are the row of their entries. -/
theorem rowMat_concat {q : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix q N 1 1} {z1 : ExactMatrix q N 1 2}
    {z2 : ExactMatrix q N 1 3} {z3 : ExactMatrix q N 1 4} {z4 : ExactMatrix q N 1 5}
    {z : ExactMatrix q N 1 6} (c1 : concatColumns x0 x1 z1) (c2 : concatColumns z1 x2 z2)
    (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4) (c5 : concatColumns z4 x5 z) :
    z = rowMat (fun k ↦ ![x0 0 0, x1 0 0, x2 0 0, x3 0 0, x4 0 0, x5 0 0] k) := by
  funext r k
  obtain rfl : r = 0 := Subsingleton.elim _ _
  exact concat_six c1 c2 c3 c4 c5 k

/-- The fresh columns side by side: a row of target `Q_q e'`. -/
theorem fresh_glue {sm : ErrorPoly N} {B : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix Q1 N 1 1}
    {K0 K1 K2 K3 K4 K5 : Fin 2 → ExactMatrix Q1 N 1 1} {t0 t1 t2 t3 t4 t5 : ErrorPoly N}
    (h0 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x0 = maskPart (reducePoly Q1 N sm) K0 -
      reducePoly Q1 N t0 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h1 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x1 = maskPart (reducePoly Q1 N sm) K1 -
      reducePoly Q1 N t1 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h2 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x2 = maskPart (reducePoly Q1 N sm) K2 -
      reducePoly Q1 N t2 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h3 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x3 = maskPart (reducePoly Q1 N sm) K3 -
      reducePoly Q1 N t3 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h4 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x4 = maskPart (reducePoly Q1 N sm) K4 -
      reducePoly Q1 N t4 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    (h5 : ∃ e : ErrorMatrix N 1 1, CoeffBound e B ∧ x5 = maskPart (reducePoly Q1 N sm) K5 -
      reducePoly Q1 N t5 • (matrixPolynomial [1152919668508929361] : ExactMatrix Q1 N 1 1) + reduceMatrix Q1 N 1 1 e)
    {z1 : ExactMatrix Q1 N 1 2} {z2 : ExactMatrix Q1 N 1 3} {z3 : ExactMatrix Q1 N 1 4}
    {z4 : ExactMatrix Q1 N 1 5} {z : ExactMatrix Q1 N 1 6} (c1 : concatColumns x0 x1 z1)
    (c2 : concatColumns z1 x2 z2) (c3 : concatColumns z2 x3 z3) (c4 : concatColumns z3 x4 z4)
    (c5 : concatColumns z4 x5 z) :
    ∃ ewI : ErrorMatrix N 1 6, CoeffBound ewI B ∧
      z = maskPart (reducePoly Q1 N sm) (fun i ↦ rowMat (fun k ↦ ![K0, K1, K2, K3, K4, K5] k i 0 0)) -
        QqPoly • rowMat (fun k ↦ reducePoly Q1 N (![t0, t1, t2, t3, t4, t5] k)) +
          reduceMatrix Q1 N 1 6 ewI := by
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
  · refine six_rows (x := ![x0, x1, x2, x3, x4, x5]) (fun k ↦ ?_) (K := ![K0, K1, K2, K3, K4, K5])
      (t := ![t0, t1, t2, t3, t4, t5]) (e := ![e0, e1, e2, e3, e4, e5]) (fun k ↦ ?_)
    · exact (concat_six c1 c2 c3 c4 c5 k).trans (by fin_cases k <;> rfl)
    · fin_cases k
      · exact h0
      · exact h1
      · exact h2
      · exact h3
      · exact h4
      · exact h5

set_option maxRecDepth 65536 in
/-- A smudge minus the half constant is small. -/
theorem smudge_glue (w : Fin 20 → Int) (hw : w 0 = halfQ1) (hwb : ∑ r : Fin 19, (w r.succ).natAbs ≤ 274877906942)
    (z : Fin 160 → Nat) (sp : ErrorPoly N) {H : ExactMatrix Q1 N 1 1}
    (hH : H = matrixPolynomial [halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1, halfQ1]) :
    ∃ M : ErrorPoly N, CoeffBound (fun _ _ ↦ M : ErrorMatrix N 1 1) (8 * (2 + 3 * 274877906942)) ∧
      reducePoly Q1 N ((∑ o : Fin 160, Xe (n := N) (o / 20) *
        (((w ⟨o % 20, Nat.mod_lt _ (by norm_num)⟩ : Int) : ErrorPoly N) *
          ((1 : ErrorPoly N) * ((((-1 : Int) ^ (z o / 4) : Int) : ErrorPoly N) *
            Part_I.smudgeTable ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩)))) * unitPair sp 1) - H 0 0 =
        reducePoly Q1 N M := by
  set d : Fin 160 → Int := fun o ↦ (-1 : Int) ^ (z o / 4) * (![1, 1, 3, 3] ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩ : Int)
    with hd
  have hv : ∀ f : Fin 4, ((![1, 1, 3, 3] f : Int) % 2 = 1) ∧ |(![1, 1, 3, 3] f : Int)| ≤ 3 := by decide
  have hodd : ∀ o, d o % 2 = 1 := by
    intro o
    obtain ⟨h1, -⟩ := hv ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩
    rcases neg_one_pow_eq_or ℤ (z o / 4) with h | h <;> simp only [hd, h] <;> omega
  have habs : ∀ o, |d o| ≤ 3 := by
    intro o
    obtain ⟨-, h2⟩ := hv ⟨z o % 4, Nat.mod_lt _ (by norm_num)⟩
    rcases neg_one_pow_eq_or ℤ (z o / 4) with h | h <;> simp only [hd, h] <;> simpa [abs_neg] using h2
  refine ⟨smudgeSmall w d, ?_, ?_⟩
  · refine (smudgeSmall_bound w d (A := 2 + 3 * 274877906942) fun c ↦ ?_)
    exact (digit_sum_bound w d habs c).trans (by omega)
  · rw [← smudge_minus_half w hw d hodd, hH, show unitPair sp 1 = 1 from rfl, mul_one]
    congr 2
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    simp only [hd, Part_I.smudgeTable, Int.cast_mul, one_mul]

/-- A row with target `(a b) s` has target `a (b s)`. -/
theorem row_assoc {q m : Nat} {c M E : ExactMatrix q N 1 m} {a b s : ErrorPoly N}
    {g : ExactMatrix q N 1 m} (h : c = M - reducePoly q N (a * b * s) • g + E) :
    c = M - reducePoly q N (a * (b * s)) • g + E := by
  rw [h, mul_assoc]

/-- An entry of a fresh key: the online key, switched down and times the digits of `Q_q`, is the
preprocessing's. -/
theorem key_entry {q : Nat} {a b : ExactMatrix q N 1 6} (hab : a = b) {y : ExactMatrix Q1 N 1 6}
    (hy : blockModSwitchRuns [q0, q1, q2, q3] 1 b y) {d d' : ExactMatrix Q1 N 6 1} (hd : d = d') :
    ((blockModSwitch [q0, q1, q2, q3] 1 a : ExactMatrix Q1 N 1 6) * d) 0 0 = (y * d') 0 0 := by
  rw [hab, hd, blockModSwitchRuns_eq hy]

/-- A row whose entries are those of six columns side by side. -/
theorem row_of_entries {u : Fin 6 → ExactPoly Q1 N} {x0 x1 x2 x3 x4 x5 : ExactMatrix Q1 N 1 1}
    {z1 : ExactMatrix Q1 N 1 2} {z2 : ExactMatrix Q1 N 1 3} {z3 : ExactMatrix Q1 N 1 4}
    {z4 : ExactMatrix Q1 N 1 5} {z : ExactMatrix Q1 N 1 6} (c1 : concatColumns x0 x1 z1)
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

end RefreshTopProof
