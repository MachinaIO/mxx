import ExponentDecomp

/-!
# The pairing with the error leaving a block of cells

A block of cells multiplies the aligned members `va i` by the public digits `Ds` of the state key,
adds the state error `es` times `α i`, multiplies by the private digits `Dr i j` of the RHS of cell
`i`, and ModDowns by `P`: the error leaving the block is
`Σ_i (cellNum i j / P - c i eB i j)`. A weight `V` on it splits into the pairings of the aligned
members with the alignment weights, of the state error with the state weight, and of the residues
and RHS errors with `V` itself. An alignment numerator splits the same way.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

section CellBlock

variable {C mA ms mo : Nat}

/-- The numerator of cell `i`, component `j`. -/
noncomputable def cellNum (va : Fin C → Fin 2 → ErrorMatrix n 1 mA) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    (α : Fin C → ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 ms)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (sm : ErrorPoly n)
    (X M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo) (i : Fin C) (j : Fin 2) : ErrorMatrix n 1 mo :=
  pairMul (fun j' ↦ pairMul (va i) (Ds j') + α i • es j') (Dr i j) - X i j + sm • M0 i j + M1 i j

/-- The weight on the aligned member `k` of cell `i`, before the division by `P`. -/
noncomputable def alignWeight (V : Fin 2 → Weight n 1 mo) (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo)
    (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms) (i : Fin C) (k : Fin 2) : Weight n 1 mA :=
  ∑ j, ∑ j', pullMul (pullMul (V j) (halfRows (Dr i j) j')) (halfRows (Ds j') k)

/-- The weight on the state error, before the division by `P`. -/
noncomputable def stateWeight (V : Fin 2 → Weight n 1 mo) (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo)
    (α : Fin C → ErrorPoly n) (j' : Fin 2) : Weight n 1 ms :=
  ∑ i, ∑ j, pullSmul (pullMul (V j) (halfRows (Dr i j) j')) (α i)

theorem pairing_cellNum (hn : 0 < n) (V : Weight n 1 mo) (va : Fin C → Fin 2 → ErrorMatrix n 1 mA)
    (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms) (α : Fin C → ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 ms)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (sm : ErrorPoly n)
    (X M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo) (i : Fin C) (j : Fin 2) :
    pairing V (cellNum va Ds α es Dr sm X M0 M1 i j) =
      (∑ j', ∑ k, pairing (pullMul (pullMul V (halfRows (Dr i j) j')) (halfRows (Ds j') k)) (va i k)) +
        (∑ j', pairing (pullSmul (pullMul V (halfRows (Dr i j) j')) (α i)) (es j')) -
        pairing V (X i j) + pairing (pullSmul V sm) (M0 i j) + pairing V (M1 i j) :=
  pairing_cellNumer hn V (va i) Ds (α i) es (Dr i j) sm (X i j) (M0 i j) (M1 i j)

/-- The pairing with the error leaving a block of cells. -/
theorem pairing2_cellBlock (hn : 0 < n) (V : Fin 2 → Weight n 1 mo) {P : ℤ} (hP : P ≠ 0)
    (va : Fin C → Fin 2 → ErrorMatrix n 1 mA) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    (α : Fin C → ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 ms)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (sm : ErrorPoly n)
    (X M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo)
    (hdiv : ∀ i j r c, cellNum va Ds α es Dr sm X M0 M1 i j r c =
      (P : ErrorPoly n) * divInt P (cellNum va Ds α es Dr sm X M0 M1 i j) r c)
    (ce : Fin C → ErrorPoly n) (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) :
    pairing2 V (∑ i, fun j ↦ divInt P (cellNum va Ds α es Dr sm X M0 M1 i j) - ce i • eB i j) =
      (∑ i, ∑ k, pairing (alignWeight V Dr Ds i k) (va i k)) / P +
        pairing2 (stateWeight V Dr α) es / P +
        (∑ j, ∑ i, (-pairing (V j) (X i j) + pairing (pullSmul (V j) sm) (M0 i j) +
          pairing (V j) (M1 i j))) / P -
        ∑ j, ∑ i, pairing (pullSmul (V j) (ce i)) (eB i j) := by
  have hj : ∀ j, pairing (V j) ((∑ i, fun j ↦ divInt P (cellNum va Ds α es Dr sm X M0 M1 i j) -
      ce i • eB i j) j) = ∑ i, (pairing (V j) (cellNum va Ds α es Dr sm X M0 M1 i j) / P -
        pairing (pullSmul (V j) (ce i)) (eB i j)) := by
    intro j
    rw [Finset.sum_apply]
    exact pairing_blockErr hn (V j) hP (fun i ↦ cellNum va Ds α es Dr sm X M0 M1 i j)
      (fun i r c ↦ hdiv i j r c) ce (fun i ↦ eB i j)
  have halign : ∑ i, ∑ k, pairing (alignWeight V Dr Ds i k) (va i k) =
      ∑ j, ∑ i, ∑ j', ∑ k, pairing (pullMul (pullMul (V j) (halfRows (Dr i j) j'))
        (halfRows (Ds j') k)) (va i k) := by
    simp only [alignWeight, pairing_weight_sum]
    calc ∑ i, ∑ k, ∑ j, ∑ j', pairing (pullMul (pullMul (V j) (halfRows (Dr i j) j'))
          (halfRows (Ds j') k)) (va i k)
        = ∑ i, ∑ j, ∑ k, ∑ j', pairing (pullMul (pullMul (V j) (halfRows (Dr i j) j'))
          (halfRows (Ds j') k)) (va i k) := Finset.sum_congr rfl fun i _ ↦ Finset.sum_comm
      _ = ∑ i, ∑ j, ∑ j', ∑ k, pairing (pullMul (pullMul (V j) (halfRows (Dr i j) j'))
          (halfRows (Ds j') k)) (va i k) :=
          Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦ Finset.sum_comm
      _ = _ := Finset.sum_comm
  have hstate : pairing2 (stateWeight V Dr α) es =
      ∑ j, ∑ i, ∑ j', pairing (pullSmul (pullMul (V j) (halfRows (Dr i j) j')) (α i)) (es j') := by
    simp only [pairing2, stateWeight, pairing_weight_sum]
    calc ∑ j', ∑ i, ∑ j, pairing (pullSmul (pullMul (V j) (halfRows (Dr i j) j')) (α i)) (es j')
        = ∑ i, ∑ j', ∑ j, pairing (pullSmul (pullMul (V j) (halfRows (Dr i j) j')) (α i)) (es j') :=
          Finset.sum_comm
      _ = ∑ i, ∑ j, ∑ j', pairing (pullSmul (pullMul (V j) (halfRows (Dr i j) j')) (α i)) (es j') :=
          Finset.sum_congr rfl fun i _ ↦ Finset.sum_comm
      _ = _ := Finset.sum_comm
  rw [pairing2, Finset.sum_congr rfl fun j _ ↦ hj j, halign, hstate]
  simp only [Finset.sum_div, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  refine Finset.sum_congr rfl fun j _ ↦ Finset.sum_congr rfl fun i _ ↦ ?_
  rw [pairing_cellNum hn, Finset.sum_add_distrib]
  simp only [← Finset.sum_div]
  ring

/-- The numerator `A va` of the alignment of the member of cell `i`: its error after the digits
`H`, minus the row residue, plus the mask-weighted key residues. -/
noncomputable def alignNum {mT : Nat} (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT) (H : ErrorMatrix n mT mA)
    (sm : ErrorPoly n) (R K0 K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) (i : Fin C) (k : Fin 2) :
    ErrorMatrix n 1 mA :=
  ea i k * H - R i k + sm • K0 i k + K1 i k

theorem pairing_alignDiv (hn : 0 < n) {mT : Nat} (U : Weight n 1 mA) {A : ℤ} (hA : A ≠ 0)
    (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT) (H : ErrorMatrix n mT mA) (sm : ErrorPoly n)
    (R K0 K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) (i : Fin C) (k : Fin 2)
    (hdiv : ∀ r c, alignNum ea H sm R K0 K1 i k r c =
      (A : ErrorPoly n) * divInt A (alignNum ea H sm R K0 K1 i k) r c) :
    pairing U (divInt A (alignNum ea H sm R K0 K1 i k)) =
      (pairing (pullMul U H) (ea i k) - pairing U (R i k) + pairing (pullSmul U sm) (K0 i k) +
        pairing U (K1 i k)) / A := by
  rw [pairing_divInt hn U hA hdiv]
  unfold alignNum
  rw [pairing_alignNumer hn]

/-! ## Bounds -/

theorem coeffBound_halfRows {m m' : Nat} {d : ErrorMatrix n (m + m) m'} {β : Nat}
    (hd : CoeffBound d β) (j : Fin 2) : CoeffBound (halfRows d j) β := by
  unfold halfRows
  split
  · exact coeffBound_topRows hd
  · exact coeffBound_bottomRows hd

theorem sum_sq_le_sq_sum {ι : Type} (s : Finset ι) (a : ι → ℝ) (ha : ∀ x ∈ s, 0 ≤ a x) :
    ∑ x ∈ s, a x ^ 2 ≤ (∑ x ∈ s, a x) ^ 2 := by
  rw [sq (∑ x ∈ s, a x), Finset.sum_mul]
  refine Finset.sum_le_sum fun x hx ↦ ?_
  rw [sq]
  exact mul_le_mul_of_nonneg_left (Finset.single_le_sum ha hx) (ha x hx)

/-- The energy of a weight is at most its squared `l1` norm. -/
theorem weightEnergy_le_sq {rows columns : Nat} (W : Weight n rows columns) :
    weightEnergy W ≤ weightL1 W ^ 2 := by
  unfold weightEnergy weightL1
  calc ∑ r, ∑ c, ∑ k, W r c k ^ 2 = ∑ r, ∑ c, ∑ k, |W r c k| ^ 2 := by simp only [sq_abs]
    _ ≤ ∑ r, ∑ c, (∑ k, |W r c k|) ^ 2 := Finset.sum_le_sum fun r _ ↦ Finset.sum_le_sum fun c _ ↦
        sum_sq_le_sq_sum _ _ fun _ _ ↦ abs_nonneg _
    _ ≤ ∑ r, (∑ c, ∑ k, |W r c k|) ^ 2 := Finset.sum_le_sum fun r _ ↦
        sum_sq_le_sq_sum _ _ fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ abs_nonneg _
    _ ≤ (∑ r, ∑ c, ∑ k, |W r c k|) ^ 2 :=
        sum_sq_le_sq_sum _ _ fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦ Finset.sum_nonneg fun _ _ ↦
          abs_nonneg _

theorem sum_weightL1_nonneg {ι : Type} [Fintype ι] {rows columns : Nat} (V : ι → Weight n rows columns) :
    0 ≤ ∑ j, weightL1 (V j) :=
  Finset.sum_nonneg fun _ _ ↦ weightL1_nonneg _

/-- The `l1` norm of an alignment weight. -/
theorem weightL1_alignWeight_le (V : Fin 2 → Weight n 1 mo)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hDs : ∀ j', CoeffBound (Ds j') βs)
    (i : Fin C) (k : Fin 2) :
    weightL1 (alignWeight V Dr Ds i k) ≤
      (∑ j, weightL1 (V j)) * (2 * (ms * n * βr)) * (mA * n * βs) := by
  unfold alignWeight
  refine (weightL1_finsetSum _ _).trans ?_
  rw [Finset.sum_mul, Finset.sum_mul]
  refine Finset.sum_le_sum fun j _ ↦ (weightL1_finsetSum _ _).trans ?_
  calc ∑ j', weightL1 (pullMul (pullMul (V j) (halfRows (Dr i j) j')) (halfRows (Ds j') k))
      ≤ ∑ _j' : Fin 2, weightL1 (V j) * (ms * n * βr) * (mA * n * βs) :=
        Finset.sum_le_sum fun j' _ ↦
          (weightL1_pullMul_le _ (coeffBound_halfRows (hDs j') k)).trans
            (mul_le_mul_of_nonneg_right (weightL1_pullMul_le _ (coeffBound_halfRows (hDr i j) j'))
              (by positivity))
    _ = _ := by simp; ring

/-- The energy of an alignment weight. -/
theorem weightEnergy_alignWeight_le (V : Fin 2 → Weight n 1 mo)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    {βr βs : Nat} (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hDs : ∀ j', CoeffBound (Ds j') βs)
    (i : Fin C) (k : Fin 2) :
    weightEnergy (alignWeight V Dr Ds i k) ≤
      ((∑ j, weightL1 (V j)) * (2 * (ms * n * βr))) ^ 2 * (mA * n * βs ^ 2) := by
  have hfam : alignWeight V Dr Ds i k =
      pullMulFam (fun x : Fin 2 × Fin 2 ↦ pullMul (V x.1) (halfRows (Dr i x.1) x.2))
        (fun x ↦ halfRows (Ds x.2) k) := by
    unfold alignWeight pullMulFam
    rw [Fintype.sum_prod_type]
  have hsum : ∑ x : Fin 2 × Fin 2, weightL1 (pullMul (V x.1) (halfRows (Dr i x.1) x.2)) ≤
      (∑ j, weightL1 (V j)) * (2 * (ms * n * βr)) := by
    rw [Fintype.sum_prod_type, Finset.sum_mul]
    refine Finset.sum_le_sum fun j _ ↦ ?_
    calc ∑ j', weightL1 (pullMul (V j) (halfRows (Dr i j) j'))
        ≤ ∑ _j' : Fin 2, weightL1 (V j) * (ms * n * βr) :=
          Finset.sum_le_sum fun j' _ ↦ weightL1_pullMul_le _ (coeffBound_halfRows (hDr i j) j')
      _ = _ := by simp; ring
  have hnn : 0 ≤ ∑ x : Fin 2 × Fin 2, weightL1 (pullMul (V x.1) (halfRows (Dr i x.1) x.2)) :=
    Finset.sum_nonneg fun _ _ ↦ weightL1_nonneg _
  have hfac : (0 : ℝ) ≤ mA * n * βs ^ 2 := by positivity
  rw [hfam]
  calc weightEnergy (pullMulFam (fun x : Fin 2 × Fin 2 ↦ pullMul (V x.1) (halfRows (Dr i x.1) x.2))
        (fun x ↦ halfRows (Ds x.2) k))
      ≤ (∑ x : Fin 2 × Fin 2, weightL1 (pullMul (V x.1) (halfRows (Dr i x.1) x.2))) ^ 2 *
          (mA * n * βs ^ 2) :=
        weightEnergy_pullMulFam_le _ fun x ↦ coeffBound_halfRows (hDs x.2) k
    _ ≤ _ := mul_le_mul_of_nonneg_right (pow_le_pow_left₀ hnn hsum 2) hfac

/-- The total `l1` norm of the state weight. -/
theorem weightL1_stateWeight_le (V : Fin 2 → Weight n 1 mo)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (α : Fin C → ErrorPoly n) {βr : Nat} {Sα : ℝ}
    (hDr : ∀ i j, CoeffBound (Dr i j) βr) (hSα : 0 ≤ Sα)
    (hα : ∀ i, ∑ k, |(((α i).coeff k : ℤ) : ℝ)| ≤ Sα) :
    ∑ j', weightL1 (stateWeight V Dr α j') ≤
      (∑ j, weightL1 (V j)) * (2 * (ms * n * βr)) * (C * Sα) := by
  have hone : ∀ j', weightL1 (stateWeight V Dr α j') ≤
      ∑ _i : Fin C, ∑ j : Fin 2, weightL1 (V j) * (ms * n * βr) * Sα := by
    intro j'
    unfold stateWeight
    refine (weightL1_finsetSum _ _).trans (Finset.sum_le_sum fun i _ ↦
      (weightL1_finsetSum _ _).trans (Finset.sum_le_sum fun j _ ↦ ?_))
    exact (weightL1_pullSmul_le _ _).trans (mul_le_mul
      (weightL1_pullMul_le _ (coeffBound_halfRows (hDr i j) j')) (hα i)
      (Finset.sum_nonneg fun _ _ ↦ abs_nonneg _) (mul_nonneg (weightL1_nonneg _) (by positivity)))
  refine (Finset.sum_le_sum fun j' _ ↦ hone j').trans (le_of_eq ?_)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.sum_mul]
  push_cast
  ring

/-- A sum of pairings is at most the sum of the `l1` norms of the weights times a common
coefficient bound. -/
theorem abs_sum_pairing_le {ι : Type} (s : Finset ι) {rows columns : Nat}
    (W : ι → Weight n rows columns) (e : ι → ErrorMatrix n rows columns) {B : Nat}
    (he : ∀ x ∈ s, CoeffBound (e x) B) :
    |∑ x ∈ s, pairing (W x) (e x)| ≤ (∑ x ∈ s, weightL1 (W x)) * B := by
  rw [Finset.sum_mul]
  exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun x hx ↦
    abs_pairing_le (W x) (he x hx))


/-! ## Site form -/

theorem sum_sum_split4 {ι κ : Type} [Fintype ι] [Fintype κ] (a b c d : ι → κ → ℝ) (D : ℝ) :
    ∑ i, ∑ k, (a i k - b i k + c i k + d i k) / D =
      ((∑ i, ∑ k, a i k) - (∑ i, ∑ k, b i k) + (∑ i, ∑ k, c i k) + ∑ i, ∑ k, d i k) / D := by
  simp only [add_div, sub_div, Finset.sum_div, Finset.sum_add_distrib, Finset.sum_sub_distrib]

theorem sum_sum_split3 {ι κ : Type} [Fintype ι] [Fintype κ] (a b c : ι → κ → ℝ) :
    ∑ i, ∑ k, (-a i k + b i k + c i k) =
      -(∑ i, ∑ k, a i k) + (∑ i, ∑ k, b i k) + ∑ i, ∑ k, c i k := by
  simp only [Finset.sum_add_distrib, Finset.sum_neg_distrib]



/-- A pairing with a row as a sum over its columns and coefficients. -/
theorem pairing_row {m : Nat} (W : Weight n 1 m) (e : ErrorMatrix n 1 m) :
    pairing W e = ∑ x : Fin m × Fin n, W 0 x.1 x.2 * (((e 0 x.1).coeff x.2 : ℤ) : ℝ) := by
  rw [pairing, Fin.sum_univ_one, Fintype.sum_prod_type]

/-- The value of a residue site `(i, j, column, coefficient)`. -/
noncomputable def siteVal {m : Nat} (X : Fin C → Fin 2 → ErrorMatrix n 1 m) (x : Fin C × Fin 2 × Fin m × Fin n) : ℤ :=
  (X x.1 x.2.1 0 x.2.2.1).coeff x.2.2.2

/-- The coefficient of a residue site subtracted with the weight `U i j`, divided by `D`. -/
noncomputable def subCoef {m : Nat} (U : Fin C → Fin 2 → Weight n 1 m) (D : ℝ)
    (x : Fin C × Fin 2 × Fin m × Fin n) : ℝ :=
  -U x.1 x.2.1 0 x.2.2.1 x.2.2.2 / D

/-- The coefficient of a mask-weighted residue site, divided by `D`. -/
noncomputable def maskCoef {m : Nat} (U : Fin C → Fin 2 → Weight n 1 m) (sm : ErrorPoly n) (D : ℝ)
    (x : Fin C × Fin 2 × Fin m × Fin n) : ℝ :=
  pullSmul (U x.1 x.2.1) sm 0 x.2.2.1 x.2.2.2 / D

theorem sum_pairing_sites {m : Nat} (U : Fin C → Fin 2 → Weight n 1 m)
    (X : Fin C → Fin 2 → ErrorMatrix n 1 m) (D : ℝ) :
    -(∑ i, ∑ j, pairing (U i j) (X i j)) / D = ∑ x, subCoef U D x * (siteVal X x : ℝ) := by
  simp only [pairing_row, subCoef, siteVal, Fintype.sum_prod_type, neg_div, ← Finset.sum_neg_distrib,
    Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦
    Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ by ring

theorem sum_pairing_mask_sites {m : Nat} (U : Fin C → Fin 2 → Weight n 1 m) (sm : ErrorPoly n)
    (K : Fin C → Fin 2 → ErrorMatrix n 1 m) (D : ℝ) :
    (∑ i, ∑ j, pairing (pullSmul (U i j) sm) (K i j)) / D =
      ∑ x, maskCoef U sm D x * (siteVal K x : ℝ) := by
  simp only [pairing_row, maskCoef, siteVal, Fintype.sum_prod_type, Finset.sum_div]
  refine Finset.sum_congr rfl fun i _ ↦ Finset.sum_congr rfl fun j _ ↦
    Finset.sum_congr rfl fun c _ ↦ Finset.sum_congr rfl fun k _ ↦ by ring

/-- The remainder of a block of cells: the mask-weighted and plain key residues of the cells and
their RHS errors. -/
noncomputable def cellRem (V : Fin 2 → Weight n 1 mo) (P : ℝ) (sm : ErrorPoly n)
    (M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo) (ce : Fin C → ErrorPoly n)
    (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) : ℝ :=
  ((∑ j, ∑ i, pairing (pullSmul (V j) sm) (M0 i j)) + ∑ j, ∑ i, pairing (V j) (M1 i j)) / P -
    ∑ j, ∑ i, pairing (pullSmul (V j) (ce i)) (eB i j)

/-- The remainder of the alignments: the member errors after the digits `H` and the plain key
residues. -/
noncomputable def alignRem {mT : Nat} (U : Fin C → Fin 2 → Weight n 1 mA) (D : ℝ)
    (H : ErrorMatrix n mT mA) (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT)
    (K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) : ℝ :=
  ((∑ i, ∑ k, pairing (pullMul (U i k) H) (ea i k)) + ∑ i, ∑ k, pairing (U i k) (K1 i k)) / D

/-- The pairing with the error leaving a block of aligned cells, in site form. -/
theorem pairing2_alignedBlock (hn : 0 < n) {mT : Nat} (V : Fin 2 → Weight n 1 mo) {P A : ℤ}
    (hP : P ≠ 0) (hA : A ≠ 0) (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT) (H : ErrorMatrix n mT mA)
    (R K0 K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    (α : Fin C → ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 ms)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (sm : ErrorPoly n)
    (X M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo)
    (hdivA : ∀ i k r c, alignNum ea H sm R K0 K1 i k r c =
      (A : ErrorPoly n) * divInt A (alignNum ea H sm R K0 K1 i k) r c)
    (hdiv : ∀ i j r c, cellNum (fun i k ↦ divInt A (alignNum ea H sm R K0 K1 i k)) Ds α es Dr sm X M0 M1 i j r c =
      (P : ErrorPoly n) * divInt P (cellNum (fun i k ↦ divInt A (alignNum ea H sm R K0 K1 i k)) Ds α es Dr sm X M0 M1 i j) r c)
    (ce : Fin C → ErrorPoly n) (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) :
    pairing2 V (∑ i, fun j ↦ divInt P (cellNum (fun i k ↦ divInt A (alignNum ea H sm R K0 K1 i k))
        Ds α es Dr sm X M0 M1 i j) - ce i • eB i j) =
      pairing2 (fun j' ↦ (P : ℝ)⁻¹ • stateWeight V Dr α j') es +
        ∑ x, subCoef (fun i j ↦ V j) P x * (siteVal X x : ℝ) +
        ∑ x, subCoef (alignWeight V Dr Ds) (P * A) x * (siteVal R x : ℝ) +
        ∑ x, maskCoef (alignWeight V Dr Ds) sm (P * A) x * (siteVal K0 x : ℝ) +
        alignRem (alignWeight V Dr Ds) (P * A) H ea K1 + cellRem V P sm M0 M1 ce eB := by
  rw [pairing2_cellBlock hn V hP _ Ds α es Dr sm X M0 M1 hdiv ce eB]
  have halign : ∀ i k, pairing (alignWeight V Dr Ds i k) (divInt A (alignNum ea H sm R K0 K1 i k)) =
      (pairing (pullMul (alignWeight V Dr Ds i k) H) (ea i k) - pairing (alignWeight V Dr Ds i k) (R i k) +
        pairing (pullSmul (alignWeight V Dr Ds i k) sm) (K0 i k) +
        pairing (alignWeight V Dr Ds i k) (K1 i k)) / A := fun i k ↦
    pairing_alignDiv hn _ hA ea H sm R K0 K1 i k (hdivA i k)
  simp only [halign]
  rw [sum_sum_split4, sum_sum_split3]
  have hstate : pairing2 (fun j' ↦ (P : ℝ)⁻¹ • stateWeight V Dr α j') es =
      pairing2 (stateWeight V Dr α) es / P := by
    simp only [pairing2, pairing_smul_weight, ← Finset.mul_sum, div_eq_inv_mul]
  have hX := sum_pairing_sites (fun (i : Fin C) j ↦ V j) X P
  have hR := sum_pairing_sites (alignWeight V Dr Ds) R (P * A)
  have hK := sum_pairing_mask_sites (alignWeight V Dr Ds) sm K0 (P * A)
  have hXc : ∑ j, ∑ i, pairing (V j) (X i j) = ∑ i, ∑ j, pairing (V j) (X i j) := Finset.sum_comm
  rw [hstate, ← hX, ← hR, ← hK, alignRem, cellRem, hXc]
  ring

/-- The pairing with the error leaving block `0`, whose members enter without alignment. -/
theorem pairing2_firstBlock (hn : 0 < n) (V : Fin 2 → Weight n 1 mo) {P : ℤ} (hP : P ≠ 0)
    (ea : Fin C → Fin 2 → ErrorMatrix n 1 mA) (Ds : Fin 2 → ErrorMatrix n (mA + mA) ms)
    (α : Fin C → ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 ms)
    (Dr : Fin C → Fin 2 → ErrorMatrix n (ms + ms) mo) (sm : ErrorPoly n)
    (X M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo)
    (hdiv : ∀ i j r c, cellNum ea Ds α es Dr sm X M0 M1 i j r c =
      (P : ErrorPoly n) * divInt P (cellNum ea Ds α es Dr sm X M0 M1 i j) r c)
    (ce : Fin C → ErrorPoly n) (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) :
    pairing2 V (∑ i, fun j ↦ divInt P (cellNum ea Ds α es Dr sm X M0 M1 i j) - ce i • eB i j) =
      pairing2 (fun j' ↦ (P : ℝ)⁻¹ • stateWeight V Dr α j') es +
        ∑ x, subCoef (fun i j ↦ V j) P x * (siteVal X x : ℝ) +
        (∑ i, ∑ k, pairing (alignWeight V Dr Ds i k) (ea i k)) / P + cellRem V P sm M0 M1 ce eB := by
  rw [pairing2_cellBlock hn V hP ea Ds α es Dr sm X M0 M1 hdiv ce eB, sum_sum_split3]
  have hstate : pairing2 (fun j' ↦ (P : ℝ)⁻¹ • stateWeight V Dr α j') es =
      pairing2 (stateWeight V Dr α) es / P := by
    simp only [pairing2, pairing_smul_weight, ← Finset.mul_sum, div_eq_inv_mul]
  have hX := sum_pairing_sites (fun (i : Fin C) j ↦ V j) X P
  have hXc : ∑ j, ∑ i, pairing (V j) (X i j) = ∑ i, ∑ j, pairing (V j) (X i j) := Finset.sum_comm
  rw [hstate, ← hX, cellRem, hXc]
  ring


/-- The remainder of a block of cells is at most the `l1` norm of the weight times the cell count
and the bounds of the residues and the RHS errors. -/
theorem abs_cellRem_le (V : Fin 2 → Weight n 1 mo) {P : ℝ} (hP : 0 < P) (sm : ErrorPoly n)
    (M0 M1 : Fin C → Fin 2 → ErrorMatrix n 1 mo) (ce : Fin C → ErrorPoly n)
    (eB : Fin C → Fin 2 → ErrorMatrix n 1 mo) {Ssm Sce : ℝ} {B0 B1 BB : Nat}
    (hsm : ∑ k, |((sm.coeff k : ℤ) : ℝ)| ≤ Ssm) (hce : ∀ i, ∑ k, |(((ce i).coeff k : ℤ) : ℝ)| ≤ Sce)
    (hM0 : ∀ i j, CoeffBound (M0 i j) B0) (hM1 : ∀ i j, CoeffBound (M1 i j) B1)
    (heB : ∀ i j, CoeffBound (eB i j) BB) :
    |cellRem V P sm M0 M1 ce eB| ≤
      (∑ j, weightL1 (V j)) * C * ((Ssm * B0 + B1) / P + Sce * BB) := by
  have hlam := sum_weightL1_nonneg V
  have h0 : ∀ j, |∑ i, pairing (pullSmul (V j) sm) (M0 i j)| ≤ C * (weightL1 (V j) * Ssm * B0) := by
    intro j
    refine (abs_sum_pairing_le _ _ _ fun i _ ↦ hM0 i j).trans ?_
    calc (∑ i : Fin C, weightL1 (pullSmul (V j) sm)) * B0
        ≤ (∑ _i : Fin C, weightL1 (V j) * Ssm) * B0 :=
          mul_le_mul_of_nonneg_right (Finset.sum_le_sum fun i _ ↦
            (weightL1_pullSmul_le _ _).trans (mul_le_mul_of_nonneg_left hsm (weightL1_nonneg _)))
            (Nat.cast_nonneg _)
      _ = C * (weightL1 (V j) * Ssm * B0) := by simp; ring
  have h1 : ∀ j, |∑ i, pairing (V j) (M1 i j)| ≤ C * (weightL1 (V j) * B1) := by
    intro j
    refine (abs_sum_pairing_le _ _ _ fun i _ ↦ hM1 i j).trans (le_of_eq ?_)
    simp; ring
  have hB : ∀ j, |∑ i, pairing (pullSmul (V j) (ce i)) (eB i j)| ≤ C * (weightL1 (V j) * Sce * BB) := by
    intro j
    refine (abs_sum_pairing_le _ _ _ fun i _ ↦ heB i j).trans ?_
    calc (∑ i : Fin C, weightL1 (pullSmul (V j) (ce i))) * BB
        ≤ (∑ _i : Fin C, weightL1 (V j) * Sce) * BB :=
          mul_le_mul_of_nonneg_right (Finset.sum_le_sum fun i _ ↦
            (weightL1_pullSmul_le _ _).trans (mul_le_mul_of_nonneg_left (hce i) (weightL1_nonneg _)))
            (Nat.cast_nonneg _)
      _ = C * (weightL1 (V j) * Sce * BB) := by simp; ring
  have hsum : ∀ (f : Fin 2 → ℝ) (b : Fin 2 → ℝ), (∀ j, |f j| ≤ b j) → |∑ j, f j| ≤ ∑ j, b j :=
    fun f b h ↦ (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun j _ ↦ h j)
  have e0 := hsum _ _ h0
  have e1 := hsum _ _ h1
  have eB' := hsum _ _ hB
  simp only [← Finset.mul_sum, ← Finset.sum_mul] at e0 e1 eB'
  unfold cellRem
  rw [abs_le] at e0 e1 eB' ⊢
  constructor
  · rw [show ((∑ j, weightL1 (V j)) * C * ((Ssm * B0 + B1) / P + Sce * BB)) =
        ((∑ j, weightL1 (V j)) * C * (Ssm * B0 + B1)) / P + (∑ j, weightL1 (V j)) * C * Sce * BB by
        field_simp]
    have : -((∑ j, weightL1 (V j)) * C * (Ssm * B0 + B1)) / P ≤
        ((∑ j, ∑ i, pairing (pullSmul (V j) sm) (M0 i j)) + ∑ j, ∑ i, pairing (V j) (M1 i j)) / P := by
      rw [div_le_div_iff_of_pos_right hP]
      nlinarith [e0.1, e1.1]
    rw [neg_add, neg_div] at *
    nlinarith [eB'.2]
  · have : ((∑ j, ∑ i, pairing (pullSmul (V j) sm) (M0 i j)) + ∑ j, ∑ i, pairing (V j) (M1 i j)) / P ≤
        ((∑ j, weightL1 (V j)) * C * (Ssm * B0 + B1)) / P := by
      rw [div_le_div_iff_of_pos_right hP]
      nlinarith [e0.2, e1.2]
    have h2 : ((∑ j, weightL1 (V j)) * C * ((Ssm * B0 + B1) / P + Sce * BB)) =
        ((∑ j, weightL1 (V j)) * C * (Ssm * B0 + B1)) / P + (∑ j, weightL1 (V j)) * C * Sce * BB := by
      field_simp
    rw [h2]
    nlinarith [eB'.1]

/-- The remainder of the alignments is at most the `l1` norms of the alignment weights times the
column `l1` norm `SH` of the digits, the member error bound, and the key residue bound. -/
theorem abs_alignRem_le {mT : Nat} (U : Fin C → Fin 2 → Weight n 1 mA) {D : ℝ} (hD : 0 < D)
    (H : ErrorMatrix n mT mA) (ea : Fin C → Fin 2 → ErrorMatrix n 1 mT)
    (K1 : Fin C → Fin 2 → ErrorMatrix n 1 mA) {SH : ℝ} {Bea B1 : Nat}
    (hH : ∀ c, ∑ l, ∑ j, |(((H l c).coeff j : ℤ) : ℝ)| ≤ SH)
    (hea : ∀ i k, CoeffBound (ea i k) Bea) (hK1 : ∀ i k, CoeffBound (K1 i k) B1) :
    |alignRem U D H ea K1| ≤ (∑ i, ∑ k, weightL1 (U i k)) * (SH * Bea + B1) / D := by
  unfold alignRem
  rw [abs_div, abs_of_pos hD]
  refine div_le_div_of_nonneg_right ?_ hD.le
  refine (abs_add_le _ _).trans ?_
  have hpt : ∀ i k, |pairing (pullMul (U i k) H) (ea i k)| + |pairing (U i k) (K1 i k)| ≤
      weightL1 (U i k) * (SH * Bea + B1) := by
    intro i k
    have h1 := abs_pairing_le (pullMul (U i k) H) (hea i k)
    have h2 := abs_pairing_le (U i k) (hK1 i k)
    have h3 := weightL1_pullMul_le_column (U i k) hH
    have h4 := weightL1_nonneg (U i k)
    have h5 : (0 : ℝ) ≤ Bea := Nat.cast_nonneg _
    nlinarith
  calc |∑ i, ∑ k, pairing (pullMul (U i k) H) (ea i k)| + |∑ i, ∑ k, pairing (U i k) (K1 i k)|
      ≤ (∑ i, ∑ k, |pairing (pullMul (U i k) H) (ea i k)|) + ∑ i, ∑ k, |pairing (U i k) (K1 i k)| := by
        refine add_le_add ((Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ ↦
          Finset.abs_sum_le_sum_abs _ _)) ((Finset.abs_sum_le_sum_abs _ _).trans
          (Finset.sum_le_sum fun i _ ↦ Finset.abs_sum_le_sum_abs _ _))
    _ = ∑ i, ∑ k, (|pairing (pullMul (U i k) H) (ea i k)| + |pairing (U i k) (K1 i k)|) := by
        simp only [Finset.sum_add_distrib]
    _ ≤ ∑ i, ∑ k, weightL1 (U i k) * (SH * Bea + B1) :=
        Finset.sum_le_sum fun i _ ↦ Finset.sum_le_sum fun k _ ↦ hpt i k
    _ = _ := by simp only [Finset.sum_mul]


end CellBlock

section LutStep

variable {Wn m mF : Nat}

/-- Branch `i` of the readout LUT for one output component: the helper error after the digits
`DE i`, the substituted input error after the digits `DD i`, and the key residues of the branch. -/
noncomputable def lutBranch (eh : Fin Wn → Fin 2 → ErrorMatrix n 1 m) (DE : Fin Wn → ErrorMatrix n (m + m) m)
    (σ : Fin Wn → Nat) (hσ : ∀ i, Odd (σ i)) (ec : Fin 2 → ErrorMatrix n 1 m)
    (DD : Fin Wn → ErrorMatrix n (m + m) m) (s : Fin Wn → ErrorPoly n)
    (KM0 KM1 : Fin Wn → ErrorMatrix n 1 m) (i : Fin Wn) : ErrorMatrix n 1 m :=
  pairMul (eh i) (DE i) + pairMul (fun j' ↦ substituteMatrix (σ i) (hσ i) (ec j')) (DD i) +
    s i • KM0 i + KM1 i

/-- The weight on the LUT input error, before the division by `P`. -/
noncomputable def lutStateWeight (hn : 0 < n) (Wl : Weight n 1 m) (DD : Fin Wn → ErrorMatrix n (m + m) m)
    (σ : Fin Wn → Nat) (j' : Fin 2) : Weight n 1 m :=
  ∑ i, pullSubst hn (σ i) (pullMul Wl (halfRows (DD i) j'))

/-- The pairing with the finalized error of the readout LUT. -/
theorem pairing_lutFinal (hn : 0 < n) (Wf : Weight n 1 mF) (F : ErrorMatrix n m mF) {P : ℤ} (hP : P ≠ 0)
    (eh : Fin Wn → Fin 2 → ErrorMatrix n 1 m) (DE : Fin Wn → ErrorMatrix n (m + m) m)
    (σ : Fin Wn → Nat) (hσ : ∀ i, Odd (σ i)) (ec : Fin 2 → ErrorMatrix n 1 m)
    (DD : Fin Wn → ErrorMatrix n (m + m) m) (s : Fin Wn → ErrorPoly n)
    (KM0 KM1 : Fin Wn → ErrorMatrix n 1 m) (sm : ErrorPoly n) (R K0 K1 : ErrorMatrix n 1 m)
    (hdiv : ∀ r c, (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) r c =
      (P : ErrorPoly n) * divInt P (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) r c)
    (a : Fin Wn → ErrorPoly n) (sw : Fin Wn → ErrorMatrix n 1 m) :
    pairing Wf ((divInt P (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) -
        ∑ i, a i • sw i) * F) =
      ((∑ i, ∑ j', pairing (pullMul (pullMul Wf F) (halfRows (DE i) j')) (eh i j')) +
          (∑ i, (pairing (pullSmul (pullMul Wf F) (s i)) (KM0 i) + pairing (pullMul Wf F) (KM1 i))) -
          pairing (pullMul Wf F) R + pairing (pullSmul (pullMul Wf F) sm) K0 +
          pairing (pullMul Wf F) K1) / P +
        pairing2 (lutStateWeight hn (pullMul Wf F) DD σ) ec / P -
        ∑ i, pairing (pullSmul (pullMul Wf F) (a i)) (sw i) := by
  set Wl := pullMul Wf F
  have hbranch : ∀ i, pairing Wl (lutBranch eh DE σ hσ ec DD s KM0 KM1 i) =
      (∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j')) +
        (∑ j', pairing (pullSubst hn (σ i) (pullMul Wl (halfRows (DD i) j'))) (ec j')) +
        pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i) := by
    intro i
    unfold lutBranch
    rw [pairing_add, pairing_add, pairing_add, pairing_pairMul_sum hn, pairing_pairMul_sum hn,
      pairing_smul hn]
    simp only [pairing_substitute hn (hσ i)]
  have hstate : pairing2 (lutStateWeight hn Wl DD σ) ec =
      ∑ i, ∑ j', pairing (pullSubst hn (σ i) (pullMul Wl (halfRows (DD i) j'))) (ec j') := by
    simp only [pairing2, lutStateWeight, pairing_weight_sum]
    exact Finset.sum_comm
  rw [pairing_mul hn, pairing_sub, pairing_divInt hn Wl hP hdiv, pairing_sum, hstate]
  simp only [pairing_add, pairing_sub, pairing_smul hn, pairing_sum, hbranch]
  simp only [Finset.sum_add_distrib, add_div, sub_div]
  ring

/-- The remainder of the readout LUT: the helper errors, the key residues of the branches and of
the finish, and the switching errors. -/
noncomputable def lutRem (Wl : Weight n 1 m) (P : ℝ) (eh : Fin Wn → Fin 2 → ErrorMatrix n 1 m)
    (DE : Fin Wn → ErrorMatrix n (m + m) m) (s : Fin Wn → ErrorPoly n)
    (KM0 KM1 : Fin Wn → ErrorMatrix n 1 m) (sm : ErrorPoly n) (K0 K1 : ErrorMatrix n 1 m)
    (a : Fin Wn → ErrorPoly n) (sw : Fin Wn → ErrorMatrix n 1 m) : ℝ :=
  ((∑ i, ∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j')) +
      (∑ i, (pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i))) +
      pairing (pullSmul Wl sm) K0 + pairing Wl K1) / P -
    ∑ i, pairing (pullSmul Wl (a i)) (sw i)

/-- The coefficient of a site `(column, coefficient)` of the LUT row residue. -/
noncomputable def lutCoef (Wl : Weight n 1 m) (P : ℝ) (x : Fin m × Fin n) : ℝ := -Wl 0 x.1 x.2 / P

/-- The value of a site of the LUT row residue. -/
noncomputable def lutVal (R : ErrorMatrix n 1 m) (x : Fin m × Fin n) : ℤ := (R 0 x.1).coeff x.2

/-- The pairing with the finalized error of the readout LUT, in site form. -/
theorem pairing_lutFinal_sites (hn : 0 < n) (Wf : Weight n 1 mF) (F : ErrorMatrix n m mF) {P : ℤ}
    (hP : P ≠ 0) (eh : Fin Wn → Fin 2 → ErrorMatrix n 1 m) (DE : Fin Wn → ErrorMatrix n (m + m) m)
    (σ : Fin Wn → Nat) (hσ : ∀ i, Odd (σ i)) (ec : Fin 2 → ErrorMatrix n 1 m)
    (DD : Fin Wn → ErrorMatrix n (m + m) m) (s : Fin Wn → ErrorPoly n)
    (KM0 KM1 : Fin Wn → ErrorMatrix n 1 m) (sm : ErrorPoly n) (R K0 K1 : ErrorMatrix n 1 m)
    (hdiv : ∀ r c, (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) r c =
      (P : ErrorPoly n) * divInt P (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) r c)
    (a : Fin Wn → ErrorPoly n) (sw : Fin Wn → ErrorMatrix n 1 m) :
    pairing Wf ((divInt P (∑ i, lutBranch eh DE σ hσ ec DD s KM0 KM1 i - R + sm • K0 + K1) -
        ∑ i, a i • sw i) * F) =
      pairing2 (fun j' ↦ (P : ℝ)⁻¹ • lutStateWeight hn (pullMul Wf F) DD σ j') ec +
        ∑ x, lutCoef (pullMul Wf F) P x * (lutVal R x : ℝ) +
        lutRem (pullMul Wf F) P eh DE s KM0 KM1 sm K0 K1 a sw := by
  rw [pairing_lutFinal hn Wf F hP eh DE σ hσ ec DD s KM0 KM1 sm R K0 K1 hdiv a sw]
  have hstate : pairing2 (fun j' ↦ (P : ℝ)⁻¹ • lutStateWeight hn (pullMul Wf F) DD σ j') ec =
      pairing2 (lutStateWeight hn (pullMul Wf F) DD σ) ec / P := by
    simp only [pairing2, pairing_smul_weight, ← Finset.mul_sum, div_eq_inv_mul]
  have hsites : ∑ x, lutCoef (pullMul Wf F) P x * (lutVal R x : ℝ) = -pairing (pullMul Wf F) R / P := by
    rw [pairing_row, neg_div, Finset.sum_div, ← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun x _ ↦ ?_
    simp only [lutCoef, lutVal]
    ring
  rw [hstate, hsites, lutRem]
  ring

/-- The total `l1` norm of the weight on the LUT input. -/
theorem weightL1_lutStateWeight_le (hn : 0 < n) (Wl : Weight n 1 m)
    (DD : Fin Wn → ErrorMatrix n (m + m) m) (σ : Fin Wn → Nat) (hcop : ∀ i, n.Coprime (σ i))
    {βD : Nat} (hDD : ∀ i, CoeffBound (DD i) βD) :
    ∑ j', weightL1 (lutStateWeight hn Wl DD σ j') ≤ weightL1 Wl * (2 * (m * n * βD)) * Wn := by
  have hone : ∀ j', weightL1 (lutStateWeight hn Wl DD σ j') ≤
      ∑ _i : Fin Wn, weightL1 Wl * (m * n * βD) := by
    intro j'
    unfold lutStateWeight
    refine (weightL1_finsetSum _ _).trans (Finset.sum_le_sum fun i _ ↦ ?_)
    rw [weightL1_pullSubst hn (hcop i)]
    exact weightL1_pullMul_le _ (coeffBound_halfRows (hDD i) j')
  refine (Finset.sum_le_sum fun j' _ ↦ hone j').trans (le_of_eq ?_)
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  push_cast
  ring

/-- The remainder of the readout LUT is at most the `l1` norm of its weight times the bounds of
the helper errors, the key residues, and the switching errors. -/
theorem abs_lutRem_le (Wl : Weight n 1 m) {P : ℝ} (hP : 0 < P)
    (eh : Fin Wn → Fin 2 → ErrorMatrix n 1 m) (DE : Fin Wn → ErrorMatrix n (m + m) m)
    (s : Fin Wn → ErrorPoly n) (KM0 KM1 : Fin Wn → ErrorMatrix n 1 m) (sm : ErrorPoly n)
    (K0 K1 : ErrorMatrix n 1 m) (a : Fin Wn → ErrorPoly n) (sw : Fin Wn → ErrorMatrix n 1 m)
    {βE Beh B0 B1 Bsw : Nat} {Ss Sa : ℝ}
    (hDE : ∀ i, CoeffBound (DE i) βE) (heh : ∀ i j, CoeffBound (eh i j) Beh)
    (hs : ∀ i, ∑ k, |(((s i).coeff k : ℤ) : ℝ)| ≤ Ss) (hsm : ∑ k, |((sm.coeff k : ℤ) : ℝ)| ≤ Ss)
    (hKM0 : ∀ i, CoeffBound (KM0 i) B0) (hKM1 : ∀ i, CoeffBound (KM1 i) B1)
    (hK0 : CoeffBound K0 B0) (hK1 : CoeffBound K1 B1)
    (ha : ∀ i, ∑ k, |(((a i).coeff k : ℤ) : ℝ)| ≤ Sa) (hsw : ∀ i, CoeffBound (sw i) Bsw) :
    |lutRem Wl P eh DE s KM0 KM1 sm K0 K1 a sw| ≤
      weightL1 Wl * ((Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P +
        Wn * Sa * Bsw) := by
  have hl := weightL1_nonneg Wl
  have hpE : ∀ i, |∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j')| ≤
      weightL1 Wl * (2 * (m * n * βE) * Beh) := by
    intro i
    refine (abs_sum_pairing_le _ _ _ fun j' _ ↦ heh i j').trans ?_
    calc (∑ j', weightL1 (pullMul Wl (halfRows (DE i) j'))) * Beh
        ≤ (∑ _j' : Fin 2, weightL1 Wl * (m * n * βE)) * Beh :=
          mul_le_mul_of_nonneg_right (Finset.sum_le_sum fun j' _ ↦
            weightL1_pullMul_le _ (coeffBound_halfRows (hDE i) j')) (Nat.cast_nonneg _)
      _ = _ := by simp; ring
  have hpK : ∀ i, |pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i)| ≤
      weightL1 Wl * (Ss * B0 + B1) := by
    intro i
    refine (abs_add_le _ _).trans ?_
    have h1 := abs_pairing_le (pullSmul Wl (s i)) (hKM0 i)
    have h2 := abs_pairing_le Wl (hKM1 i)
    have h3 := (weightL1_pullSmul_le Wl (s i)).trans (mul_le_mul_of_nonneg_left (hs i) hl)
    have h4 : (0 : ℝ) ≤ B0 := Nat.cast_nonneg _
    nlinarith
  have hpF : |pairing (pullSmul Wl sm) K0 + pairing Wl K1| ≤ weightL1 Wl * (Ss * B0 + B1) := by
    refine (abs_add_le _ _).trans ?_
    have h1 := abs_pairing_le (pullSmul Wl sm) hK0
    have h2 := abs_pairing_le Wl hK1
    have h3 := (weightL1_pullSmul_le Wl sm).trans (mul_le_mul_of_nonneg_left hsm hl)
    have h4 : (0 : ℝ) ≤ B0 := Nat.cast_nonneg _
    nlinarith
  have hpS : ∀ i, |pairing (pullSmul Wl (a i)) (sw i)| ≤ weightL1 Wl * Sa * Bsw := by
    intro i
    have h1 := abs_pairing_le (pullSmul Wl (a i)) (hsw i)
    have h3 := (weightL1_pullSmul_le Wl (a i)).trans (mul_le_mul_of_nonneg_left (ha i) hl)
    have h4 : (0 : ℝ) ≤ Bsw := Nat.cast_nonneg _
    nlinarith
  have hsum : ∀ (f b : Fin Wn → ℝ), (∀ i, |f i| ≤ b i) → |∑ i, f i| ≤ ∑ i, b i :=
    fun f b h ↦ (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ ↦ h i)
  have eE := hsum _ _ hpE
  have eK := hsum _ _ hpK
  have eS := hsum _ _ hpS
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul] at eE eK eS
  unfold lutRem
  set X := (∑ i, ∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j')) +
      (∑ i, (pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i))) +
      pairing (pullSmul Wl sm) K0 + pairing Wl K1 with hX
  have hXb : |X| ≤ weightL1 Wl * (Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) := by
    rw [hX]
    refine (abs_add_le _ _).trans ?_
    have := abs_add_le ((∑ i, ∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j')) +
      (∑ i, (pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i)))) (pairing (pullSmul Wl sm) K0)
    have h2 := abs_add_le (∑ i, ∑ j', pairing (pullMul Wl (halfRows (DE i) j')) (eh i j'))
      (∑ i, (pairing (pullSmul Wl (s i)) (KM0 i) + pairing Wl (KM1 i)))
    have h5 := abs_pairing_le Wl hK1
    have h6 := abs_pairing_le (pullSmul Wl sm) hK0
    have h7 := (weightL1_pullSmul_le Wl sm).trans (mul_le_mul_of_nonneg_left hsm hl)
    have h8 : (0 : ℝ) ≤ B0 := Nat.cast_nonneg _
    have h9 : (0 : ℝ) ≤ B1 := Nat.cast_nonneg _
    have hSs : 0 ≤ Ss := (Finset.sum_nonneg fun _ _ ↦ abs_nonneg _).trans hsm
    nlinarith [mul_le_mul_of_nonneg_right h7 h8]
  rw [abs_le] at hXb eS ⊢
  have hP' : 0 < P := hP
  constructor
  · have h1 : -(weightL1 Wl * (Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1))) / P ≤ X / P :=
      div_le_div_of_nonneg_right hXb.1 hP.le
    rw [neg_div] at h1
    have : weightL1 Wl * ((Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P + Wn * Sa * Bsw) =
        weightL1 Wl * (Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P +
          Wn * (weightL1 Wl * Sa * Bsw) := by ring
    rw [this]
    linarith [eS.2]
  · have h1 : X / P ≤ weightL1 Wl * (Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P :=
      div_le_div_of_nonneg_right hXb.2 hP.le
    have : weightL1 Wl * ((Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P + Wn * Sa * Bsw) =
        weightL1 Wl * (Wn * (2 * (m * n * βE) * Beh + (Ss * B0 + B1)) + (Ss * B0 + B1)) / P +
          Wn * (weightL1 Wl * Sa * Bsw) := by ring
    rw [this]
    linarith [eS.1]

end LutStep

end MxxExponentLut
