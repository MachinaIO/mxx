import ExponentLut

/-!
# Public multiplication and public alignment

A public-payload encoding of `α` uses the mask as its payload: its target is
`t_j = (α (s₀, 1)_j) G` for the native gadget `G` of its modulus. Two operations act on it.

* `align_down`: the right action by the digits `H` of `p g`, for the gadget `g` of the next
  modulus re-encoded at `q = p q'`, followed by the runtime ModDown, gives the public encoding of
  `α` at `q'` with the native gadget of `q'`.
* `mul_public`: the stacked public encoding acting on the decomposition of a private key `A_s`,
  plus `α` times the private row, is an encoding of `α t_s` under the acted key: the gadget term
  `α (s₀ A_{s,0j} + A_{s,1j})` cancels against `α` times the private row.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- The one-row gadget times the digits of a one-row target. -/
theorem gadget_mul_decompose {q m m' : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {X : ExactMatrix q n 1 m'} {D : ExactMatrix q n m m'}
    (hD : gadgetDecomposeRuns backend base digits X D) : G * D = X := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hG⟩ := hG
  obtain ⟨layout₂, hl₂, _, _, hw₂, hD, _⟩ := hD
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  have hrec := regularGadgetMatrix_reconstruct (rows := 1) layout X hq hn hexact
  funext row column
  rw [← congrFun (congrFun hrec row) column]
  simp only [Matrix.mul_apply, hG, hD, castMatrixColumns_apply, castMatrixRows_apply]
  exact Fintype.sum_equiv (finCongr hw₁) _ _ (fun k ↦ rfl)

/-- A gadget decomposition is a function of its target. -/
theorem gadgetDecomposeRuns_unique {q rows columns inner : Nat} {backend : BackendContext}
    {base digits : Int} {X : ExactMatrix q n rows columns} {D D' : ExactMatrix q n inner columns}
    (hD : gadgetDecomposeRuns backend base digits X D)
    (hD' : gadgetDecomposeRuns backend base digits X D') : D = D' := by
  obtain ⟨layout, hl, _, _, hw, hD, _⟩ := hD
  obtain ⟨layout', hl', _, _, hw', hD', _⟩ := hD'
  rw [hl] at hl'
  cases hl'
  rw [hD, hD']

/-- The digits of a gadget decomposition are small. -/
theorem gadgetDecomposeRuns_small {q rows columns inner : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} {X : ExactMatrix q n rows columns}
    {D : ExactMatrix q n inner columns} (hD : gadgetDecomposeRuns backend base digits X D) :
    ∃ d : ErrorMatrix n inner columns, D = reduceMatrix q n inner columns d ∧
      CoeffBound d (base.toNat / 2) := by
  obtain ⟨layout, _, hbase, _, hw, hD, _⟩ := hD
  obtain ⟨d, hd, hb⟩ := regularDecomposeMatrix_bounded layout X hq hn
  refine ⟨fun row column ↦ d (Fin.cast hw row) column, ?_, fun row column ↦ ?_⟩
  · funext row column
    rw [hD, castMatrixRows_apply, hd]
    rfl
  · rw [hbase, Int.toNat_natCast]
    exact hb _ column

/-- `p x g` for the re-encoded gadget `g` of `q'` is `scaleUp` of `x g'`. -/
theorem scaleUpMatrix_rebasedGadget {q p q' m : Nat} (hq : q = p * q') (hq' : 1 < q')
    (hn : 0 < n) {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg : centeredRebaseRuns g' g) (x : ErrorPoly n) :
    scaleUpMatrix q p (reducePoly q' n x • g') = reducePoly q n (((p : Int) : ErrorPoly n) * x) • g := by
  let G : ErrorMatrix n 1 m := fun row column ↦ centeredIntLift (g' row column)
  have hgq : g = reduceMatrix q n 1 m G := centeredRebaseRuns_reduce hg
  have hgq' : g' = reduceMatrix q' n 1 m G := (reduceMatrix_centeredIntLift hq' hn g').symm
  rw [hgq, hgq', ← reduceMatrix_smul, ← reduceMatrix_smul]
  funext row column
  simp only [scaleUpMatrix, reduceMatrix, Matrix.smul_apply, smul_eq_mul]
  rw [scaleUp_reducePoly hq hq' hn, mul_assoc]

theorem matrixMulScalarRight_smul {q rows columns : Nat} (x : ExactMatrix q n rows columns)
    (a : ExactMatrix q n 1 1) : matrixMulScalarRight x a = a 0 0 • x := by
  funext i j
  simp [matrixMulScalarRight, mul_comm]

/-- A right action of one small matrix on every block. -/
theorem Rel.mulRight {q m m' : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat}
    (h : Rel sm A t c B) (d : ErrorMatrix n m m') (hd : CoeffBound d β) :
    Rel sm (fun k j ↦ A k j * reduceMatrix q n m m' d) (fun j ↦ t j * reduceMatrix q n m m' d)
      (fun j ↦ c j * reduceMatrix q n m m' d) (m * n * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j * d, fun j ↦ coeffBound_mul hn (he j) hd, fun j ↦ ?_⟩
  beta_reduce
  rw [hc j, reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]

/-- The balanced digits of zero are zero. -/
theorem balancedDigitAt_zero {base : Nat} (hbase : 1 < base) (digit : Nat) :
    balancedDigitAt base 0 digit = 0 := by
  have hd : balancedDigit base 0 = 0 := by
    unfold balancedDigit
    have : (0 : Int) < (base : Int) / 2 := by omega
    simp [this]
  have hr : ∀ j, balancedResidual base 0 j = 0 := by
    intro j
    induction j with
    | zero => rfl
    | succ j ih => rw [balancedResidual_succ, hd]; simpa using ih
  unfold balancedDigitAt
  rw [hr, hd]

/-- The digits of a constant polynomial are constant. -/
theorem regularDigitPoly_isConstant (layout : OrderedCrtLayout) (hn : 0 < n)
    {value : ExactPoly layout.modulusProduct n}
    (hvalue : ∀ k : Fin n, k.val ≠ 0 → value.coeff k = 0) (limb : CrtDecomposition.RegularLimb layout) :
    IsConstantPoly (CrtDecomposition.regularDigitPoly layout value limb) := by
  intro k hk
  rw [CrtDecomposition.regularDigitPoly_coeff layout value limb hn k]
  unfold CrtDecomposition.regularDigitCoefficient
  rw [hvalue k hk, map_zero]
  simp [centeredLift, balancedDigitAt_zero layout.base_gt_one]

/-- A preimage with constant witness survives a modulus cast. -/
theorem castExactMatrix_constantPreimage {q q' rows columns : Nat} (h : q = q')
    {value : ExactMatrix q n rows columns} {bound : Nat}
    (hv : ∃ w : ErrorMatrix n rows columns, value = reduceMatrix q n rows columns w ∧
      CoeffBound w bound ∧ IsConstantMatrix w) :
    ∃ w : ErrorMatrix n rows columns, castExactMatrix h value = reduceMatrix q' n rows columns w ∧
      CoeffBound w bound ∧ IsConstantMatrix w := by
  cases h
  exact hv

/-- A modulus cast of a matrix of constant polynomials is constant. -/
theorem castExactMatrix_coeff_zero {q q' rows columns : Nat} (h : q = q')
    {value : ExactMatrix q n rows columns}
    (hv : ∀ r c (k : Fin n), k.val ≠ 0 → (value r c).coeff k = 0) :
    ∀ r c (k : Fin n), k.val ≠ 0 → (castExactMatrix h value r c).coeff k = 0 := by
  cases h
  exact hv

/-- The full decomposition of a matrix of constant polynomials has a constant, small witness. -/
theorem fullDecomposeMatrix_constant {q rows columns : Nat} (layout : RegularLayout q)
    (target : ExactMatrix q n rows columns) (hq : 1 < q) (hn : 0 < n)
    (htarget : ∀ r c (k : Fin n), k.val ≠ 0 → (target r c).coeff k = 0) :
    ∃ w, fullDecomposeMatrix layout target = reduceMatrix q n _ columns w ∧
      CoeffBound w (layout.base / 2) ∧ IsConstantMatrix w := by
  unfold fullDecomposeMatrix
  apply castExactMatrix_constantPreimage
  let value := castExactMatrix layout.product_eq target
  have hvalue := castExactMatrix_coeff_zero layout.product_eq htarget
  let witness : ErrorMatrix n
      (rows * (layout.crtModuli.length * layout.digitsPerTower)) columns :=
    fun row column ↦ CrtDecomposition.regularDigitPoly layout.ordered
      (value ((regularIndexEquiv layout rows).symm row).1 column)
      ((regularIndexEquiv layout rows).symm row).2
  refine ⟨witness, ?_, ?_, ?_⟩
  · funext row column
    exact CrtDecomposition.regularDigitExactPoly_eq_reduce layout.ordered _ _
      (by rw [layout.ordered.modulusProduct_eq_q]; exact hq) hn
  · intro row column coefficient
    exact (coeff_natAbs_le_polyNorm _ coefficient).trans
      (CrtDecomposition.regularDigitPoly_bound layout.ordered hn _ _)
  · intro row column
    exact regularDigitPoly_isConstant layout.ordered hn (fun k hk ↦ hvalue _ _ k hk) _

/-- The gadget decomposition of a matrix of constant polynomials under an exact gadget has a
constant witness of digits at most half the base. -/
theorem gadgetDecomposeRuns_constant {q rows columns inner : Nat} (hq : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} {X : ExactMatrix q n rows columns}
    {D : ExactMatrix q n inner columns} (hD : gadgetDecomposeRuns backend base digits X D)
    (hexact : ∀ layout, backend.regularLayout q n = some layout → layout.droppedModuli = 0)
    (hX : ∀ r c (k : Fin n), k.val ≠ 0 → (X r c).coeff k = 0) :
    ∃ d : ErrorMatrix n inner columns, D = reduceMatrix q n inner columns d ∧
      CoeffBound d (base.toNat / 2) ∧ IsConstantMatrix d := by
  obtain ⟨layout, hl, hbase, _, hw, hD, _⟩ := hD
  have hcorrected : regularCorrectedTarget layout X = X := by
    simp [regularCorrectedTarget, hexact layout hl]
  obtain ⟨w, hwq, hwb, hwc⟩ := fullDecomposeMatrix_constant layout X hq hn hX
  refine ⟨fun row column ↦ w (regularColumnIndex layout (Fin.cast hw row)) column, ?_,
    fun row column ↦ ?_, fun row column ↦ hwc _ column⟩
  · funext row column
    rw [hD, castMatrixRows_apply]
    simp only [regularDecomposeMatrix, hcorrected, hwq]
    rfl
  · rw [hbase, Int.toNat_natCast]
    exact hwb _ column

/-- A right action of one constant small matrix (the digits of a constant, such as the
finalization `G⁻¹(k)` of a coefficient output) on every block: its gain is `m β`, without the
ring dimension. -/
theorem Rel.mulRightConstant {q m m' : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat}
    (h : Rel sm A t c B) (d : ErrorMatrix n m m') (hd : CoeffBound d β)
    (hconstant : IsConstantMatrix d) :
    Rel sm (fun k j ↦ A k j * reduceMatrix q n m m' d) (fun j ↦ t j * reduceMatrix q n m m' d)
      (fun j ↦ c j * reduceMatrix q n m m' d) (m * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j * d, fun j ↦ ?_, fun j ↦ ?_⟩
  · have hej := coeffBound_iff_matrixNorm_le.mp (he j)
    have hdn := coeffBound_iff_matrixNorm_le.mp hd
    exact coeffBound_iff_matrixNorm_le.mpr ((matrixNorm_mul_right_constant_le hn hconstant).trans
      (Nat.mul_le_mul (Nat.mul_le_mul_left m hej) hdn))
  · beta_reduce
    rw [hc j, reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]

/-- A constant polynomial has no coefficient past the first. -/
theorem intCast_coeff_ne_zero (hn : 0 < n) (x : Int) {i : Fin n} (hi : i.val ≠ 0) :
    ((x : ErrorPoly n)).coeff i = 0 := by
  have h := intCast_mul_coeff x (1 : ErrorPoly n) i
  rw [mul_one] at h
  have h1 := Negacyclic.coeff_root_pow (R := Int) (n := n) hn ⟨0, hn⟩ i
  rw [show ((⟨0, hn⟩ : Fin n) : Nat) = 0 from rfl, pow_zero,
    if_neg (fun h ↦ hi (by rw [← h]))] at h1
  rw [h, h1, mul_zero]

set_option maxHeartbeats 1000000 in
/-- The finalization of a coefficient output: an encoding of `ν` against the full gadget `G` of an
exact layout, acted on by the decomposition `H'` of the constant `k` (rebased to its own modulus),
is an encoding of `k ν` against the gadget `1`; its error grows by `m (base / 2)`. -/
theorem Rel.finalizeCoefficient {q m : Nat} (hq1 : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm sp ν : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A (fun j ↦ reducePoly q n (ν * unitPair sp j) • G) c B) (k : Int)
    {K : ExactMatrix q n 1 1} (hK : K = matrixPolynomial [k])
    {H : ExactMatrix q n m 1} (hH : gadgetDecomposeRuns backend base digits K H)
    {bound : Nat} {H' : ExactMatrix q n m 1} (hH' : compactCenteredRebaseRuns bound H H') :
    Rel sm (fun a j ↦ matrixMul (A a j) H')
      (fun j ↦ reducePoly q n ((k : ErrorPoly n) * ν * unitPair sp j) • (1 : ExactMatrix q n 1 1))
      (fun j ↦ matrixMul (c j) H') (m * B * (base.toNat / 2)) := by
  have hHH' : H = H' := centeredRebaseRuns_unique (centeredRebaseRuns_self hq1 hn H) hH'.2.2.1
  subst hHH'
  have hconst : ∀ r c (i : Fin n), i.val ≠ 0 → (K r c).coeff i = 0 := by
    intro r c i hi
    rw [hK, matrixPolynomial_single, ← reducePoly_intCast, reducePoly_coeff_eq hq1 hn,
      intCast_coeff_ne_zero hn k hi, Int.cast_zero]
  obtain ⟨d, hd, hdb, hdc⟩ := gadgetDecomposeRuns_constant hq1 hn hH
    (fun l hl ↦ by rw [hlayout] at hl; cases hl; exact hexact) hconst
  have hact := h.mulRightConstant hn d hdb hdc
  rw [← hd] at hact
  have hrec := gadget_mul_decompose hq1 hn layout hlayout hexact hG hH
  refine hact.target_eq (funext fun j ↦ ?_)
  rw [Matrix.smul_mul, hrec, hK]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, Matrix.one_apply_eq,
    mul_one, map_mul, reducePoly_intCast]
  ring

set_option maxHeartbeats 1000000 in
/-- Public alignment: the right action by the digits `H` of `p g` and the runtime ModDown move a
public encoding of `α` with the native gadget `G` of `q = p q'` to one with the native gadget
`g'` of `q'`. -/
theorem align_down {q q' p m m' : Nat} (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {g' : ExactMatrix q' n 1 m'} {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {sm α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (ha : Rel sm A (fun j ↦ reducePoly q n (α * unitPair sm j) • G) c B)
    {scaled : ExactMatrix q n 1 m'} (hscaled : scaled = reducePoly q n ((p : Int) : ErrorPoly n) • g)
    {H : ExactMatrix q n m m'} (hH : gadgetDecomposeRuns backend base digits scaled H)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, Y j = matrixMul (c j) H) (hK : ∀ k j, K k j = matrixMul (A k j) H)
    {basis : List Nat} {c' : Fin 2 → ExactMatrix q' n 1 m'}
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hc' : ∀ j, blockModSwitchRuns basis 1 (Y j) (c' j))
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (K k j) (A' k j)) :
    Rel sm A' (fun j ↦ reducePoly q' n (α * unitPair sm j) • g') c'
      ((m * n * B * (base.toNat / 2) + (n + 2) * (p / 2)) / p) := by
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 hn hH
  have hact := ha.mulRight hn d hdb
  rw [← hd] at hact
  have hrec := gadget_mul_decompose hq1 hn layout hlayout hexact hG hH
  have ht : ∀ j, reducePoly q n (α * unitPair sm j) • G * H =
      scaleUpMatrix q p (reducePoly q' n (α * unitPair sm j) • g') := by
    intro j
    rw [Matrix.smul_mul, hrec, hscaled, smul_smul, ← map_mul,
      scaleUpMatrix_rebasedGadget hq hq' hn hg, mul_comm]
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  exact hact.modDown hn hq hq1 hq' hp hsm (A' := A') (c' := c') ht
    (fun k j ↦ by
      obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hA' k j)
      rw [hpq] at hr hrel
      exact ⟨r, hr, by rw [hrel, hK k j]; rfl⟩)
    (fun j ↦ by
      obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hc' j)
      rw [hpq] at hr hrel
      exact ⟨r, hr, by rw [hrel, hY j]; rfl⟩)

set_option maxHeartbeats 1000000 in
/-- Public multiplication: the stacked public encoding of `α` acting on the decomposition of the
private key `A_s`, plus `α` times the private row, encodes `α t_s` under the acted key. -/
theorem mul_public_gain {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} {K : Nat} (hα : SmulGain α K)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ca Ba)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (ca 0) (Aa 0 0) l01) (hl : concatRows l01 (Aa 1 0) l)
    (hr01 : concatRows (ca 1) (Aa 0 1) r01) (hr : concatRows r01 (Aa 1 1) r)
    (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {payload : ExactMatrix q n 1 1} (hpayload : payload = fun _ _ ↦ reducePoly q n α)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) payload) :
    Rel sm A' (fun j ↦ reducePoly q n α • ts j) c' ((m + m) * n * Ba * bound + K * Bs) := by
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hDD' : ∀ j, D j = D' j := fun j ↦
    centeredRebaseRuns_unique (hD j).2.2.1 (centeredRebaseRuns_self hq1 hn (D' j))
  have hact := ha.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hYe : ∀ j, Y j = pairMul ca (D j) := by
    intro j
    rw [sliceMatrix_row (rows := 1 + 1 + 1) 0 (hY j), ← pair_eta ca]
    exact (hrows j).1
  have hAe : ∀ k j, A' k j = pairMul (Aa k) (D j) := by
    intro k j
    match k with
    | 0 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 1 (hA0 j), ← pair_eta (Aa 0)]
      exact (hrows j).2.1
    | 1 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 2 (hA1 j), ← pair_eta (Aa 1)]
      exact (hrows j).2.2
  have ht : ∀ j, pairMul (fun i ↦ reducePoly q n (α * unitPair sm i) • G) (D j) =
      reducePoly q n α • (reducePoly q n sm • As 0 j + As 1 j) := by
    intro j
    rw [hDD' j, gadget_pairMul_decompose hq1 hn layout hlayout hexact hG (hD' j),
      (concatRows_two (hXs j)).1, (concatRows_two (hXs j)).2]
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one,
      smul_add, smul_smul, mul_comm]
  obtain ⟨ea, hea, hca⟩ := hact
  obtain ⟨es, hes, hcs⟩ := hs
  beta_reduce at hca
  refine ⟨fun j ↦ ea j + α • es j, fun j ↦ coeffBound_add (hea j) (hα (hes j)),
    fun j ↦ ?_⟩
  beta_reduce
  rw [hc' j, hYe j, hca j, ht j, matrixMulScalarRight_smul, hcs j, hpayload, hAe 0 j, hAe 1 j,
    reduceMatrix_add, reduceMatrix_smul]
  simp only [smul_add, smul_sub]
  abel

/-- `mul_public_gain` for a public plaintext of coefficient bound `1`. -/
theorem mul_public {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm α : ErrorPoly n} (hα : polyNorm α ≤ 1)
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m} {Ba : Nat}
    (ha : Rel sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ca Ba)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As ts cs Bs)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (ca 0) (Aa 0 0) l01) (hl : concatRows l01 (Aa 1 0) l)
    (hr01 : concatRows (ca 1) (Aa 0 1) r01) (hr : concatRows r01 (Aa 1 1) r)
    (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {payload : ExactMatrix q n 1 1} (hpayload : payload = fun _ _ ↦ reducePoly q n α)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) payload) :
    Rel sm A' (fun j ↦ reducePoly q n α • ts j) c' ((m + m) * n * Ba * bound + n * 1 * Bs) :=
  mul_public_gain hn hq1 layout hlayout hexact hG (smulGain_of_polyNorm hn hα) ha hs hl01 hl hr01 hr hboth hXs hD' hD hpayload hY hA0 hA1 hc'

end MxxExponentLut
