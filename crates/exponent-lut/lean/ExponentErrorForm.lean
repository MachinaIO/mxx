import ExponentPublic
import ExponentDecode

/-!
# The encoding relation with an explicit error

`RelE sm A t c e` is `Rel` with the error `e` named: `c_j = s₀ A_{0j} + A_{1j} - t_j + e_j`. An
average-case analysis needs the error of every wire as an explicit function of the run, not only
a bound, so the combinators here return the error they produce:

* a stacked right action by `d` gives the error `pairMul e d`;
* sums add the errors;
* the runtime ModDown gives an error `v` with `p v = e - r_c + s₀ r₀ + r₁`, where `r_c`, `r₀`
  and `r₁` are the centered remainders of the row and of the key modulo the dropped factor `p`.

`RelE.toRel` recovers the bounded relation.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- `c_j = s₀ A_{0j} + A_{1j} - t_j + e_j` for the named error `e`. -/
def RelE {q m : Nat} (sm : ErrorPoly n) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (t c : Fin 2 → ExactMatrix q n 1 m) (e : Fin 2 → ErrorMatrix n 1 m) : Prop :=
  ∀ j, c j = reducePoly q n sm • A 0 j + A 1 j - t j + reduceMatrix q n 1 m (e j)

/-- The centered remainder modulo `p` of every coefficient of `x`: the residue that the runtime
ModDown removes. -/
noncomputable def downResidue {q rows columns : Nat} (p : Nat) (x : ExactMatrix q n rows columns) :
    ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun i ↦ rnsCentered p (Int.ofNat ((x row column).coeff i).val)

/-- `blockModSwitchRuns_scaleUp` with its remainder named: `scaleUp` of the output is the input
minus `downResidue (q / q') input`. -/
theorem blockModSwitchRuns_residue (hn : 0 < n) {q q' rows columns : Nat} {basis : List Nat}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix q' n rows columns}
    (h : blockModSwitchRuns basis 1 input output) :
    CoeffBound (downResidue (q / q') input) ((q / q') / 2) ∧
      scaleUpMatrix q (q / q') output =
        input - reduceMatrix q n rows columns (downResidue (q / q') input) := by
  obtain ⟨hbasis, hne, _, _, hq'1, hlt, _, hdne, hprod, _, _, hout⟩ := h
  set dropped := basis.filter (fun prime ↦ q' % prime != 0)
  have hp : q / q' = dropped.prod := by
    rw [← hprod, Nat.mul_div_cancel_left _ (by omega)]
  set p := q / q' with hpdef
  have hq : q = p * q' := by rw [hp, Nat.mul_comm, hprod]
  have hpgt : 1 < p := by
    by_contra hle
    rcases (show p = 0 ∨ p = 1 by omega) with h0 | h1
    · rw [h0, zero_mul] at hq; omega
    · rw [h1, one_mul] at hq; omega
  have hq1 : 1 < q := by nlinarith
  refine ⟨?_, ?_⟩
  · intro row column i
    rw [downResidue, intPoly_coeff hn]
    exact rnsCentered_natAbs_le (by omega) _
  · funext row column
    have hentry := hout row column
    simp only at hentry
    rw [← hp] at hentry
    simp only [scaleUpMatrix, Matrix.sub_apply, reduceMatrix, downResidue]
    rw [hentry, polynomialOfCoefficients_eq_reduce, scaleUp_reducePoly hq hq'1 hn]
    rw [show ∀ r : ErrorPoly n, input row column - reducePoly q n r =
        reducePoly q n (canonicalLift (input row column) - r) from fun r ↦ by
      rw [map_sub, reducePoly_canonicalLift hq1 hn]]
    rw [reducePoly_eq_iff hq1 hn]
    intro i
    rw [intCast_mul_coeff, intPoly_coeff hn, Negacyclic.coeff_sub, canonicalLift_coeff hn,
      intPoly_coeff hn]
    simpa only [one_mul, Int.ofNat_eq_natCast] using
      blockModSwitch_coefficient hq hpgt (((input row column).coeff i).val : Int)

/-- A right action by a small matrix acts on the error. -/
theorem RelE.mulRight {q m m' : Nat} {sm : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e) (d : ErrorMatrix n m m') :
    RelE sm (fun k j ↦ A k j * reduceMatrix q n m m' d) (fun j ↦ t j * reduceMatrix q n m m' d)
      (fun j ↦ c j * reduceMatrix q n m m' d) (fun j ↦ e j * d) := by
  intro j
  beta_reduce
  rw [h j, reduceMatrix_mul, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul]

/-- Coefficientwise integer division by `p`: the error after a ModDown, as a function of its
numerator. -/
noncomputable def divInt {rows columns : Nat} (p : ℤ) (x : ErrorMatrix n rows columns) :
    ErrorMatrix n rows columns :=
  fun r c ↦ intPoly fun k ↦ (x r c).coeff k / p

theorem divInt_eq (hn : 0 < n) {rows columns : Nat} {p : ℤ} (hp : p ≠ 0)
    {x v : ErrorMatrix n rows columns} (h : ∀ r c, x r c = (p : ErrorPoly n) * v r c) :
    divInt p x = v := by
  funext r c
  apply negacyclic_ext hn
  intro k
  rw [divInt, intPoly_coeff hn, h r c, intCast_mul_coeff, Int.mul_ediv_cancel_left _ hp]

/-- The row of an encoding with mask `s₀`, key `A`, target `t` and error `e`. -/
noncomputable def encRow {q m : Nat} (sm : ErrorPoly n) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (t : Fin 2 → ExactMatrix q n 1 m) (e : Fin 2 → ErrorMatrix n 1 m) : Fin 2 → ExactMatrix q n 1 m :=
  fun j ↦ reducePoly q n sm • A 0 j + A 1 j - t j + reduceMatrix q n 1 m (e j)

theorem RelE.eq_encRow {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e) :
    c = encRow sm A t e :=
  funext h

section ErrorForm

variable {q m : Nat} {sm : ErrorPoly n}

theorem RelE.toRel {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m} {B : Nat} (h : RelE sm A t c e)
    (he : ∀ j, CoeffBound (e j) B) : Rel sm A t c B :=
  ⟨e, he, h⟩

theorem Rel.exists_relE {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B) :
    ∃ e, RelE sm A t c e ∧ ∀ j, CoeffBound (e j) B := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e, hc, he⟩

theorem RelE.target_eq {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t t' c : Fin 2 → ExactMatrix q n 1 m} {e : Fin 2 → ErrorMatrix n 1 m}
    (h : RelE sm A t c e) (ht : t = t') : RelE sm A t' c e :=
  ht ▸ h

/-- A small additive term of the target moves into the error. -/
theorem RelE.absorb {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e f : Fin 2 → ErrorMatrix n 1 m}
    (h : RelE sm A (fun j ↦ t j + reduceMatrix q n 1 m (f j)) c e) :
    RelE sm A t c (fun j ↦ e j - f j) := by
  intro j
  simp only [h j, reduceMatrix_sub]
  abel

/-- Relations with the same mask add, and so do their errors. -/
theorem RelE.add {A A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t t' c c' : Fin 2 → ExactMatrix q n 1 m} {e e' : Fin 2 → ErrorMatrix n 1 m}
    (h : RelE sm A t c e) (h' : RelE sm A' t' c' e') :
    RelE sm (A + A') (t + t') (c + c') (e + e') := by
  intro j
  simp only [Pi.add_apply, h j, h' j, reduceMatrix_add, smul_add]
  abel

theorem RelE.zero : RelE (q := q) (m := m) sm 0 0 0 0 := by
  intro j
  funext row column
  simp [reduceMatrix]

theorem RelE.sum {ι : Type} [DecidableEq ι] (s : Finset ι)
    {A : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : ι → Fin 2 → ExactMatrix q n 1 m}
    {e : ι → Fin 2 → ErrorMatrix n 1 m} (h : ∀ i ∈ s, RelE sm (A i) (t i) (c i) (e i)) :
    RelE sm (∑ i ∈ s, A i) (∑ i ∈ s, t i) (∑ i ∈ s, c i) (∑ i ∈ s, e i) := by
  induction s using Finset.induction_on with
  | empty => simpa using (RelE.zero (q := q) (m := m) (sm := sm))
  | @insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact (h i (Finset.mem_insert_self i s)).add (ih fun j hj ↦ h j (Finset.mem_insert_of_mem hj))

/-- A stacked right action acts on the error. -/
theorem RelE.action {m' : Nat} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e)
    (d : Fin 2 → ErrorMatrix n (m + m) m') :
    RelE sm (fun k j ↦ pairMul (A k) (reduceMatrix q n (m + m) m' (d j)))
      (fun j ↦ pairMul t (reduceMatrix q n (m + m) m' (d j)))
      (fun j ↦ pairMul c (reduceMatrix q n (m + m) m' (d j))) (fun j ↦ pairMul e (d j)) := by
  intro j
  have hcfun : c = reducePoly q n sm • A 0 + A 1 - t + fun i ↦ reduceMatrix q n 1 m (e i) := by
    funext i
    simp [h i]
  simp only []
  rw [hcfun, reduceMatrix_pairMul, pairMul_add, pairMul_sub, pairMul_add, pairMul_smul]

/-- The runtime ModDown of the row and the key divides a target `scaleUp T` by the dropped factor
`p`. With the remainders `r k j` of the key and `rc j` of the row, the new error `v` satisfies
`p v_j = e_j - rc_j + s₀ r_{0j} + r_{1j}`. -/
theorem RelE.modDown (hn : 0 < n) {p q' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e)
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m} {T c' : Fin 2 → ExactMatrix q' n 1 m}
    (ht : ∀ j, t j = scaleUpMatrix q p (T j))
    {r : Fin 2 → Fin 2 → ErrorMatrix n 1 m}
    (hA : ∀ k j, scaleUpMatrix q p (A' k j) = A k j - reduceMatrix q n 1 m (r k j))
    {rc : Fin 2 → ErrorMatrix n 1 m}
    (hc : ∀ j, scaleUpMatrix q p (c' j) = c j - reduceMatrix q n 1 m (rc j)) :
    ∃ v : Fin 2 → ErrorMatrix n 1 m, RelE sm A' T c' v ∧
      ∀ j row column, e j row column - rc j row column + sm * r 0 j row column + r 1 j row column =
        ((p : Int) : ErrorPoly n) * v j row column := by
  have hdiv : ∀ j, ∃ v : ErrorMatrix n 1 m,
      (∀ row column, e j row column - rc j row column + sm * r 0 j row column +
        r 1 j row column = ((p : Int) : ErrorPoly n) * v row column) ∧
      c' j = reducePoly q' n sm • A' 0 j + A' 1 j - T j + reduceMatrix q' n 1 m v := by
    intro j
    let w : ErrorMatrix n 1 m := e j - rc j + sm • r 0 j + r 1 j
    have hscale : ∀ row column, scaleUp q p
        ((c' j - (reducePoly q' n sm • A' 0 j + A' 1 j - T j)) row column) =
        reducePoly q n (w row column) := by
      intro row column
      have hc' := congrFun (congrFun (hc j) row) column
      have h0 := congrFun (congrFun (hA 0 j) row) column
      have h1 := congrFun (congrFun (hA 1 j) row) column
      have ht' := congrFun (congrFun (ht j) row) column
      have hr := congrFun (congrFun (h j) row) column
      simp only [scaleUpMatrix, Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply,
        smul_eq_mul, reduceMatrix] at hc' h0 h1 ht' hr
      simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      rw [scaleUp_sub hq hq' hn, scaleUp_sub hq hq' hn, scaleUp_add hq hq' hn,
        scaleUp_reducePoly_mul hq hq' hn, hc', h0, h1, ← ht', hr]
      simp only [w, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, map_add,
        map_sub, map_mul]
      ring
    choose v hv using fun row column ↦ scaleUp_eq_reducePoly hq hq1 hq' hn (hscale row column)
    refine ⟨fun row column ↦ v row column, fun row column ↦ ?_, ?_⟩
    · have := (hv row column).1
      simp only [w, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul] at this
      exact this
    · funext row column
      have := (hv row column).2
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
        reduceMatrix] at this ⊢
      linear_combination this
  choose v hv using hdiv
  exact ⟨v, fun j ↦ (hv j).2, fun j ↦ (hv j).1⟩

/-- `RelE.modDown` for the runtime ModDown of the row and the key: the new error `v` satisfies
`p v_j = e_j - ρ(c_j) + s₀ ρ(A_{0j}) + ρ(A_{1j})` for `ρ = downResidue p`. -/
theorem RelE.modDownRuns (hn : 0 < n) {p q' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e)
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m} {T c' : Fin 2 → ExactMatrix q' n 1 m}
    (ht : ∀ j, t j = scaleUpMatrix q p (T j)) {basis : List Nat}
    (hA : ∀ k j, blockModSwitchRuns basis 1 (A k j) (A' k j))
    (hc : ∀ j, blockModSwitchRuns basis 1 (c j) (c' j)) :
    ∃ v : Fin 2 → ErrorMatrix n 1 m, RelE sm A' T c' v ∧
      ∀ j row column, e j row column - downResidue p (c j) row column +
        sm * downResidue p (A 0 j) row column + downResidue p (A 1 j) row column =
        ((p : Int) : ErrorPoly n) * v j row column := by
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  exact h.modDown hn hq hq1 hq' ht (r := fun k j ↦ downResidue p (A k j))
    (fun k j ↦ by have := (blockModSwitchRuns_residue hn (hA k j)).2; rw [hpq] at this; exact this)
    (rc := fun j ↦ downResidue p (c j))
    (fun j ↦ by have := (blockModSwitchRuns_residue hn (hc j)).2; rw [hpq] at this; exact this)

/-- An odd substitution with index coprime to `n` acts on the relation, its mask and its error. -/
theorem RelE.substitute {index : Nat} (hodd : Odd index)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m} (h : RelE sm A t c e) :
    RelE (substitution index hodd sm) (fun k j ↦ substituteMatrix index hodd (A k j))
      (fun j ↦ substituteMatrix index hodd (t j)) (fun j ↦ substituteMatrix index hodd (c j))
      (fun j ↦ substituteMatrix index hodd (e j)) := by
  intro j
  funext row column
  simp only [substituteMatrix, h j, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
    smul_eq_mul, reduceMatrix, map_add, map_sub, map_mul, reducePoly_substitution]

/-- `Rel.lutBranch` with its error named: the helper error, the routed error, and the residues
`r` of the ModDown of the routed key `M`, the first weighted by the substituted mask `σ s`. -/
theorem RelE.lutBranch (hn : 0 < n) {p q' : Nat} (hq : q = p * q') (hq' : 1 < q')
    {smσ : ErrorPoly n} {M K : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {tc X th Y : Fin 2 → ExactMatrix q n 1 m} {ec eh : Fin 2 → ErrorMatrix n 1 m}
    (hc : RelE smσ M tc X ec) {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m}
    {r : Fin 2 → Fin 2 → ErrorMatrix n 1 m}
    (hM : ∀ k j, scaleUpMatrix q p (Asw k j) = M k j - reduceMatrix q n 1 m (r k j))
    (hh : RelE sm K th Y eh)
    (hth : ∀ j, th j = scaleUpMatrix q p (reducePoly q' n smσ • Asw 0 j + Asw 1 j)) :
    RelE sm K tc (Y + X) (fun j ↦ eh j + ec j + (smσ • r 0 j + r 1 j)) := by
  have hsplit : ∀ j, reducePoly q n smσ • M 0 j + M 1 j =
      th j + reduceMatrix q n 1 m (smσ • r 0 j + r 1 j) := by
    intro j
    rw [hth j]
    funext row column
    have h0' := congrFun (congrFun (hM 0 j) row) column
    have h1' := congrFun (congrFun (hM 1 j) row) column
    simp only [scaleUpMatrix, Matrix.sub_apply, reduceMatrix] at h0' h1'
    simp only [scaleUpMatrix, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix,
      map_add, map_mul]
    rw [scaleUp_add hq hq' hn, scaleUp_reducePoly_mul hq hq' hn, h0', h1']
    ring
  intro j
  rw [Pi.add_apply, hh j, hc j]
  rw [show reducePoly q n smσ • M 0 j + M 1 j - tc j + reduceMatrix q n 1 m (ec j) =
      (reducePoly q n smσ • M 0 j + M 1 j) - tc j + reduceMatrix q n 1 m (ec j) from rfl, hsplit j]
  simp only [reduceMatrix_add]
  abel

end ErrorForm

section Operations

variable {q q' p m m' : Nat} {sm : ErrorPoly n}

set_option maxHeartbeats 1000000 in
/-- `align_down` with its error named: the member error acted on by the digits `d` of `p g`, minus
the row residue, plus the mask-weighted key residues, divided by `p`. -/
theorem align_downE (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {g' : ExactMatrix q' n 1 m'} {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {α : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m}
    {e : Fin 2 → ErrorMatrix n 1 m}
    (ha : RelE sm A (fun j ↦ reducePoly q n (α * unitPair sm j) • G) c e)
    {scaled : ExactMatrix q n 1 m'} (hscaled : scaled = reducePoly q n ((p : Int) : ErrorPoly n) • g)
    {H : ExactMatrix q n m m'} (hH : gadgetDecomposeRuns backend base digits scaled H)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, Y j = matrixMul (c j) H) (hK : ∀ k j, K k j = matrixMul (A k j) H)
    {basis : List Nat} {c' : Fin 2 → ExactMatrix q' n 1 m'}
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hc' : ∀ j, blockModSwitchRuns basis 1 (Y j) (c' j))
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (K k j) (A' k j)) :
    ∃ d : ErrorMatrix n m m', H = reduceMatrix q n m m' d ∧ CoeffBound d (base.toNat / 2) ∧
      ∃ v : Fin 2 → ErrorMatrix n 1 m',
        RelE sm A' (fun j ↦ reducePoly q' n (α * unitPair sm j) • g') c' v ∧
        ∀ j row column, (e j * d) row column - downResidue p (Y j) row column +
          sm * downResidue p (K 0 j) row column + downResidue p (K 1 j) row column =
          ((p : Int) : ErrorPoly n) * v j row column := by
  obtain ⟨d, hd, hdb⟩ := gadgetDecomposeRuns_small hq1 hn hH
  have hact := ha.mulRight d
  rw [← hd] at hact
  have hrec := gadget_mul_decompose hq1 hn layout hlayout hexact hG hH
  have ht : ∀ j, reducePoly q n (α * unitPair sm j) • G * H =
      scaleUpMatrix q p (reducePoly q' n (α * unitPair sm j) • g') := by
    intro j
    rw [Matrix.smul_mul, hrec, hscaled, smul_smul, ← map_mul,
      scaleUpMatrix_rebasedGadget hq hq' hn hg, mul_comm]
  have hYe : ∀ j, Y j = c j * H := fun j ↦ hY j
  have hKe : ∀ k j, K k j = A k j * H := fun k j ↦ hK k j
  simp only [← hYe, ← hKe] at hact
  have hc'' : ∀ j, blockModSwitchRuns basis 1 (Y j) (c' j) := hc'
  obtain ⟨v, hv, hvp⟩ := hact.modDownRuns hn hq hq1 hq' (A' := A') (c' := c') ht hA' hc''
  exact ⟨d, hd, hdb, v, hv, hvp⟩

set_option maxHeartbeats 1000000 in
/-- `mul_public_gain` with its error named: the member error acted on by the digits `d` of the
private key, plus `α` times the private error. -/
theorem mul_publicE {bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {α : ErrorPoly n}
    {Aa : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {ca : Fin 2 → ExactMatrix q n 1 m}
    {ea : Fin 2 → ErrorMatrix n 1 m}
    (ha : RelE sm Aa (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ca ea)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {ts cs : Fin 2 → ExactMatrix q n 1 m'}
    {es : Fin 2 → ErrorMatrix n 1 m'} (hs : RelE sm As ts cs es)
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
    ∃ d : Fin 2 → ErrorMatrix n (m + m) m', (∀ j, D j = reduceMatrix q n (m + m) m' (d j)) ∧
      (∀ j, CoeffBound (d j) bound) ∧ (∀ k j, A' k j = pairMul (Aa k) (D j)) ∧
      (∀ j, Y j = pairMul ca (D j)) ∧
      RelE sm A' (fun j ↦ reducePoly q n α • ts j) c' (fun j ↦ pairMul ea (d j) + α • es j) := by
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hDD' : ∀ j, D j = D' j := fun j ↦
    centeredRebaseRuns_unique (hD j).2.2.1 (centeredRebaseRuns_self hq1 hn (D' j))
  have hact := ha.action d
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
  refine ⟨d, hDd, fun j ↦ (hd j).2, hAe, hYe, fun j ↦ ?_⟩
  have hca := hact j
  beta_reduce at hca ⊢
  rw [hc' j, hYe j, hca, ht j, matrixMulScalarRight_smul, hs j, hpayload, hAe 0 j, hAe 1 j,
    reduceMatrix_add, reduceMatrix_smul]
  simp only [smul_add, smul_sub]
  abel

set_option maxHeartbeats 1000000 in
/-- `mul_private_down_gain` with its error named: the state error acted on by the digits `d` of
the RHS, minus the row residue, plus the mask-weighted key residues, divided by `p`, plus `μ`
times the RHS error. -/
theorem mul_private_downE {bound : Nat} (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {e : Fin 2 → ErrorMatrix n 1 m}
    (henc : RelE sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c e)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (c 0) (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows (c 1) (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {top bot : Fin 2 → ExactMatrix q' n 1 m'} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {T : Fin 2 → ExactMatrix q' n 1 m'} {eB : Fin 2 → ErrorMatrix n 1 m'}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j = T j + reduceMatrix q' n 1 m' (eB j))
    {X : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {c' : Fin 2 → ExactMatrix q' n 1 m'}
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hc' : ∀ j, blockModSwitchRuns basis 1 (X j) (c' j))
    (hA' : ∀ k j, blockModSwitchRuns basis 1 (M k j) (A' k j)) :
    ∃ d : Fin 2 → ErrorMatrix n (m + m) m', (∀ j, D j = reduceMatrix q n (m + m) m' (d j)) ∧
      (∀ j, CoeffBound (d j) bound) ∧ (∀ k j, M k j = pairMul (A k) (D j)) ∧
      (∀ j, X j = pairMul c (D j)) ∧ ∃ v : Fin 2 → ErrorMatrix n 1 m',
        RelE sm A' (fun j ↦ reducePoly q' n μ • T j) c' (fun j ↦ v j - μ • eB j) ∧
        ∀ j row column, pairMul e (d j) row column - downResidue p (X j) row column +
          sm * downResidue p (M 0 j) row column + downResidue p (M 1 j) row column =
          ((p : Int) : ErrorPoly n) * v j row column := by
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := henc.action d
  simp only [← hDd] at hact
  have hXe : ∀ j, X j = pairMul c (D j) := by
    intro j
    rw [sliceMatrix_row (rows := 1 + 1 + 1) 0 (hX j), ← pair_eta c]
    exact (hrows j).1
  have hMe : ∀ k j, M k j = pairMul (A k) (D j) := by
    intro k j
    match k with
    | 0 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 1 (hM0 j), ← pair_eta (A 0)]
      exact (hrows j).2.1
    | 1 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 2 (hM1 j), ← pair_eta (A 1)]
      exact (hrows j).2.2
  have ht : ∀ j, pairMul (fun i ↦ reducePoly q n ((p : Int) * μ * unitPair sp i) • g) (D j) =
      scaleUpMatrix q p (reducePoly q' n μ • T j + reduceMatrix q' n 1 m' (μ • eB j)) := by
    intro j
    have hgad := gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg (hD' j) (hD j)
      (fun i ↦ μ * unitPair sp i)
    simp only [← mul_assoc] at hgad
    rw [hgad, (concatRows_two (hXb j)).1, (concatRows_two (hXb j)).2, reduceMatrix_smul,
      ← smul_add, ← hrhs j]
    congr 1
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one, smul_add,
      smul_smul]
  have hact' : RelE sm (fun k j ↦ M k j)
      (fun j ↦ pairMul (fun i ↦ reducePoly q n ((p : Int) * μ * unitPair sp i) • g) (D j))
      (fun j ↦ X j) (fun j ↦ pairMul e (d j)) := by
    intro j
    beta_reduce
    rw [hXe j, hMe 0 j, hMe 1 j]
    exact hact j
  obtain ⟨v, hv, hvp⟩ := hact'.modDownRuns hn hq hq1 hq' (A' := A') (c' := c') ht hA' hc'
  exact ⟨d, hDd, fun j ↦ (hd j).2, hMe, hXe, v, hv.absorb, hvp⟩

set_option maxHeartbeats 1000000 in
/-- `lut_branch` with its error named: the helper and routed errors after their actions, and the
residues of the ModDown of the routed key, the first weighted by the substituted mask. -/
theorem lut_branchE {q q' p m m' log bound : Nat} (hn : 0 < n) (hpow : n = 2 ^ log)
    (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m}
    {ec : Fin 2 → ErrorMatrix n 1 m}
    (henc : RelE sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c ec)
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {bsw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (bsw 0 j) (bsw 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {eh : Fin 2 → ErrorMatrix n 1 m}
    (hodd : Odd index.toNat)
    (hhelp : RelE sm Ad (fun j ↦ reducePoly q n ((p : Int) *
      unitPair (substitution index.toNat hodd sm) j) • g) h eh)
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    ∃ d ε : Fin 2 → ErrorMatrix n (m + m) m',
      (∀ j, D j = reduceMatrix q n (m + m) m' (d j) ∧ CoeffBound (d j) bound) ∧
      (∀ j, E j = reduceMatrix q n (m + m) m' (ε j) ∧ CoeffBound (ε j) bound) ∧
      (∀ k j, M k j = pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j)) ∧
      (∀ k j, K k j = pairMul (Ad k) (E j)) ∧
      RelE sm K (fun j ↦ scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd μ) •
        (reducePoly q' n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j)))
      (fun j ↦ Y j + X j)
      (fun j ↦ pairMul eh (ε j) + pairMul (fun i ↦ substituteMatrix index.toNat hodd (ec i)) (d j) +
        (substitution index.toNat hodd sm • downResidue p (M 0 j) + downResidue p (M 1 j))) := by
  have hcop : n.Coprime index.toNat := by rw [hpow]; exact coprime_of_odd_pow_two hodd
  -- The routed blocks.
  have hroute : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = substituteMatrix index.toNat hodd x := by
    intro x y hxy
    obtain ⟨_, hy⟩ := ringAutomorphismRuns_eq hq1 hn hxy
    exact hy
  have hsub := henc.substitute hodd
  -- Rows of the stacked products.
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  have hhrows := fun j ↦ stacked_row hhl01 hhl hhr01 hhr hH (E j)
  have hslice : ∀ {rows cols : Nat} {x : ExactMatrix q n rows cols} {out : ExactMatrix q n 1 cols}
      (k : Fin rows), sliceMatrix x k.val (k.val + 1) 0 cols out → out = rowOf x k :=
    fun k hk ↦ sliceMatrix_row k hk
  have hσc : (fun i ↦ substituteMatrix index.toNat hodd (c i)) = ![rc0, rc1] := by
    rw [hroute hr0, hroute hr3]; exact (pair_eta _).symm
  have hσA0 : (fun i ↦ substituteMatrix index.toNat hodd (A 0 i)) = ![rA00, rA01] := by
    rw [hroute hr1, hroute hr4]; exact (pair_eta _).symm
  have hσA1 : (fun i ↦ substituteMatrix index.toNat hodd (A 1 i)) = ![rA10, rA11] := by
    rw [hroute hr2, hroute hr5]; exact (pair_eta _).symm
  -- The switching action on the routed encoding.
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := hsub.action d
  simp only [← hDd] at hact
  have hXe : ∀ j, X j = pairMul (fun i ↦ substituteMatrix index.toNat hodd (c i)) (D j) := by
    intro j
    rw [hslice (rows := 1 + 1 + 1) 0 (hX j), hσc]
    exact (hrows j).1
  have hMe : ∀ k j, M k j = pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j) := by
    intro k j
    match k with
    | 0 =>
      rw [hslice (rows := 1 + 1 + 1) 1 (hM0 j), hσA0]
      exact (hrows j).2.1
    | 1 =>
      rw [hslice (rows := 1 + 1 + 1) 2 (hM1 j), hσA1]
      exact (hrows j).2.2
  -- The helper action.
  choose ε he using fun j ↦ (hE j).2.2.2
  have hEe : ∀ j, E j = reduceMatrix q n (m + m) m' (ε j) := fun j ↦ (he j).1
  have hhact := hhelp.action ε
  simp only [← hEe] at hhact
  have hYe : ∀ j, Y j = pairMul h (E j) := by
    intro j
    rw [hslice (rows := 1 + 1 + 1) 0 (hY j), ← pair_eta h]
    exact (hhrows j).1
  have hKe : ∀ k j, K k j = pairMul (Ad k) (E j) := by
    intro k j
    match k with
    | 0 =>
      rw [hslice (rows := 1 + 1 + 1) 1 (hK0 j), ← pair_eta (Ad 0)]
      exact (hhrows j).2.1
    | 1 =>
      rw [hslice (rows := 1 + 1 + 1) 2 (hK1 j), ← pair_eta (Ad 1)]
      exact (hhrows j).2.2
  -- The routed target after the switching action.
  have htc : ∀ j, pairMul (fun i ↦ substituteMatrix index.toNat hodd
      (reducePoly q n ((p : Int) * μ * unitPair sp i) • g)) (D j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd μ) •
        (reducePoly q' n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j)) := by
    intro j
    simp only [substituteMatrix_target hn hg' hg]
    have hgad := gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg (hD' j) (hD j)
      (fun i ↦ substitution index.toNat hodd μ * substitution index.toNat hodd (unitPair sp i))
    simp only [← mul_assoc] at hgad
    rw [hgad, (concatRows_two (hXb j)).1, (concatRows_two (hXb j)).2]
    congr 1
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_one, map_mul, mul_one,
      smul_add, smul_smul]
  -- The helper target after its action.
  have hth : ∀ j, pairMul (fun i ↦ reducePoly q n ((p : Int) *
      unitPair (substitution index.toNat hodd sm) i) • g) (E j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd sm) • Asw 0 j + Asw 1 j) := by
    intro j
    rw [gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg (hE' j) (hE j),
      (concatRows_two (hZ j)).1, (concatRows_two (hZ j)).2]
    simp [unitPair]
  -- ModDown of the switched key.
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  have hsw : ∀ k j, scaleUpMatrix q p (Asw k j) =
      pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j) -
        reduceMatrix q n 1 m' (downResidue p (M k j)) := by
    intro k j
    have hrel := (blockModSwitchRuns_residue hn (hAsw k j)).2
    rw [hpq] at hrel
    rw [hrel, hMe k j]
  have hbranch := RelE.lutBranch hn hq hq' hact hsw hhact hth
  simp only [htc] at hbranch
  have hYX : (fun j ↦ Y j + X j) = (fun j ↦ pairMul h (E j)) +
      fun j ↦ pairMul (fun i ↦ substituteMatrix index.toNat hodd (c i)) (D j) := by
    funext j
    rw [Pi.add_apply, hYe j, hXe j]
  have hKfun : K = fun k j ↦ pairMul (Ad k) (E j) := by
    funext k j
    exact hKe k j
  rw [hYX, hKfun]
  exact ⟨d, ε, hd, he, hMe, fun k j ↦ rfl, hbranch⟩

set_option maxHeartbeats 1000000 in
/-- `lut_finish` with its error named: the summed branch error, minus the row residue, plus the
mask-weighted key residues, divided by `p`, minus the switching errors times `σ_i(μ)`. -/
theorem lut_finishE {q q' p m' W : Nat} (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') {sm sp μ ν Δ' : ErrorPoly n}
    (σ : Fin W → ErrorPoly n → ErrorPoly n)
    {bsw : Fin W → Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {Ksum : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rsum : Fin 2 → ExactMatrix q n 1 m'}
    {es : Fin 2 → ErrorMatrix n 1 m'}
    (hsum : RelE sm Ksum (fun j ↦ ∑ i, scaleUpMatrix q p (reducePoly q' n (σ i μ) •
      (reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j))) Rsum es)
    {g' : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {eB : Fin W → Fin 2 → ErrorMatrix n 1 m'}
    (hswitch : ∀ i j, reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j =
      (C i * reducePoly q' n (Δ' * unitPair sp j)) • g' + reduceMatrix q' n 1 m' (eB i j))
    (hwalsh : ∑ i, reducePoly q' n (σ i μ) * C i = reducePoly q' n ν) {basis : List Nat}
    {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (Ksum k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (Rsum j) (R' j)) :
    ∃ v : Fin 2 → ErrorMatrix n 1 m',
      RelE sm K' (fun j ↦ reducePoly q' n (Δ' * ν * unitPair sp j) • g') R'
        (fun j ↦ v j - ∑ i, σ i μ • eB i j) ∧
      ∀ j row column, es j row column - downResidue p (Rsum j) row column +
        sm * downResidue p (Ksum 0 j) row column + downResidue p (Ksum 1 j) row column =
        ((p : Int) : ErrorPoly n) * v j row column := by
  -- The summed target.
  let f : Fin 2 → ErrorMatrix n 1 m' := fun j ↦ ∑ i, σ i μ • eB i j
  have htarget : ∀ j, (∑ i, scaleUpMatrix q p (reducePoly q' n (σ i μ) •
      (reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j))) =
      scaleUpMatrix q p (reducePoly q' n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' n 1 m' (f j)) := by
    intro j
    simp only [hswitch]
    funext row column
    simp only [scaleUpMatrix, Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      reduceMatrix, f]
    rw [← scaleUp_sum hq hq' hn]
    congr 1
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin W)), reducePoly q' n (σ i μ) *
        (C i * reducePoly q' n (Δ' * unitPair sp j) * g' row column +
          reducePoly q' n (eB i j row column)) =
        (reducePoly q' n (σ i μ) * C i) * (reducePoly q' n (Δ' * unitPair sp j) * g' row column) +
          reducePoly q' n (σ i μ * eB i j row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh,
      map_sum]
    simp only [map_mul]
    ring
  have hsum' : RelE sm Ksum (fun j ↦ scaleUpMatrix q p
      (reducePoly q' n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' n 1 m' (f j))) Rsum es := by
    intro j
    rw [hsum j]
    simp only []
    rw [htarget j]
  obtain ⟨v, hv, hvp⟩ := hsum'.modDownRuns hn hq hq1 hq' (A' := K') (c' := R') (fun j ↦ rfl) hK' hR'
  exact ⟨v, hv.absorb, hvp⟩

set_option maxHeartbeats 1000000 in
/-- `Rel.finalizeCoefficient` with its error named: the error acted on by the digits `d` of the
constant `k`, each a constant polynomial. -/
theorem RelE.finalizeCoefficient {q m : Nat} (hq1 : 1 < q) (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sp ν : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {e : Fin 2 → ErrorMatrix n 1 m}
    (h : RelE sm A (fun j ↦ reducePoly q n (ν * unitPair sp j) • G) c e) (k : Int)
    {K : ExactMatrix q n 1 1} (hK : K = matrixPolynomial [k])
    {H : ExactMatrix q n m 1} (hH : gadgetDecomposeRuns backend base digits K H)
    {bound : Nat} {H' : ExactMatrix q n m 1} (hH' : compactCenteredRebaseRuns bound H H') :
    ∃ d : ErrorMatrix n m 1, H' = reduceMatrix q n m 1 d ∧ CoeffBound d (base.toNat / 2) ∧
      (∀ r c (i : Fin n), i.val ≠ 0 → (d r c).coeff i = 0) ∧
      RelE sm (fun a j ↦ matrixMul (A a j) H')
        (fun j ↦ reducePoly q n ((k : ErrorPoly n) * ν * unitPair sp j) • (1 : ExactMatrix q n 1 1))
        (fun j ↦ matrixMul (c j) H') (fun j ↦ e j * d) := by
  have hHH' : H = H' := centeredRebaseRuns_unique (centeredRebaseRuns_self hq1 hn H) hH'.2.2.1
  subst hHH'
  have hconst : ∀ r c (i : Fin n), i.val ≠ 0 → (K r c).coeff i = 0 := by
    intro r c i hi
    rw [hK, matrixPolynomial_single, ← reducePoly_intCast, reducePoly_coeff_eq hq1 hn,
      intCast_coeff_ne_zero hn k hi, Int.cast_zero]
  obtain ⟨d, hd, hdb, hdc⟩ := gadgetDecomposeRuns_constant hq1 hn hH
    (fun l hl ↦ by rw [hlayout] at hl; cases hl; exact hexact) hconst
  have hact := h.mulRight d
  rw [← hd] at hact
  have hrec := gadget_mul_decompose hq1 hn layout hlayout hexact hG hH
  refine ⟨d, hd, hdb, hdc, hact.target_eq (funext fun j ↦ ?_)⟩
  rw [Matrix.smul_mul, hrec, hK]
  funext r c
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain rfl : c = 0 := Subsingleton.elim _ _
  simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, Matrix.one_apply_eq,
    mul_one, map_mul, reducePoly_intCast]
  ring

end Operations

/-! ## Decryption -/

/-- `decode_terminal_scaled` with the error named: decoding coefficient `i` is correct when that
coefficient of the error satisfies `2 t |e| + 2 r W < q`. -/
theorem decode_terminal_scaledE {q t k r W : Nat} (hq : 1 < q) (hn : 0 < n) (ht : 0 < t)
    (hk : t * k + r = q) {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin 2 → ExactMatrix q n 1 1} {e : Fin 2 → ErrorMatrix n 1 1}
    (hrel : RelE sm A (fun j ↦ reducePoly q n (((k : Int) : ErrorPoly n) * μ * unitPair sp j) •
      (1 : ExactMatrix q n 1 1)) c e) {i : Fin n}
    (he : 2 * t * |(e 1 0 0).coeff i| + 2 * r * W < q) (hμ : |μ.coeff i| ≤ W)
    {phase : ExactMatrix q n 1 1}
    (hphase : phase = -(c 1 - (reducePoly q n sm • A 0 1 + A 1 1))) {length : Int} {out : Int}
    (hdecode : thresholdDecode t length i.val phase out) : out = μ.coeff i % t := by
  obtain ⟨_, _, _, _, index, hindex, _, hout⟩ := hdecode
  have hi : index = i := Fin.ext (by exact_mod_cast hindex)
  subst hi
  rw [hout]
  have hentry : (phase 0 0).coeff index =
      ((k * μ.coeff index - (e 1 0 0).coeff index : Int) : ZMod q) := by
    rw [hphase, hrel 1]
    simp only [unitPair, Matrix.cons_val_one, Matrix.neg_apply, Matrix.sub_apply,
      Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix, mul_one,
      Matrix.one_apply_eq]
    rw [show ∀ x y z w : ExactPoly q n, -(x + y - z + w - (x + y)) = z - w from fun x y z w ↦ by
      ring, ← map_sub, reducePoly_coeff_eq hq hn]
    have hkμ := intCast_mul_coeff (k : Int) μ index
    push_cast at hkμ ⊢
    simp [hkμ]
  rw [hentry]
  exact threshold_decode_scaled hq ht hk hμ he

end MxxExponentLut
