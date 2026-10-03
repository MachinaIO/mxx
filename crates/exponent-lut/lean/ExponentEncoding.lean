import ExponentAutomorphism
import ExponentScale

/-!
# The RNS-BGG encoding relation

An encoding at modulus `q` with `m` gadget columns is a row `c = (c₀, c₁)` and a public key
`A = (A_{kj})` of `1 × m` blocks. Under the mask secret `s = (s₀, 1)` it satisfies

  `c_j = s₀ · A_{0j} + A_{1j} - t_j + e_j`

for a target `t` and a small integer error `e` (`Rel`). The target of a fresh encoding of `μ` is
`t_j = (Δ μ r_j) · G` for the payload secret `r = (r₀, 1)`, the message scale `Δ`, and the level
gadget `G`; evaluation transforms targets and errors:

* a right action `[x₀ | x₁] · D` by a stacked matrix with small entries (`pairMul`) acts on the
  row, the key, and the target, and multiplies the error bound by the action gain;
* an odd substitution acts on everything, including the mask;
* the runtime ModDown divides a target `scaleUp T` by the dropped factor `p`;
* sums of relations with the same mask add their targets and errors.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-! ## Stacked right actions -/

/-- The first `m` rows of a stacked matrix. -/
def topRows {α : Type} {m m' : Nat} (d : Matrix (Fin (m + m)) (Fin m') α) :
    Matrix (Fin m) (Fin m') α :=
  d.submatrix (Fin.castAdd m) id

/-- The last `m` rows of a stacked matrix. -/
def bottomRows {α : Type} {m m' : Nat} (d : Matrix (Fin (m + m)) (Fin m') α) :
    Matrix (Fin m) (Fin m') α :=
  d.submatrix (Fin.natAdd m) id

/-- `[x₀ | x₁] · d` for a pair of row blocks and a stacked matrix. -/
def pairMul {α : Type} [CommRing α] {m m' : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) : Matrix (Fin 1) (Fin m') α :=
  x 0 * topRows d + x 1 * bottomRows d

section PairMul

variable {α : Type} [CommRing α] {m m' : Nat}

theorem pairMul_add (x y : Fin 2 → Matrix (Fin 1) (Fin m) α) (d : Matrix (Fin (m + m)) (Fin m') α) :
    pairMul (x + y) d = pairMul x d + pairMul y d := by
  simp only [pairMul, Pi.add_apply, Matrix.add_mul]
  abel

theorem pairMul_sub (x y : Fin 2 → Matrix (Fin 1) (Fin m) α) (d : Matrix (Fin (m + m)) (Fin m') α) :
    pairMul (x - y) d = pairMul x d - pairMul y d := by
  simp only [pairMul, Pi.sub_apply, Matrix.sub_mul]
  abel

theorem pairMul_smul (a : α) (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) :
    pairMul (a • x) d = a • pairMul x d := by
  simp only [pairMul, Pi.smul_apply, Matrix.smul_mul, smul_add]

end PairMul

theorem reduceMatrix_pairMul {q m m' : Nat} (x : Fin 2 → ErrorMatrix n 1 m)
    (d : ErrorMatrix n (m + m) m') :
    reduceMatrix q n 1 m' (pairMul x d) =
      pairMul (fun j ↦ reduceMatrix q n 1 m (x j)) (reduceMatrix q n (m + m) m' d) := by
  simp only [pairMul, reduceMatrix_add, reduceMatrix_mul]
  rfl

theorem reduceMatrix_sub {q rows columns : Nat} (x y : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (x - y) =
      reduceMatrix q n rows columns x - reduceMatrix q n rows columns y := by
  funext row column
  simp [reduceMatrix]

/-! ## Coefficient bounds -/

theorem coeffBound_mono {rows columns : Nat} {e : ErrorMatrix n rows columns} {a b : Nat}
    (h : CoeffBound e a) (hab : a ≤ b) : CoeffBound e b :=
  fun row column coefficient ↦ (h row column coefficient).trans hab

theorem coeffBound_neg {rows columns : Nat} {e : ErrorMatrix n rows columns} {a : Nat}
    (h : CoeffBound e a) : CoeffBound (-e) a := by
  intro row column coefficient
  simpa using h row column coefficient

theorem coeffBound_sub {rows columns : Nat} {e f : ErrorMatrix n rows columns} {a b : Nat}
    (he : CoeffBound e a) (hf : CoeffBound f b) : CoeffBound (e - f) (a + b) := by
  rw [sub_eq_add_neg]
  exact coeffBound_add he (coeffBound_neg hf)

theorem coeffBound_sum {ι : Type} [DecidableEq ι] {rows columns : Nat} (s : Finset ι)
    {e : ι → ErrorMatrix n rows columns} {b : ι → Nat} (h : ∀ i ∈ s, CoeffBound (e i) (b i)) :
    CoeffBound (∑ i ∈ s, e i) (∑ i ∈ s, b i) := by
  induction s using Finset.induction_on with
  | empty => simpa using (coeffBound_zero (n := n) (rows := rows) (columns := columns))
  | @insert i s hi ih =>
    rw [Finset.sum_insert hi, Finset.sum_insert hi]
    exact coeffBound_add (h i (Finset.mem_insert_self i s))
      (ih fun j hj ↦ h j (Finset.mem_insert_of_mem hj))

theorem coeffBound_polyNorm {rows columns : Nat} {e : ErrorMatrix n rows columns} {b : Nat}
    (h : CoeffBound e b) (row : Fin rows) (column : Fin columns) : polyNorm (e row column) ≤ b :=
  polyNorm_le_of_coeff (h row column)

/-- Multiplication by a polynomial of coefficient bound `a` multiplies a bound by `n a`. -/
theorem coeffBound_smul (hn : 0 < n) {rows columns : Nat} {x : ErrorPoly n}
    {e : ErrorMatrix n rows columns} {a b : Nat} (hx : polyNorm x ≤ a) (he : CoeffBound e b) :
    CoeffBound (x • e) (n * a * b) := by
  intro row column coefficient
  apply (coeff_natAbs_le_polyNorm _ coefficient).trans
  change polyNorm (x * e row column) ≤ _
  apply (polyNorm_mul_le_tight hn _ _).trans
  exact Nat.mul_le_mul (Nat.mul_le_mul_left n hx) (coeffBound_polyNorm he row column)

theorem coeffBound_pairMul (hn : 0 < n) {m m' : Nat} {x : Fin 2 → ErrorMatrix n 1 m}
    {d : ErrorMatrix n (m + m) m'} {a b : Nat} (hx : ∀ j, CoeffBound (x j) a)
    (hd : CoeffBound d b) : CoeffBound (pairMul x d) ((m + m) * n * a * b) := by
  have htop : CoeffBound (topRows d) b := fun row column ↦ hd _ _
  have hbottom : CoeffBound (bottomRows d) b := fun row column ↦ hd _ _
  have h := coeffBound_add (coeffBound_mul hn (hx 0) htop) (coeffBound_mul hn (hx 1) hbottom)
  refine coeffBound_mono h (le_of_eq ?_)
  ring

/-! ## The relation -/

/-- The second payload or mask component is one. -/
def unitPair {α : Type} [One α] (x : α) : Fin 2 → α := ![x, 1]

/-- `c_j = s₀ A_{0j} + A_{1j} - t_j + e_j` with `CoeffBound e B`, for the mask `(s₀, 1)`. -/
def Rel {q m : Nat} (sm : ErrorPoly n) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (t c : Fin 2 → ExactMatrix q n 1 m) (B : Nat) : Prop :=
  ∃ e : Fin 2 → ErrorMatrix n 1 m, (∀ j, CoeffBound (e j) B) ∧
    ∀ j, c j = reducePoly q n sm • A 0 j + A 1 j - t j + reduceMatrix q n 1 m (e j)

section Relation

variable {q m : Nat} {sm : ErrorPoly n}

theorem Rel.mono {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {B B' : Nat} (h : Rel sm A t c B) (hB : B ≤ B') : Rel sm A t c B' := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e, fun j ↦ coeffBound_mono (he j) hB, hc⟩

/-- A small additive term of the target moves into the error. -/
theorem Rel.absorb {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {f : Fin 2 → ErrorMatrix n 1 m} {B F : Nat} (h : Rel sm A (fun j ↦ t j + reduceMatrix q n 1 m (f j)) c B)
    (hf : ∀ j, CoeffBound (f j) F) : Rel sm A t c (B + F) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j - f j, fun j ↦ coeffBound_sub (he j) (hf j), fun j ↦ ?_⟩
  simp only [hc j, reduceMatrix_sub]
  abel

/-- Relations with the same mask add. -/
theorem Rel.add {A A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t t' c c' : Fin 2 → ExactMatrix q n 1 m}
    {B B' : Nat} (h : Rel sm A t c B) (h' : Rel sm A' t' c' B') :
    Rel sm (A + A') (t + t') (c + c') (B + B') := by
  obtain ⟨e, he, hc⟩ := h
  obtain ⟨e', he', hc'⟩ := h'
  refine ⟨e + e', fun j ↦ coeffBound_add (he j) (he' j), fun j ↦ ?_⟩
  simp only [Pi.add_apply, hc j, hc' j, reduceMatrix_add, smul_add]
  abel

theorem Rel.target_eq {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t t' c : Fin 2 → ExactMatrix q n 1 m}
    {B : Nat} (h : Rel sm A t c B) (ht : t = t') : Rel sm A t' c B :=
  ht ▸ h

theorem Rel.zero : Rel (q := q) (m := m) sm 0 0 0 0 := by
  refine ⟨0, fun j ↦ coeffBound_zero, fun j ↦ ?_⟩
  funext row column
  simp [reduceMatrix]

theorem Rel.sum {ι : Type} [DecidableEq ι] (s : Finset ι)
    {A : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : ι → Fin 2 → ExactMatrix q n 1 m}
    {B : ι → Nat} (h : ∀ i ∈ s, Rel sm (A i) (t i) (c i) (B i)) :
    Rel sm (∑ i ∈ s, A i) (∑ i ∈ s, t i) (∑ i ∈ s, c i) (∑ i ∈ s, B i) := by
  induction s using Finset.induction_on with
  | empty => simpa using (Rel.zero (q := q) (m := m) (sm := sm))
  | @insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact (h i (Finset.mem_insert_self i s)).add (ih fun j hj ↦ h j (Finset.mem_insert_of_mem hj))

/-- A stacked right action with small entries: every block is acted on, and the error bound gains
the factor `(m + m) n β`. -/
theorem Rel.action (hn : 0 < n) {m' : Nat} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B β : Nat} (h : Rel sm A t c B)
    (d : Fin 2 → ErrorMatrix n (m + m) m') (hd : ∀ j, CoeffBound (d j) β) :
    Rel sm (fun k j ↦ pairMul (A k) (reduceMatrix q n (m + m) m' (d j)))
      (fun j ↦ pairMul t (reduceMatrix q n (m + m) m' (d j)))
      (fun j ↦ pairMul c (reduceMatrix q n (m + m) m' (d j))) ((m + m) * n * B * β) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ pairMul e (d j), fun j ↦ coeffBound_pairMul hn he (hd j), fun j ↦ ?_⟩
  have hcfun : c = reducePoly q n sm • A 0 + A 1 - t + fun i ↦ reduceMatrix q n 1 m (e i) := by
    funext i
    simp [hc i]
  simp only []
  rw [hcfun, reduceMatrix_pairMul, pairMul_add, pairMul_sub, pairMul_add, pairMul_smul]

/-- The substitution `X ↦ X^index` applied to every polynomial of a matrix. -/
noncomputable def substituteMatrix {R : Type} [CommRing R] {rows columns : Nat} (index : Nat)
    (hodd : Odd index) (x : Matrix (Fin rows) (Fin columns) (Negacyclic n R)) :
    Matrix (Fin rows) (Fin columns) (Negacyclic n R) :=
  fun row column ↦ substitution index hodd (x row column)

/-- An odd substitution with index coprime to `n` acts on the relation, including its mask. -/
theorem Rel.substitute (hn : 0 < n) {index : Nat} (hodd : Odd index) (hcoprime : n.Coprime index)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A t c B) :
    Rel (substitution index hodd sm) (fun k j ↦ substituteMatrix index hodd (A k j))
      (fun j ↦ substituteMatrix index hodd (t j)) (fun j ↦ substituteMatrix index hodd (c j)) B := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ substituteMatrix index hodd (e j), fun j ↦ ?_, fun j ↦ ?_⟩
  · intro row column coefficient
    exact (coeff_natAbs_le_polyNorm _ coefficient).trans
      ((polyNorm_substitution_le hn hodd hcoprime _).trans (coeffBound_polyNorm (he j) row column))
  · funext row column
    simp only [substituteMatrix, hc j, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
      smul_eq_mul, reduceMatrix, map_add, map_sub, map_mul, reducePoly_substitution]

/-- The runtime ModDown of the row and the key divides a target `scaleUp T` by the dropped
factor `p`, for a ternary mask. -/
theorem Rel.modDown (hn : 0 < n) {p q' : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (hp : 0 < p) {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m}
    {B : Nat} (h : Rel sm A t c B) (hsm : polyNorm sm ≤ 1)
    {A' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m} {T c' : Fin 2 → ExactMatrix q' n 1 m}
    (ht : ∀ j, t j = scaleUpMatrix q p (T j))
    (hA : ∀ k j, ∃ r : ErrorMatrix n 1 m, CoeffBound r (p / 2) ∧
      scaleUpMatrix q p (A' k j) = A k j - reduceMatrix q n 1 m r)
    (hc : ∀ j, ∃ r : ErrorMatrix n 1 m, CoeffBound r (p / 2) ∧
      scaleUpMatrix q p (c' j) = c j - reduceMatrix q n 1 m r) :
    Rel sm A' T c' ((B + (n + 2) * (p / 2)) / p) := by
  obtain ⟨e, he, hrel⟩ := h
  -- The integer residual of each component and its division by `p`.
  have hdiv : ∀ j, ∃ v : ErrorMatrix n 1 m, CoeffBound v ((B + (n + 2) * (p / 2)) / p) ∧
      c' j = reducePoly q' n sm • A' 0 j + A' 1 j - T j + reduceMatrix q' n 1 m v := by
    intro j
    obtain ⟨r0, hr0, hA0⟩ := hA 0 j
    obtain ⟨r1, hr1, hA1⟩ := hA 1 j
    obtain ⟨rc, hrc, hcj⟩ := hc j
    let w : ErrorMatrix n 1 m := e j - rc + sm • r0 + r1
    have hw : CoeffBound w (B + (n + 2) * (p / 2)) := by
      have h1 := coeffBound_add (coeffBound_add (coeffBound_sub (he j) hrc)
        (coeffBound_smul hn hsm hr0)) hr1
      refine coeffBound_mono h1 (le_of_eq ?_)
      ring
    have hscale : ∀ row column, scaleUp q p
        ((c' j - (reducePoly q' n sm • A' 0 j + A' 1 j - T j)) row column) =
        reducePoly q n (w row column) := by
      intro row column
      have hc' := congrFun (congrFun hcj row) column
      have h0 := congrFun (congrFun hA0 row) column
      have h1 := congrFun (congrFun hA1 row) column
      have ht' := congrFun (congrFun (ht j) row) column
      have hr := congrFun (congrFun (hrel j) row) column
      simp only [scaleUpMatrix, Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply,
        smul_eq_mul, reduceMatrix] at hc' h0 h1 ht' hr
      simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul]
      rw [scaleUp_sub hq hq' hn, scaleUp_sub hq hq' hn, scaleUp_add hq hq' hn,
        scaleUp_reducePoly_mul hq hq' hn, hc', h0, h1, ← ht', hr]
      simp only [w, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, map_add,
        map_sub, map_mul]
      ring
    choose v hv using fun row column ↦ scaleUp_eq_reducePoly hq hq1 hq' hn (hscale row column)
    refine ⟨fun row column ↦ v row column, ?_, ?_⟩
    · intro row column coefficient
      have hwv := congrArg (fun z : ErrorPoly n ↦ (z.coeff coefficient).natAbs) (hv row column).1
      simp only [intCast_mul_coeff, Int.natAbs_mul, Int.natAbs_natCast] at hwv
      rw [Nat.le_div_iff_mul_le hp, Nat.mul_comm, ← hwv]
      exact hw row column coefficient
    · funext row column
      have := (hv row column).2
      simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul,
        reduceMatrix] at this ⊢
      linear_combination this

  choose v hv using hdiv
  exact ⟨v, fun j ↦ (hv j).1, fun j ↦ (hv j).2⟩

/-- One LUT branch. The routed input, after the switching action, is related under the
substituted mask `σ s` to the action `M` of the routed key; the helper, after the action by the
decomposition of `ModDown M`, is related under the mask `s` with the target `p (σ s · ModDown M)`.
Their sum is related under `s` to the helper key, with the routed target. -/
theorem Rel.lutBranch (hn : 0 < n) {p q' : Nat} (hq : q = p * q') (hq' : 1 < q')
    {smσ : ErrorPoly n} (hsmσ : polyNorm smσ ≤ 1)
    {M K : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {tc X th Y : Fin 2 → ExactMatrix q n 1 m}
    {Bc Bh : Nat} (hc : Rel smσ M tc X Bc)
    {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m}
    (hM : ∀ k j, ∃ r : ErrorMatrix n 1 m, CoeffBound r (p / 2) ∧
      scaleUpMatrix q p (Asw k j) = M k j - reduceMatrix q n 1 m r)
    (hh : Rel sm K th Y Bh)
    (hth : ∀ j, th j = scaleUpMatrix q p (reducePoly q' n smσ • Asw 0 j + Asw 1 j)) :
    Rel sm K tc (Y + X) (Bh + Bc + (n * 1 * (p / 2) + p / 2)) := by
  obtain ⟨ec, hec, hX⟩ := hc
  obtain ⟨eh, heh, hY⟩ := hh
  have hsplit : ∀ j, ∃ r : ErrorMatrix n 1 m, CoeffBound r (n * 1 * (p / 2) + p / 2) ∧
      reducePoly q n smσ • M 0 j + M 1 j = th j + reduceMatrix q n 1 m r := by
    intro j
    obtain ⟨r0, hr0, h0⟩ := hM 0 j
    obtain ⟨r1, hr1, h1⟩ := hM 1 j
    refine ⟨smσ • r0 + r1, coeffBound_add (coeffBound_smul hn hsmσ hr0) hr1, ?_⟩
    rw [hth j]
    funext row column
    have h0' := congrFun (congrFun h0 row) column
    have h1' := congrFun (congrFun h1 row) column
    simp only [scaleUpMatrix, Matrix.sub_apply, reduceMatrix] at h0' h1'
    simp only [scaleUpMatrix, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix,
      map_add, map_mul]
    rw [scaleUp_add hq hq' hn, scaleUp_reducePoly_mul hq hq' hn, h0', h1']
    ring
  choose r hr using hsplit
  refine ⟨fun j ↦ eh j + ec j + r j, fun j ↦ coeffBound_add (coeffBound_add (heh j) (hec j))
    (hr j).1, fun j ↦ ?_⟩
  rw [Pi.add_apply, hY j, hX j]
  have hrj := (hr j).2
  rw [show reducePoly q n smσ • M 0 j + M 1 j - tc j + reduceMatrix q n 1 m (ec j) =
      (reducePoly q n smσ • M 0 j + M 1 j) - tc j + reduceMatrix q n 1 m (ec j) from rfl, hrj,
    reduceMatrix_add, reduceMatrix_add]
  abel

end Relation

end MxxExponentLut
