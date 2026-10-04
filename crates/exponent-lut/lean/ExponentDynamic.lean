import ExponentPublic
import ExponentIterate
import ExponentRuntime

/-!
# Dynamic RHS products

A Dynamic product replaces the key action `A Dec(C)` of a private multiplication with keys `T`
fixed by public companion keys. For every column `a` of the state key and every output column, the
encryption stores a digit companion `z = s B - d (s, 1) G + ε` of the digit `d` of the ciphertext
under the companion key `B`. Online evaluation forms `T = Σ_a B Dec(A_{:,a})` and
`E = Σ_a z Dec(A_{:,a})`; since the gadget reconstructs `A_{:,a}` from its digits,
`E = s T - s A Dec(C) + Σ_a ε Dec(A_{:,a})`, and `c Dec(C) + E` encodes the product under `T`.

The runtime decomposes a ciphertext one column at a time, so the digits of a column are
`columnDigits` and the digits of a key entry are `entryDigits`; `gadget_action_columns` assembles
the columns into the action of the whole decomposition. The generated loops accumulate one term per
iteration into a row through a window of a constant unit row; `unitRow_window`,
`sum_unitRow_entry`, and `rowWindow` read those rows back entry by entry.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-! ## Windows of constant rows -/

/-- A window of a unit row that starts `c` columns before its one is the unit row with its one
at `c`. -/
theorem unitRow_window {q w W : Nat} {k s t c : Int} (hs : s = k - c)
    {out : ExactMatrix q n 1 w}
    (h : sliceMatrix (unitRow (q := q) (n := n) (columns := W) k) 0 1 s t out) :
    out = unitRow c := by
  obtain ⟨-, -, -, hs0, -, he, -, hw, hentry⟩ := h
  simp only [Int.ofNat_eq_natCast] at he hw
  funext row column
  rw [hentry row column (by have := row.isLt; omega) (by have := column.isLt; omega)]
  simp only [unitRow]
  have hstart : (s.toNat : Int) = s := Int.toNat_of_nonneg hs0
  apply if_congr _ rfl rfl
  constructor <;> intro h <;> push_cast at h ⊢ <;> omega

/-- The `m` columns of a row from `start` on. -/
def rowWindow {α : Type} {W : Nat} (m start : Nat) (h : start + m ≤ W)
    (x : Matrix (Fin 1) (Fin W) α) : Matrix (Fin 1) (Fin m) α :=
  fun _ l ↦ x 0 ⟨start + l, by omega⟩

/-- A slice of a row is its window. -/
theorem sliceMatrix_window {q W m : Nat} {x : ExactMatrix q n 1 W} {s e : Int}
    {y : ExactMatrix q n 1 m} (start : Nat) (h : sliceMatrix x 0 1 s e y) (hs : s = start)
    (hw : start + m ≤ W) : y = rowWindow m start hw x := by
  subst hs
  obtain ⟨-, -, -, -, -, he, -, hm, hentry⟩ := h
  simp only [Int.ofNat_eq_natCast] at he hm
  funext r l
  rw [Subsingleton.elim r 0, hentry 0 l (by simp) (by simp; omega)]
  simp [rowWindow]

/-- A slice of a slice of a row is a window of the row. -/
theorem sliceMatrix_window₂ {q W V m : Nat} {x : ExactMatrix q n 1 W} {y : ExactMatrix q n 1 V}
    {z : ExactMatrix q n 1 m} {s e s' e' : Int} (a b : Nat) (h1 : sliceMatrix x 0 1 s e y)
    (h2 : sliceMatrix y 0 1 s' e' z) (hs : s = a) (hs' : s' = b) (hw : a + b + m ≤ W) :
    z = rowWindow m (a + b) hw x := by
  have hV : a + V ≤ W := by
    obtain ⟨-, -, -, -, -, he, -, hm, -⟩ := h1
    simp only [Int.ofNat_eq_natCast] at he hm
    omega
  have hm : b + m ≤ V := by
    obtain ⟨-, -, -, -, -, he, -, hm, -⟩ := h2
    simp only [Int.ofNat_eq_natCast] at he hm
    omega
  rw [sliceMatrix_window b h2 hs' hm, sliceMatrix_window a h1 hs hV]
  funext r l
  simp [rowWindow, Nat.add_assoc]

/-- A window of a sum is the sum of the windows. -/
theorem rowWindow_add {α : Type} [Add α] {W m start : Nat} (h : start + m ≤ W)
    (x y : Matrix (Fin 1) (Fin W) α) :
    rowWindow m start h (x + y) = rowWindow m start h x + rowWindow m start h y := rfl

/-- A one-column window of a row is its entry. -/
theorem sliceMatrix_column {q m : Nat} {x : ExactMatrix q n 1 m} {a b : Int}
    {y : ExactMatrix q n 1 1}
    (h : sliceMatrix x 0 1 a b y) :
    ∃ ha : a.toNat < m, y 0 0 = x 0 ⟨a.toNat, ha⟩ := by
  obtain ⟨-, -, -, hs, -, he, -, hw, hentry⟩ := h
  simp only [Int.ofNat_eq_natCast] at he hw
  have ha : a.toNat < m := by omega
  refine ⟨ha, ?_⟩
  rw [hentry 0 0 (by omega) (by simpa using ha)]
  rfl

/-- The unit row with its one at `a` picks row `a` of a column. -/
theorem unitRow_mul_entry {q k : Nat} (a : Nat) (ha : a < k) (D : ExactMatrix q n k 1) :
    ((unitRow (a : Int) : ExactMatrix q n 1 k) * D) 0 0 = D ⟨a, ha⟩ 0 := by
  rw [Matrix.mul_apply, Finset.sum_eq_single ⟨a, ha⟩]
  · simp [unitRow]
  · intro b _ hb
    have : (b.val : Int) ≠ a := fun h ↦ hb (Fin.ext (by exact_mod_cast h))
    simp [unitRow, this]
  · simp

/-- The tensor of a scalar with a row scales the row. -/
theorem tensorRuns_scalar {q D : Nat} {x : ExactMatrix q n 1 1} {g : ExactMatrix q n 1 D}
    {out : ExactMatrix q n (1 * 1) (1 * D)} (h : tensorRuns x g out) (c : Nat) (hc : c < D) :
    out ⟨0, by omega⟩ ⟨c, by omega⟩ = x 0 0 * g 0 ⟨c, hc⟩ := by
  have := h 0 0 0 ⟨c, hc⟩
  simpa using this

/-- A family read at an index that evaluates to `k`. -/
theorem familyGet_index {α : Type} {count k : Nat} {f : Fin count → α} {idx : Int} {out : α}
    (h : familyGetDynamic f idx out) (hidx : idx = Int.ofNat k) :
    ∃ hk : k < count, out = f ⟨k, hk⟩ := by
  subst hidx
  exact familyGet_eq h

/-! ## Changing the key of a relation -/

/-- A row `E` that adds `s (T_0 - A_0) + (T_1 - A_1)` and a small error moves an encoding under
`A` to the key `T`. -/
theorem Rel.rekey {q m : Nat} {sm : ErrorPoly n} {A T : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c E : Fin 2 → ExactMatrix q n 1 m} {f : Fin 2 → ErrorMatrix n 1 m} {B F : Nat}
    (h : Rel sm A t c B)
    (hE : ∀ j, E j = reducePoly q n sm • (T 0 j - A 0 j) + (T 1 j - A 1 j) +
      reduceMatrix q n 1 m (f j))
    (hf : ∀ j, CoeffBound (f j) F) :
    Rel sm T t (c + E) (B + F) := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ e j + f j, fun j ↦ coeffBound_add (he j) (hf j), fun j ↦ ?_⟩
  simp only [Pi.add_apply, hc j, hE j, reduceMatrix_add, smul_sub]
  abel

/-! ## Digits of one key entry -/

/-- The public gadget reconstructs a key entry from its digits, which are at most `base / 2`. -/
theorem decompose_entry {q D : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g : ExactMatrix q n 1 D} (hg : gadgetMatrixRuns backend base digits g)
    {x : ExactMatrix q n 1 1} {u : ExactMatrix q n D 1}
    (hu : gadgetDecomposeRuns backend base digits x u) :
    g * u = x ∧ PreimageWithin u (layout.base / 2) := by
  obtain ⟨layout₁, hl₁, _, _, hw₁, hg⟩ := hg
  obtain ⟨layout₂, hl₂, _, _, hw₂, hu, _⟩ := hu
  rw [hlayout] at hl₁ hl₂
  cases hl₁
  cases hl₂
  have hd : D = layout.digitCount := by simpa using hw₁
  subst hd
  subst hg hu
  constructor
  · have hrec := regularGadgetMatrix_reconstruct (rows := 1) layout x hq hn hexact
    funext row column
    rw [← congrFun (congrFun hrec row) column]
    simp only [Matrix.mul_apply, castMatrixColumns_apply, castMatrixRows_apply]
    exact Fintype.sum_equiv (finCongr (by simp)) _ _ (fun _ ↦ rfl)
  · obtain ⟨w, hw, hb⟩ := regularDecomposeMatrix_bounded (rows := 1) layout x hq hn
    refine ⟨fun row column ↦ w (Fin.cast (by simp) row) column, ?_, fun row column ↦ hb _ _⟩
    funext row column
    simp only [castMatrixRows_apply, hw, reduceMatrix]

/-! ## One digit companion -/

/-- A companion `z` of the digit `d` under the key `B`, paired with the digits `u_r` of the key
entries `x_r = g u_r`: the key pairs `B_k u` replace the companion, and the digit times
`s x₀ + x₁` is removed. -/
theorem companion_pairMul {q D : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {g : ExactMatrix q n 1 D} {B : Fin 2 → Fin 2 → ExactMatrix q n 1 D}
    {z : Fin 2 → ExactMatrix q n 1 D} {d : ExactPoly q n} {bound : Nat}
    (hz : Rel sm B (fun r ↦ (d * reducePoly q n (unitPair sm r)) • g) z bound)
    {u : Fin 2 → ExactMatrix q n D 1} {x : Fin 2 → ExactMatrix q n 1 1}
    (hgu : ∀ r, g * u r = x r) {u' : Fin 2 → ErrorMatrix n D 1} {β : Nat}
    (hu : ∀ r, u r = reduceMatrix q n D 1 (u' r)) (hβ : ∀ r, CoeffBound (u' r) β) :
    ∃ f : ErrorMatrix n 1 1, CoeffBound f (D * n * bound * β + D * n * bound * β) ∧
      z 0 * u 0 + z 1 * u 1 =
        reducePoly q n sm • (B 0 0 * u 0 + B 0 1 * u 1) + (B 1 0 * u 0 + B 1 1 * u 1) -
          d • (reducePoly q n sm • x 0 + x 1) + reduceMatrix q n 1 1 f := by
  obtain ⟨e, he, hze⟩ := hz
  refine ⟨e 0 * u' 0 + e 1 * u' 1,
    coeffBound_add (coeffBound_mul hn (he 0) (hβ 0)) (coeffBound_mul hn (he 1) (hβ 1)), ?_⟩
  have hred : ∀ r, reduceMatrix q n 1 D (e r) * u r = reduceMatrix q n 1 1 (e r * u' r) := by
    intro r
    rw [hu r, reduceMatrix_mul]
  rw [hze 0, hze 1, reduceMatrix_add, ← hred 0, ← hred 1]
  simp only [Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, hgu, unitPair,
    Matrix.cons_val_zero, Matrix.cons_val_one, map_one, mul_one, smul_add, smul_smul]
  abel

theorem reduceMatrix_sum {ι : Type} (s : Finset ι) {q rows columns : Nat}
    (e : ι → ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (∑ i ∈ s, e i) =
      ∑ i ∈ s, reduceMatrix q n rows columns (e i) := by
  funext row column
  simp [reduceMatrix, Matrix.sum_apply, map_sum]

/-- The companions of every key column `a`, each paired with that column's digits: the key pairs
replace the companions, and every digit times `s x₀ + x₁` is removed. -/
theorem companion_sum {ι : Type} [Fintype ι] [DecidableEq ι] {q D : Nat} (hn : 0 < n)
    {sm : ErrorPoly n} {g : ExactMatrix q n 1 D}
    {B : ι → Fin 2 → Fin 2 → ExactMatrix q n 1 D} {z : ι → Fin 2 → ExactMatrix q n 1 D}
    {d : ι → ExactPoly q n} {bound : Nat}
    (hz : ∀ a, Rel sm (B a) (fun r ↦ (d a * reducePoly q n (unitPair sm r)) • g) (z a) bound)
    {u : ι → Fin 2 → ExactMatrix q n D 1} {x : ι → Fin 2 → ExactMatrix q n 1 1}
    (hgu : ∀ a r, g * u a r = x a r) {u' : ι → Fin 2 → ErrorMatrix n D 1} {β : Nat}
    (hu : ∀ a r, u a r = reduceMatrix q n D 1 (u' a r)) (hβ : ∀ a r, CoeffBound (u' a r) β) :
    ∃ f : ErrorMatrix n 1 1,
      CoeffBound f (Fintype.card ι * (D * n * bound * β + D * n * bound * β)) ∧
      ∑ a, (z a 0 * u a 0 + z a 1 * u a 1) =
        reducePoly q n sm • ∑ a, (B a 0 0 * u a 0 + B a 0 1 * u a 1) +
          ∑ a, (B a 1 0 * u a 0 + B a 1 1 * u a 1) -
          ∑ a, d a • (reducePoly q n sm • x a 0 + x a 1) + reduceMatrix q n 1 1 f := by
  choose f hf heq using fun a ↦ companion_pairMul hn (hz a) (hgu a) (hu a) (hβ a)
  refine ⟨∑ a, f a, ?_, ?_⟩
  · have := coeffBound_sum Finset.univ (fun a _ ↦ hf a)
    simpa [Finset.sum_const, Finset.card_univ] using this
  · rw [Finset.sum_congr rfl fun a _ ↦ heq a, reduceMatrix_sum, Finset.smul_sum]
    simp only [Finset.sum_add_distrib, Finset.sum_sub_distrib]

/-! ## Loops that accumulate -/

/-- A loop whose iteration `i` adds `f i` to the part `proj` of its state, while a side condition
`Q` is kept, ends with the sum of the first `count` values. -/
theorem IterRuns.accumulate {σ M : Type} [AddCommMonoid M] {body : Nat → σ → σ → Prop}
    {count : Nat} {init final : σ} (proj : σ → M) (f : Nat → M) (Q : Nat → σ → Prop)
    (hQ : Q 0 init)
    (hstep : ∀ i current next, i < count → Q i current → body i current next →
      proj next = proj current + f i ∧ Q (i + 1) next)
    (h : MxxIR.IterRuns body count init final) :
    proj final = proj init + ∑ i ∈ Finset.range count, f i ∧ Q count final := by
  have := MxxIR.IterRuns.invariant
    (Invariant := fun k (s : σ) ↦ k ≤ count →
      proj s = proj init + ∑ i ∈ Finset.range k, f i ∧ Q k s)
    (fun _ ↦ ⟨by simp, hQ⟩) (fun i current next hcur hbody hk ↦ by
      obtain ⟨hp, hq⟩ := hcur (by omega)
      obtain ⟨hn, hq'⟩ := hstep i current next (by omega) hq hbody
      exact ⟨by rw [hn, hp, Finset.sum_range_succ, add_assoc], hq'⟩)
    h
  exact this le_rfl

/-- Entry `w` of a sum over `i < W A` that adds `F (i / A) (i % A)` at column `i / A` is the sum
over the `A` values for `w`. -/
theorem sum_unitRow_entry {q W A : Nat} (F : Fin W → Fin A → ExactPoly q n) (w : Fin W) :
    (∑ i ∈ Finset.range (W * A), if h : i < W * A then
      F ⟨i / A, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % A, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / A : Nat) : Int)) : ExactMatrix q n 1 W) else 0) 0 w = ∑ a, F w a := by
  rcases Nat.eq_zero_or_pos A with rfl | hA
  · simp
  rw [Finset.sum_range (fun i ↦ if h : i < W * A then
      F ⟨i / A, Nat.div_lt_of_lt_mul (by rwa [Nat.mul_comm] at h)⟩
        ⟨i % A, Nat.mod_lt _ (Nat.pos_of_ne_zero fun h0 ↦ by simp [h0] at h)⟩ •
      (unitRow (((i / A : Nat) : Int)) : ExactMatrix q n 1 W) else 0),
    ← finProdFinEquiv.sum_comp, Fintype.sum_prod_type, Matrix.sum_apply]
  have hdiv : ∀ (v : Fin W) (a : Fin A),
      ((finProdFinEquiv (v, a) : Fin (W * A)) : Nat) / A = v ∧
      ((finProdFinEquiv (v, a) : Fin (W * A)) : Nat) % A = a := by
    intro v a
    simp only [finProdFinEquiv_apply_val]
    constructor
    · rw [Nat.add_mul_div_left _ _ hA, Nat.div_eq_of_lt a.isLt, zero_add]
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt a.isLt]
  rw [Finset.sum_eq_single w]
  · rw [Matrix.sum_apply]
    apply Finset.sum_congr rfl
    intro a _
    rw [dif_pos (finProdFinEquiv (w, a)).isLt]
    simp only [(hdiv w a).1, (hdiv w a).2, Fin.eta, Matrix.smul_apply, unitRow, smul_eq_mul,
      if_true, mul_one]
  · intro v _ hv
    rw [Matrix.sum_apply]
    apply Finset.sum_eq_zero
    intro a _
    rw [dif_pos (finProdFinEquiv (v, a)).isLt]
    simp only [(hdiv v a).1, (hdiv v a).2, Fin.eta, Matrix.smul_apply, unitRow, smul_eq_mul]
    rw [if_neg fun h ↦ hv (Fin.ext (by exact_mod_cast h)).symm, mul_zero]
  · simp

/-! ## Digits of key entries -/

/-- The digits of a `1 × 1` entry. -/
noncomputable def entryDigits {q : Nat} (D : Nat) (backend : BackendContext) (base digits : Int)
    (x : ExactMatrix q n 1 1) : ExactMatrix q n D 1 :=
  Classical.epsilon fun u ↦ gadgetDecomposeRuns backend base digits x u

theorem entryDigits_eq {q D : Nat} {backend : BackendContext} {base digits : Int}
    {x : ExactMatrix q n 1 1} {u : ExactMatrix q n D 1}
    (h : gadgetDecomposeRuns backend base digits x u) : u = entryDigits D backend base digits x :=
  gadgetDecomposeRuns_unique h (Classical.epsilon_spec ⟨u, h⟩)

/-- With an exact layout, every entry has digits. -/
theorem entryDigits_runs {q D : Nat} (hq : 1 < q) (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} {g : ExactMatrix q n 1 D} (hg : gadgetMatrixRuns backend base digits g)
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) (x : ExactMatrix q n 1 1) :
    gadgetDecomposeRuns backend base digits x (entryDigits D backend base digits x) := by
  obtain ⟨layout', hl, hb, hd, hw, -⟩ := hg
  rw [hlayout] at hl
  cases hl
  exact Classical.epsilon_spec (p := fun u ↦ gadgetDecomposeRuns backend base digits x u)
    ⟨castMatrixRows hw.symm (regularDecomposeMatrix layout x), layout, hlayout, hb, hd, hw, rfl,
      regularGadgetMatrix_residual_exact layout x hq hn hexact⟩

/-! ## Inverse automorphisms -/

/-- Substituting `X ↦ X^a` after `X ↦ X^b` is the identity when `a b ≡ 1 (mod 2n)`. -/
theorem substitution_inverse {R : Type} [CommRing R] {a b : Nat} (ha : Odd a) (hb : Odd b)
    (hab : a * b % (2 * n) = 1) (x : Negacyclic n R) :
    substitution a ha (substitution b hb x) = x := by
  have h : (substitution (n := n) (R := R) a ha).comp (substitution b hb) = RingHom.id _ := by
    apply AdjoinRoot.ringHom_ext
    · ext r
      simp [substitution]
    · simp only [RingHom.comp_apply, substitution_root, map_pow, RingHom.id_apply]
      have hroot : AdjoinRoot.root (negacyclicModulus n R) ^ (2 * n) = 1 := by
        rw [pow_mul', Negacyclic.root_pow_n]
        simp
      rw [← pow_mul, ← Nat.div_add_mod (a * b) (2 * n), hab, pow_add, pow_mul, hroot, one_pow,
        one_mul, pow_one]
  exact RingHom.congr_fun h x

/-! ## Joined rows -/

/-- Entry `a` of the row `[x₀ | x₁]`. -/
def joinedEntry {α : Type} {m : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α) (a : Fin (m + m)) :
    α :=
  Fin.addCases (fun i ↦ x 0 0 i) (fun i ↦ x 1 0 i) a

/-- `[x₀ | x₁] · d` entry by entry. -/
theorem pairMul_entry {α : Type} [CommRing α] {m m' : Nat}
    (x : Fin 2 → Matrix (Fin 1) (Fin m) α)
    (d : Matrix (Fin (m + m)) (Fin m') α) (l : Fin m') :
    pairMul x d 0 l = ∑ a, joinedEntry x a * d a l := by
  simp only [pairMul, Matrix.add_apply, Matrix.mul_apply, topRows, bottomRows,
    Matrix.submatrix_apply, id, Fin.sum_univ_add, joinedEntry, Fin.addCases_left,
    Fin.addCases_right]

/-- A substitution acts on joined entries entry by entry. -/
theorem joinedEntry_substitute {R : Type} [CommRing R] {m index : Nat} (hodd : Odd index)
    (x : Fin 2 → Matrix (Fin 1) (Fin m) (Negacyclic n R)) (a : Fin (m + m)) :
    joinedEntry (fun j ↦ substituteMatrix index hodd (x j)) a =
      substitution index hodd (joinedEntry x a) := by
  induction a using Fin.addCases with
  | left i => simp [joinedEntry, substituteMatrix]
  | right i =>
    simp only [joinedEntry]
    rw [show (Fin.natAdd m i : Fin (m + m)) = Fin.addNat i m from Fin.ext (by simp [Nat.add_comm])]
    simp [Fin.addCases, substituteMatrix]

/-- Entry `m j + l` of `[x₀ | x₁]` is entry `l` of `x_j`. -/
theorem joinedEntry_pair {α : Type} {m : Nat} (x : Fin 2 → Matrix (Fin 1) (Fin m) α) (j : Fin 2)
    (l : Fin m) (h : m * j + l < m + m) : joinedEntry x ⟨m * j + l, h⟩ = x j 0 l := by
  rcases j with ⟨j, hj⟩
  interval_cases j
  · have : (⟨m * ((⟨0, hj⟩ : Fin 2) : Nat) + l, h⟩ : Fin (m + m)) = Fin.castAdd m l :=
      Fin.ext (by simp)
    rw [this]
    simp only [joinedEntry, Fin.addCases_left]
    rfl
  · have : (⟨m * ((⟨1, hj⟩ : Fin 2) : Nat) + l, h⟩ : Fin (m + m)) = Fin.natAdd m l :=
      Fin.ext (by simp [Nat.add_comm])
    rw [this]
    simp only [joinedEntry, Fin.addCases_right]
    rfl

/-- A substitution scales a scaled matrix by the substituted scalar. -/
theorem substituteMatrix_smul' {R : Type} [CommRing R] {rows columns index : Nat}
    (hodd : Odd index) (c : Negacyclic n R) (x : Matrix (Fin rows) (Fin columns) (Negacyclic n R)) :
    substituteMatrix index hodd (c • x) =
      substitution index hodd c • substituteMatrix index hodd x := by
  funext i j
  simp [substituteMatrix]

/-- The entries of a row concatenation are the joined entries. -/
theorem concatColumns_entry {q m : Nat} {l r : ExactMatrix q n 1 m}
    {out : ExactMatrix q n 1 (m + m)}
    (h : concatColumns l r out) (a : Fin (m + m)) : out 0 a = joinedEntry ![l, r] a := by
  rw [h 0 a]
  induction a using Fin.addCases with
  | left i => simp [joinedEntry]
  | right i =>
    simp only [joinedEntry]
    rw [show (Fin.natAdd m i : Fin (m + m)) = Fin.addNat i m from Fin.ext (by simp [Nat.add_comm])]
    simp [Fin.addCases]

/-- The column `(x, y)` of a two-row matrix. -/
def pairColumn {α : Type} (x y : α) : Matrix (Fin 2) (Fin 1) α :=
  fun r _ ↦ if r.val = 0 then x else y

/-- Stacking two entries gives their column. -/
theorem concatRows_pairColumn {q : Nat} {x y : ExactMatrix q n 1 1}
    {out : ExactMatrix q n (1 + 1) 1}
    (h : concatRows x y out) : out = pairColumn (x 0 0) (y 0 0) := by
  funext r c
  obtain rfl : c = 0 := Subsingleton.elim _ _
  rw [h r 0]
  fin_cases r <;> simp [pairColumn]

/-! ## Column digits -/

/-- The digits of the column `(x, y)` re-encoded at `q`, as the runtime decomposes a matrix one
column at a time. -/
noncomputable def columnDigits {q' : Nat} (q k : Nat) (backend : BackendContext)
    (base digits : Int) (bound : Nat) (x y : ExactPoly q' n) : ExactMatrix q n k 1 :=
  Classical.epsilon fun D ↦ ∃ D' : ExactMatrix q' n k 1,
    gadgetDecomposeRuns backend base digits (pairColumn x y) D' ∧
      compactCenteredRebaseRuns bound D' D

/-- A decomposition of a column that runs is its `columnDigits`. -/
theorem columnDigits_eq {q q' k : Nat} {backend : BackendContext} {base digits : Int}
    {bound : Nat} {x y : ExactPoly q' n} {D' : ExactMatrix q' n k 1} {D : ExactMatrix q n k 1}
    (hD' : gadgetDecomposeRuns backend base digits (pairColumn x y) D')
    (hD : compactCenteredRebaseRuns bound D' D) :
    D = columnDigits q k backend base digits bound x y := by
  obtain ⟨E', hE', hE⟩ := Classical.epsilon_spec (p := fun D ↦ ∃ D' : ExactMatrix q' n k 1,
    gadgetDecomposeRuns backend base digits (pairColumn x y) D' ∧
      compactCenteredRebaseRuns bound D' D) ⟨D, D', hD', hD⟩
  obtain rfl := gadgetDecomposeRuns_unique hD' hE'
  exact centeredRebaseRuns_unique hD.2.2.1 hE.2.2.1

/-- Digits decomposed column by column act like the decomposition of the whole matrix. -/
theorem gadget_action_columns {q p q' m m' : Nat} (hq : q = p * q') (hq' : 1 < q') (hn : 0 < n)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g) {bound : Nat}
    {top bot : ExactMatrix q' n 1 m'} {D : ExactMatrix q n (m + m) m'}
    (hD : ∀ l, ∃ D' : ExactMatrix q' n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (top 0 l) (bot 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ D a l))
    (x : Fin 2 → ErrorPoly n) :
    pairMul (fun k ↦ reducePoly q n (((p : Int) : ErrorPoly n) * x k) • g) D =
      scaleUpMatrix q p (reducePoly q' n (x 0) • top + reducePoly q' n (x 1) • bot) := by
  funext r l
  obtain rfl : r = 0 := Subsingleton.elim _ _
  obtain ⟨D', hD', hDl⟩ := hD l
  have h := congrFun (congrFun
    (gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg hD' hDl x) 0) 0
  rw [pairMul_entry] at h ⊢
  simpa [scaleUpMatrix, rowOf, pairColumn] using h

/-- Digits bounded column by column are bounded. -/
theorem columns_bounded {q k m' bound : Nat} {D : ExactMatrix q n k m'}
    (hD : ∀ l, PreimageWithin (fun a (_ : Fin 1) ↦ D a l) bound) : PreimageWithin D bound := by
  choose w hw hb using hD
  refine ⟨fun a l ↦ w l a 0, ?_, fun a l ↦ hb l a 0⟩
  funext a l
  simpa [reduceMatrix] using congrFun (congrFun (hw l) a) 0

/-- Component `j` of a row that a column loop fills with `R Dc_p`, for the joined row `R` of
`row`, is `row` times the digits of the columns of component `j`. -/
theorem window_mul_columns {q m k : Nat} {row : Fin 2 → ExactMatrix q n 1 m}
    {R : ExactMatrix q n 1 (m + m)} (hR : concatColumns (row 0) (row 1) R)
    {acc : ExactMatrix q n 1 (k + k)} {Dc : Fin (k + k) → ExactMatrix q n (m + m) 1}
    (hacc : ∀ p, acc 0 p = (R * Dc p) 0 0) (j : Fin 2) (hj : k * j + k ≤ k + k) :
    rowWindow k (k * j) hj acc = pairMul row (fun a l ↦ Dc ⟨k * j + l, by omega⟩ a 0) := by
  funext r l
  rw [Subsingleton.elim r 0, pairMul_entry]
  simp only [rowWindow, hacc, Matrix.mul_apply, concatColumns_entry hR, pair_eta]

/-! ## The Dynamic private multiplication -/

/-- The companion row and keys of a Dynamic product: entry `l` of output component `j` sums, over
the key columns `a`, the companion of digit `D_j[a, l]` paired with the digits of key column `a`,
and the keys sum the companion keys paired with the same digits. Then the row adds
`s (M₀ - A₀ D_j) + (M₁ - A₁ D_j)` and a small error. -/
theorem companion_rows {q m m' Dg : Nat} (hn : 0 < n) {sm : ErrorPoly n}
    {g : ExactMatrix q n 1 Dg} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {D : Fin 2 → ExactMatrix q n (m + m) m'}
    {B : Fin 2 → Fin m' → Fin (m + m) → Fin 2 → Fin 2 → ExactMatrix q n 1 Dg}
    {z : Fin 2 → Fin m' → Fin (m + m) → Fin 2 → ExactMatrix q n 1 Dg} {bound : Nat}
    (hz : ∀ j l a, Rel sm (B j l a) (fun r ↦ (D j a l * reducePoly q n (unitPair sm r)) • g)
      (z j l a) bound)
    {u : Fin (m + m) → Fin 2 → ExactMatrix q n Dg 1}
    (hgu : ∀ a r, g * u a r = fun _ _ ↦ joinedEntry (A r) a)
    {u' : Fin (m + m) → Fin 2 → ErrorMatrix n Dg 1} {β : Nat}
    (hu : ∀ a r, u a r = reduceMatrix q n Dg 1 (u' a r)) (hβ : ∀ a r, CoeffBound (u' a r) β)
    {E : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hErow : ∀ j l, E j 0 l = (∑ a, (z j l a 0 * u a 0 + z j l a 1 * u a 1)) 0 0)
    (hMrow : ∀ k j l, M k j 0 l = (∑ a, (B j l a k 0 * u a 0 + B j l a k 1 * u a 1)) 0 0) :
    ∃ f : Fin 2 → ErrorMatrix n 1 m',
      (∀ j, CoeffBound (f j) ((m + m) * (Dg * n * bound * β + Dg * n * bound * β))) ∧
      ∀ j, E j = reducePoly q n sm • (M 0 j - pairMul (A 0) (D j)) +
        (M 1 j - pairMul (A 1) (D j)) + reduceMatrix q n 1 m' (f j) := by
  choose f hf heq using fun j l ↦ companion_sum (ι := Fin (m + m)) hn (hz j l) hgu hu hβ
  refine ⟨fun j _ l ↦ f j l 0 0, fun j row l c ↦ ?_, fun j ↦ ?_⟩
  · simpa using hf j l 0 0 c
  · funext row l
    obtain rfl : row = 0 := Subsingleton.elim _ _
    have h := congrFun (congrFun (heq j l) 0) 0
    rw [hErow j l, h]
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, Matrix.sum_apply,
      hMrow, pairMul_entry, reduceMatrix]
    have hdigits :
        ∑ a, D j a l * (reducePoly q n sm * joinedEntry (A 0) a + joinedEntry (A 1) a) =
        reducePoly q n sm * ∑ a, joinedEntry (A 0) a * D j a l +
          ∑ a, joinedEntry (A 1) a * D j a l := by
      rw [Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun a _ ↦ by ring
    rw [hdigits]
    ring

set_option maxHeartbeats 1000000 in
/-- A Dynamic private multiplication at a level that keeps its modulus: the row `c Dec(C) + E`
encodes `μ T` under the companion keys `M`. -/
theorem mul_private_dynamic {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {top bot : Fin 2 → ExactMatrix q n 1 m'} {D : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD : ∀ j l, ∃ D' : ExactMatrix q n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (top j 0 l) (bot j 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ D j a l))
    {T : Fin 2 → ExactMatrix q n 1 m'} {eB : Fin 2 → ErrorMatrix n 1 m'} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) {K : Nat} (hμ : SmulGain μ K)
    {E : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {f : Fin 2 → ErrorMatrix n 1 m'} {F : Nat}
    (hE : ∀ j, E j = reducePoly q n sm • (M 0 j - pairMul (A 0) (D j)) +
      (M 1 j - pairMul (A 1) (D j)) + reduceMatrix q n 1 m' (f j))
    (hf : ∀ j, CoeffBound (f j) F) :
    Rel sm M (fun j ↦ reducePoly q n μ • T j) (fun j ↦ pairMul c (D j) + E j)
      ((m + m) * n * B * bound + F + K * BB) := by
  choose d hd using fun j ↦ columns_bounded fun l ↦ (hD j l).choose_spec.2.2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := henc.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hq : q = 1 * q := (one_mul q).symm
  have ht : ∀ j,
      pairMul (fun i ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp i) • g) (D j) =
      reducePoly q n μ • T j + reduceMatrix q n 1 m' (μ • eB j) := by
    intro j
    have hgad := gadget_action_columns hq hq1 hn layout hlayout hexact hg' hg (hD j)
      (fun i ↦ μ * unitPair sp i)
    simp only [← mul_assoc] at hgad
    rw [hgad, scaleUpMatrix_one hq1 hn,
      reduceMatrix_smul, ← smul_add, ← hrhs j]
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one, smul_add,
      smul_smul]
  simp only [ht] at hact
  have hrek := hact.rekey (T := M) (E := E) hE hf
  exact (hrek.absorb (f := fun j ↦ μ • eB j) (fun j ↦ hμ (heB j))).mono (le_of_eq (by ring))

/-! ## A Dynamic LUT branch -/

set_option maxHeartbeats 1000000 in
/-- One Dynamic LUT branch. The routed input `σ(c)` is switched by the decomposition `D_j` of the
columns of the switching ciphertext `bsw` (`X`), the branch's companion row `Eσ` moves it to the
companion keys `M` under the substituted mask, `M` is moved down (`Asw`), and the helper `h`
encrypted for `σ` acts through the decomposition `Edec_j` of `Asw` (`Y`, `K`). The branch output
`Y + (X + Eσ)` is related under the original mask to the helper key `K`, with the target
`p σ(μ) (σ(r₀) bsw_{0j} + bsw_{1j})` lifted from the next modulus. -/
theorem lut_branch_dynamic {q q' p m m' log bound : Nat} (hn : 0 < n) (hpow : n = 2 ^ log)
    (hq : q = p * q') (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c B)
    {index : Nat} (hodd : Odd index)
    {bsw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {D : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD : ∀ j l, ∃ D' : ExactMatrix q' n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (bsw 0 j 0 l) (bsw 1 j 0 l)) D' ∧
        compactCenteredRebaseRuns bound D' (fun a _ ↦ D j a l))
    {X : Fin 2 → ExactMatrix q n 1 m'}
    (hX : ∀ j, X j = pairMul (fun i ↦ substituteMatrix index hodd (c i)) (D j))
    {Eσ : Fin 2 → ExactMatrix q n 1 m'} {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {f : Fin 2 → ErrorMatrix n 1 m'} {F : Nat}
    (hEσ : ∀ j, Eσ j = reducePoly q n (substitution index hodd sm) •
      (M 0 j - pairMul (fun i ↦ substituteMatrix index hodd (A 0 i)) (D j)) +
      (M 1 j - pairMul (fun i ↦ substituteMatrix index hodd (A 1 i)) (D j)) +
      reduceMatrix q n 1 m' (f j))
    (hf : ∀ j, CoeffBound (f j) F)
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Edec : Fin 2 → ExactMatrix q n (m + m) m'}
    (hEdec : ∀ j l, ∃ E' : ExactMatrix q' n (m + m) 1,
      gadgetDecomposeRuns backend base digits (pairColumn (Asw 0 j 0 l) (Asw 1 j 0 l)) E' ∧
        compactCenteredRebaseRuns bound E' (fun a _ ↦ Edec j a l))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
      unitPair (substitution index hodd sm) j) • g) h Bh)
    {Y : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, Y j = pairMul h (Edec j)) (hK : ∀ k j, K k j = pairMul (Ad k) (Edec j)) :
    Rel sm K (fun j ↦ scaleUpMatrix q p (reducePoly q' n (substitution index hodd μ) •
        (reducePoly q' n (substitution index hodd sp) • bsw 0 j + bsw 1 j)))
      (fun j ↦ Y j + (X j + Eσ j))
      ((m + m) * n * Bh * bound + ((m + m) * n * B * bound + F) + (n * 1 * (p / 2) + p / 2)) := by
  have hcop : n.Coprime index := by rw [hpow]; exact coprime_of_odd_pow_two hodd
  have hsub := henc.substitute hn hodd hcop
  -- The switching action on the routed encoding, moved to the companion keys.
  choose d hd using fun j ↦ columns_bounded fun l ↦ (hD j l).choose_spec.2.2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := hsub.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hrek := hact.rekey (T := M) (E := Eσ) hEσ hf
  -- The helper action.
  choose e he using fun j ↦ columns_bounded fun l ↦ (hEdec j l).choose_spec.2.2.2.2
  have hEe : ∀ j, Edec j = reduceMatrix q n (m + m) m' (e j) := fun j ↦ (he j).1
  have hhact := hhelp.action hn e (fun j ↦ (he j).2)
  simp only [← hEe] at hhact
  -- The routed target after the switching action.
  have htc : ∀ j, pairMul (fun i ↦ substituteMatrix index hodd
      (reducePoly q n ((p : Int) * μ * unitPair sp i) • g)) (D j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index hodd μ) •
        (reducePoly q' n (substitution index hodd sp) • bsw 0 j + bsw 1 j)) := by
    intro j
    simp only [substituteMatrix_target hn hg' hg]
    have hgad := gadget_action_columns hq hq' hn layout hlayout hexact hg' hg (hD j)
      (fun i ↦ substitution index hodd μ * substitution index hodd (unitPair sp i))
    simp only [← mul_assoc] at hgad
    rw [hgad]
    congr 1
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_one, map_mul, mul_one,
      smul_add, smul_smul]
  -- The helper target after its action.
  have hth : ∀ j, pairMul (fun i ↦ reducePoly q n ((p : Int) *
      unitPair (substitution index hodd sm) i) • g) (Edec j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index hodd sm) • Asw 0 j + Asw 1 j) := by
    intro j
    rw [gadget_action_columns hq hq' hn layout hlayout hexact hg' hg (hEdec j)]
    simp [unitPair]
  -- ModDown of the companion keys.
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  have hsw : ∀ k j, ∃ r : ErrorMatrix n 1 m', CoeffBound r (p / 2) ∧
      scaleUpMatrix q p (Asw k j) = M k j - reduceMatrix q n 1 m' r := by
    intro k j
    obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hAsw k j)
    rw [hpq] at hr hrel
    exact ⟨r, hr, hrel⟩
  have hsmσ : polyNorm (substitution index hodd sm) ≤ 1 :=
    (polyNorm_substitution_le hn hodd hcop sm).trans hsm
  have hbranch := Rel.lutBranch hn hq hq' hsmσ hrek hsw hhact hth
  simp only [htc] at hbranch
  have hYX : (fun j ↦ Y j + (X j + Eσ j)) = (fun j ↦ pairMul h (Edec j)) +
      ((fun j ↦ pairMul (fun i ↦ substituteMatrix index hodd (c i)) (D j)) + Eσ) := by
    funext j
    rw [Pi.add_apply, Pi.add_apply, hY j, hX j]
  have hKfun : K = fun k j ↦ pairMul (Ad k) (Edec j) := by
    funext k j
    exact hK k j
  rw [hYX, hKfun]
  exact hbranch

end MxxExponentLut
