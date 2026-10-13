import MxxIR
import ExponentDynamic
import GenRel
import GenIface

/-!
# Generated loops on encodings, for any parameter set

The loop roles of the generated stages as lemmas that take a `∀`-quantified step hypothesis about
the child scope (what a per-scope adapter proves by unfolding the child once) and conclude through
`MxxIR.IterRuns.invariant`. Nothing here names the ring dimension, a modulus, the number of
members, the number of columns, the block length or the number of digits, and nothing unrolls a
loop.

* `T6`, `T6.get`, `Layout6`, `RelT6`: the six matrices a generated loop carries for one encoding
  (two row entries and the four key entries, in an order the adapter names), and the encoding
  relation read through a layout. `RelT6.sum`: encodings of members add.
* `iter_affine_sum`: a loop whose iteration `i` adds member `a + b i` of a family accumulates
  those members; `block_sum_gen`: the member sums of the export (members `a`, `a + 1` added before
  the loop, `a + 2 + i` in iteration `i`) give the block `a, …, a + P - 1`.
* `placeCol`, `iter_column_sum`, `relT6_placed_sum`: the `Step::Sum` column loop. Iteration `i`
  places the block sum of column `c₀ + i` (`tensorRuns` with `unitRow (c₀ + i)`); the row of
  encodings is an encoding of the placed targets with the bound of one column (columns are
  disjoint). `tensor_unitRow_place` reads the tensor of a unit row as a placement.
* `gather_family`: a parallel Gather (`familyGetDynamic` at `tab j` for every `j`) is the family
  composed with the table; `gather_t6`: the six-family unpack of a member's cells.
* `RowEntries`, `rowEntries_concat`, `slice_rowEntries`: the digit weights of the smudge as a
  `concatColumns` fold (one step per digit, any number of digits) and the column of digit
  `o mod d`; `fmod_ofNat`, `fdiv_ofNat`: the exported digit and placement indices.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## Six-matrix states -/

/-- The six matrices a generated loop carries for one encoding. -/
abbrev T6 (α : Type) := α × α × α × α × α × α × Unit

/-- Component `k` of a six-matrix state. -/
def T6.get {α : Type} (s : T6 α) : Fin 6 → α :=
  ![s.1, s.2.1, s.2.2.1, s.2.2.2.1, s.2.2.2.2.1, s.2.2.2.2.2.1]

/-- The state with components `x`. -/
def T6.mk {α : Type} (x : Fin 6 → α) : T6 α := (x 0, x 1, x 2, x 3, x 4, x 5, ())

@[simp] theorem T6.get_mk {α : Type} (x : Fin 6 → α) (k : Fin 6) : (T6.mk x).get k = x k := by
  fin_cases k <;> rfl

theorem T6.ext {α : Type} {s t : T6 α} (h : ∀ k, s.get k = t.get k) : s = t := by
  obtain ⟨s0, s1, s2, s3, s4, s5, ⟨⟩⟩ := s
  obtain ⟨t0, t1, t2, t3, t4, t5, ⟨⟩⟩ := t
  have h0 := h 0; have h1 := h 1; have h2 := h 2; have h3 := h 3; have h4 := h 4; have h5 := h 5
  simp only [T6.get] at h0 h1 h2 h3 h4 h5
  simp_all

theorem T6.get_add {α : Type} [AddCommMonoid α] (s t : T6 α) (k : Fin 6) :
    (s + t).get k = s.get k + t.get k := by
  fin_cases k <;> rfl

theorem T6.get_zero {α : Type} [AddCommMonoid α] (k : Fin 6) : (0 : T6 α).get k = 0 := by
  fin_cases k <;> rfl

/-- Component `k` as an additive map. -/
def T6.getHom {α : Type} [AddCommMonoid α] (k : Fin 6) : T6 α →+ α where
  toFun s := s.get k
  map_zero' := T6.get_zero k
  map_add' s t := T6.get_add s t k

theorem T6.get_sum {α ι : Type} [AddCommMonoid α] (S : Finset ι) (f : ι → T6 α) (k : Fin 6) :
    (∑ i ∈ S, f i).get k = ∑ i ∈ S, (f i).get k :=
  map_sum (T6.getHom k) f S

/-- Which components of a six-matrix state are the row entries and the key entries. -/
structure Layout6 where
  row : Fin 2 → Fin 6
  key : Fin 2 → Fin 2 → Fin 6

/-- The layout `(c₀, c₁, A₀₀, A₀₁, A₁₀, A₁₁)` of the LUT loops. -/
def Layout6.std : Layout6 := ⟨![0, 1], ![![2, 3], ![4, 5]]⟩

/-- The layout `(A00, A01, A10, A11, c0, c1)` of `GenIface.EncRel` (the PRF cells). -/
def Layout6.enc : Layout6 := ⟨![4, 5], ![![0, 1], ![2, 3]]⟩

/-- The encoding relation of a six-matrix state read through a layout. -/
def RelT6 {q m : Nat} (sm : ErrorPoly n) (L : Layout6) (t : Fin 2 → ExactMatrix q n 1 m)
    (s : T6 (ExactMatrix q n 1 m)) (B : Nat) : Prop :=
  Rel sm (fun a j ↦ s.get (L.key a j)) t (fun j ↦ s.get (L.row j)) B

theorem RelT6.mono {q m : Nat} {sm : ErrorPoly n} {L : Layout6} {t : Fin 2 → ExactMatrix q n 1 m}
    {s : T6 (ExactMatrix q n 1 m)} {B B' : Nat} (h : RelT6 sm L t s B) (hB : B ≤ B') :
    RelT6 sm L t s B' :=
  Rel.mono h hB

/-- **Members add.** Encodings of the members of a finite set sum to an encoding of the summed
targets with the summed bounds. -/
theorem RelT6.sum {q m : Nat} {sm : ErrorPoly n} (L : Layout6) {ι : Type} [DecidableEq ι]
    (S : Finset ι) {t : ι → Fin 2 → ExactMatrix q n 1 m} {f : ι → T6 (ExactMatrix q n 1 m)}
    {B : ι → Nat} (h : ∀ i ∈ S, RelT6 sm L (t i) (f i) (B i)) :
    RelT6 sm L (∑ i ∈ S, t i) (∑ i ∈ S, f i) (∑ i ∈ S, B i) := by
  have hs := Rel.sum S h
  unfold RelT6
  have hkey : (fun a j ↦ (∑ i ∈ S, f i).get (L.key a j)) =
      ∑ i ∈ S, (fun a j ↦ (f i).get (L.key a j)) := by
    funext a j
    simp only [T6.get_sum, Finset.sum_apply]
  have hrow : (fun j ↦ (∑ i ∈ S, f i).get (L.row j)) = ∑ i ∈ S, (fun j ↦ (f i).get (L.row j)) := by
    funext j
    simp only [T6.get_sum, Finset.sum_apply]
  rw [hkey, hrow]
  exact hs

/-- Encodings with a uniform bound sum with the bound times the number of members. -/
theorem RelT6.sum_const {q m : Nat} {sm : ErrorPoly n} (L : Layout6) {ι : Type} [DecidableEq ι]
    (S : Finset ι) {t : ι → Fin 2 → ExactMatrix q n 1 m} {f : ι → T6 (ExactMatrix q n 1 m)}
    {B : Nat} (h : ∀ i ∈ S, RelT6 sm L (t i) (f i) B) :
    RelT6 sm L (∑ i ∈ S, t i) (∑ i ∈ S, f i) (S.card * B) := by
  have := RelT6.sum L S (B := fun _ ↦ B) h
  simpa [Finset.sum_const] using this

/-- `RelT6.sum_const` with the target summed pointwise, over all of a finite type. -/
theorem RelT6.sum_univ {q m : Nat} {sm : ErrorPoly n} (L : Layout6) {ι : Type} [Fintype ι]
    [DecidableEq ι] {t : ι → Fin 2 → ExactMatrix q n 1 m} {f : ι → T6 (ExactMatrix q n 1 m)}
    {B : Nat} (h : ∀ i, RelT6 sm L (t i) (f i) B) :
    RelT6 sm L (fun j ↦ ∑ i, t i j) (∑ i, f i) (Fintype.card ι * B) := by
  have := RelT6.sum_const L Finset.univ (fun i _ ↦ h i)
  rw [Finset.card_univ] at this
  have ht : (∑ i, t i) = fun j ↦ ∑ i, t i j := funext fun j ↦ Finset.sum_apply j _ _
  rw [ht] at this
  exact this

/-! ## Placement at `X^k` -/

/-- Every component of a six-matrix state multiplied by `X^e`. -/
noncomputable def T6.mono {q r c : Nat} (s : T6 (ExactMatrix q n r c)) (e : Int) :
    T6 (ExactMatrix q n r c) :=
  T6.mk fun k ↦ multiplyMonomial (s.get k) e

/-- **The placement `X^k` of a member** (`sampler.placement`): the target is multiplied by `X^k`
and the bound is kept. -/
theorem relT6_monomial (hn : 1 < n) {q m : Nat} {sm : ErrorPoly n} {L : Layout6}
    {t : Fin 2 → ExactMatrix q n 1 m} {s : T6 (ExactMatrix q n 1 m)} {B : Nat}
    (h : RelT6 sm L t s B) {k : Nat} (hk : k < 2 * n) :
    RelT6 sm L (fun j ↦ reducePoly q n (Xe (n := n) k) • t j) (T6.mono s (k : Int)) B := by
  unfold RelT6 T6.mono
  simp only [T6.get_mk]
  exact rel_monomial_gen hn h hk

/-! ## Member sums -/

/-- **A loop that adds members at an affine index.** If iteration `i` adds member `a + b i` of
`f` (`hstep`, what the adapter of the child proves from its Gather read), the loop accumulates
the members `a + b i`, `i < count`. -/
theorem iter_affine_sum {α : Type} [AddCommMonoid α] {K : Nat} (f : Fin K → α) (a b : Nat)
    {body : Nat → α → α → Prop} {count : Nat} {init final : α}
    (hfit : ∀ i < count, a + b * i < K)
    (hstep : ∀ i current next (hi : i < count), body i current next →
      next = current + f ⟨a + b * i, hfit i hi⟩)
    (h : MxxIR.IterRuns body count init final) :
    final = init + ∑ i : Fin count, f ⟨a + b * i, hfit i i.isLt⟩ := by
  classical
  let g : Nat → α := fun i ↦ if hi : i < count then f ⟨a + b * i, hfit i hi⟩ else 0
  have hinv := MxxIR.IterRuns.invariant
    (Invariant := fun k (s : α) ↦ k ≤ count → s = init + ∑ i ∈ Finset.range k, g i)
    (fun _ ↦ by simp) (fun i current next hcur hbody hk ↦ by
      have hi : i < count := by omega
      rw [hstep i current next hi hbody, hcur (by omega), Finset.sum_range_succ, add_assoc]
      simp only [g, dif_pos hi]) h le_rfl
  rw [hinv, Finset.sum_range]
  congr 1
  exact Finset.sum_congr rfl fun i _ ↦ by simp only [g, dif_pos i.isLt]

/-- The same loop when the child adds the member it reads at the `Int` index `a + b i` (the
exported Gather form `0 ≤ idx ∧ idx < K ∧ familyGetDynamic f idx x ∧ next = current + x`). -/
theorem iter_affine_sum_gather {α : Type} [AddCommMonoid α] {K : Nat} (f : Fin K → α) (a b : Nat)
    {body : Nat → α → α → Prop} {count : Nat} {init final : α}
    (hfit : ∀ i < count, a + b * i < K)
    (hstep : ∀ i current next, i < count → body i current next →
      ∃ x, familyGetDynamic f ((a : Int) + (b : Int) * (i : Int)) x ∧ next = current + x)
    (h : MxxIR.IterRuns body count init final) :
    final = init + ∑ i : Fin count, f ⟨a + b * i, hfit i i.isLt⟩ := by
  refine iter_affine_sum f a b hfit (fun i current next hi hbody ↦ ?_) h
  obtain ⟨x, ⟨pos, hpos, rfl⟩, rfl⟩ := hstep i current next hi hbody
  have hv : pos.val = a + b * i := by push_cast at hpos; omega
  rw [show pos = ⟨a + b * i, hfit i hi⟩ from Fin.ext hv]

/-- **A block sum of the export.** Members `a` and `a + 1` are added before the loop and the
loop's iteration `i` adds member `a + 2 + i` (`count = P - 2`): the block `a, …, a + P - 1`. -/
theorem block_sum_gen {α : Type} [AddCommMonoid α] {K : Nat} (f : Fin K → α) (a P : Nat)
    (hP : 2 ≤ P) (hK : a + P ≤ K) {body : Nat → α → α → Prop} {init final : α}
    (hinit : init = f ⟨a, by omega⟩ + f ⟨a + 1, by omega⟩)
    (hstep : ∀ i current next (hi : i < P - 2), body i current next →
      next = current + f ⟨a + 2 + 1 * i, by omega⟩)
    (h : MxxIR.IterRuns body (P - 2) init final) :
    final = ∑ j : Fin P, f ⟨a + j, by omega⟩ := by
  rw [iter_affine_sum f (a + 2) 1 (fun i hi ↦ by omega) hstep h, hinit]
  obtain ⟨P, rfl⟩ : ∃ P', P = P' + 2 := ⟨P - 2, by omega⟩
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ, ← add_assoc]
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  congr 1
  ext
  simp only [Fin.val_succ]
  omega

/-! ## The `Step::Sum` column loop -/

/-- A `1 × 1` value placed at column `c` of a `1 × C` row. -/
noncomputable def placeCol {q : Nat} (C c : Nat) (x : ExactMatrix q n 1 1) : ExactMatrix q n 1 C :=
  fun _ w ↦ if (w : Nat) = c then x 0 0 else 0

theorem placeCol_add {q : Nat} (C c : Nat) (x y : ExactMatrix q n 1 1) :
    placeCol C c (x + y) = placeCol C c x + placeCol C c y := by
  funext r w
  simp only [placeCol, Matrix.add_apply]
  split_ifs <;> simp

theorem placeCol_smul {q : Nat} (C c : Nat) (a : ExactPoly q n) (x : ExactMatrix q n 1 1) :
    placeCol C c (a • x) = a • placeCol C c x := by
  funext r w
  simp only [placeCol, Matrix.smul_apply, smul_eq_mul]
  split_ifs <;> simp

/-- The tensor of `unitRow c` (`1 × C`) with a `1 × 1` value places it at column `c`. -/
theorem tensor_unitRow_place {q C : Nat} {c : Nat} {x : ExactMatrix q n 1 1}
    {out : ExactMatrix q n (1 * 1) (C * 1)} (h : tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) x out)
    (w : Fin (C * 1)) : out 0 w = if (w : Nat) = c then x 0 0 else 0 := by
  rw [tensorRuns_unitRow h w]
  simp only [Nat.div_one]
  split_ifs
  · exact congrArg (x 0) (Subsingleton.elim _ _)
  · rfl

/-- A `(1 * 1) × (C * 1)` tensor output as a `1 × C` row. -/
def castRow {q C : Nat} (x : ExactMatrix q n (1 * 1) (C * 1)) : ExactMatrix q n 1 C :=
  fun r w ↦ x ⟨r, by simp⟩ ⟨w, by simp⟩

/-- Component-wise placement of a six-matrix state. -/
noncomputable def T6.place {q : Nat} (C c : Nat) (x : T6 (ExactMatrix q n 1 1)) :
    T6 (ExactMatrix q n 1 C) :=
  T6.mk fun k ↦ placeCol C c (x.get k)

/-- **The column loop.** Iteration `i` adds the value `x i` placed at column `c₀ + i`: the loop
places the values of its iterations. -/
theorem iter_column_sum {q C : Nat} (c₀ : Nat) {body : Nat → T6 (ExactMatrix q n 1 C) →
      T6 (ExactMatrix q n 1 C) → Prop} {count : Nat} {init final : T6 (ExactMatrix q n 1 C)}
    (x : Fin count → T6 (ExactMatrix q n 1 1))
    (hstep : ∀ i current next (hi : i < count), body i current next →
      next = current + T6.place C (c₀ + i) (x ⟨i, hi⟩))
    (h : MxxIR.IterRuns body count init final) :
    final = init + ∑ i : Fin count, T6.place C (c₀ + i) (x i) := by
  classical
  let g : Nat → T6 (ExactMatrix q n 1 C) :=
    fun i ↦ if hi : i < count then T6.place C (c₀ + i) (x ⟨i, hi⟩) else 0
  have hinv := MxxIR.IterRuns.invariant
    (Invariant := fun k (s : T6 (ExactMatrix q n 1 C)) ↦ k ≤ count →
      s = init + ∑ i ∈ Finset.range k, g i)
    (fun _ ↦ by simp) (fun i current next hcur hbody hk ↦ by
      have hi : i < count := by omega
      rw [hstep i current next hi hbody, hcur (by omega), Finset.sum_range_succ, add_assoc]
      simp only [g, dif_pos hi]) h le_rfl
  rw [hinv, Finset.sum_range]
  congr 1
  exact Finset.sum_congr rfl fun i _ ↦ by simp only [g, dif_pos i.isLt]

/-- **The full column loop.** Column `0` is placed before the loop and iteration `i` places
column `1 + i` (`count = C - 1`): the row holds every column. -/
theorem iter_column_sum_full {q C : Nat} (hC : 0 < C) {body : Nat → T6 (ExactMatrix q n 1 C) →
      T6 (ExactMatrix q n 1 C) → Prop} {init final : T6 (ExactMatrix q n 1 C)}
    (y : Fin C → T6 (ExactMatrix q n 1 1)) (hinit : init = T6.place C 0 (y ⟨0, hC⟩))
    (hstep : ∀ i current next (hi : i < C - 1), body i current next →
      next = current + T6.place C (1 + i) (y ⟨1 + i, by omega⟩))
    (h : MxxIR.IterRuns body (C - 1) init final) :
    final = ∑ c : Fin C, T6.place C c (y c) := by
  rw [iter_column_sum 1 (fun i ↦ y ⟨1 + i, by omega⟩) hstep h, hinit]
  obtain ⟨C, rfl⟩ : ∃ C', C = C' + 1 := ⟨C - 1, by omega⟩
  rw [Fin.sum_univ_succ]
  congr 1
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  have hv : 1 + (i : Nat) = ((Fin.succ i : Fin (C + 1)) : Nat) := by simp [Fin.val_succ]; omega
  have hf : (⟨1 + (i : Nat), by omega⟩ : Fin (C + 1)) = i.succ := Fin.ext hv
  rw [hf, hv]

/-- The family of six-matrix states read from six families. -/
def T6.fam {α : Type} {K : Nat} (f : Fin 6 → Fin K → α) : Fin K → T6 α :=
  fun o ↦ T6.mk fun k ↦ f k o

/-- The step of the column loop from the exported tensors: every component of the next state is
the current one plus the tensor of `unitRow (c₀ + i)` with the iteration's value. -/
theorem column_step_of_tensor {q C : Nat} (c : Nat) {current next : T6 (ExactMatrix q n 1 C)}
    {x : T6 (ExactMatrix q n 1 1)} {out : Fin 6 → ExactMatrix q n (1 * 1) (C * 1)}
    (hout : ∀ k, tensorRuns (unitRow (c : Int) : ExactMatrix q n 1 C) (x.get k) (out k))
    (hnext : ∀ k, next.get k = current.get k + castRow (out k)) :
    next = current + T6.place C c x := by
  apply T6.ext
  intro k
  rw [hnext k, T6.get_add]
  congr 1
  simp only [T6.place, T6.get_mk]
  funext r w
  rw [Subsingleton.elim r 0]
  have := tensor_unitRow_place (hout k) ⟨w, by simp⟩
  simp only [placeCol, castRow]
  exact this

/-- Column `w` of a row, as a `1 × 1` matrix. -/
def colOf {q C : Nat} (x : ExactMatrix q n 1 C) (w : Fin C) : ExactMatrix q n 1 1 :=
  fun r _ ↦ x r w

theorem colOf_sum {q C : Nat} {ι : Type} (S : Finset ι) (f : ι → ExactMatrix q n 1 C) (w : Fin C) :
    colOf (∑ i ∈ S, f i) w = ∑ i ∈ S, colOf (f i) w := by
  funext r c
  simp only [colOf, Matrix.sum_apply]

theorem colOf_placeCol {q C : Nat} (c : Nat) (x : ExactMatrix q n 1 1) (w : Fin C) :
    colOf (placeCol C c x) w = if (w : Nat) = c then x else 0 := by
  funext r c'
  rw [Subsingleton.elim r 0, Subsingleton.elim c' 0]
  simp only [colOf, placeCol]
  split_ifs <;> rfl

/-- **An encoding column by column.** A row is an encoding with bound `B` if each of its columns
is. -/
theorem rel_of_columns {q C : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 C}
    {t c : Fin 2 → ExactMatrix q n 1 C} {B : Nat}
    (h : ∀ w, Rel sm (fun a j ↦ colOf (A a j) w) (fun j ↦ colOf (t j) w) (fun j ↦ colOf (c j) w)
      B) : Rel sm A t c B := by
  choose e he hc using h
  refine ⟨fun j r w ↦ e w j r 0, fun j r w k ↦ he w j r 0 k, fun j ↦ ?_⟩
  funext r w
  have := congrFun (congrFun (hc w j) r) 0
  simpa [colOf, reduceMatrix, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply] using this

/-- **Placed encodings.** Encodings of the values of distinct columns, each with bound `B`,
placed and summed, are an encoding of the placed targets with bound `B`. -/
theorem relT6_placed_sum {q C : Nat} {sm : ErrorPoly n} (L : Layout6) {ι : Type} [DecidableEq ι]
    (S : Finset ι) (col : ι → Nat) (hcol : ∀ i ∈ S, ∀ j ∈ S, col i = col j → i = j)
    {t : ι → Fin 2 → ExactMatrix q n 1 1} {x : ι → T6 (ExactMatrix q n 1 1)} {B : Nat}
    (h : ∀ i ∈ S, RelT6 sm L (t i) (x i) B) :
    RelT6 sm L (fun j ↦ ∑ i ∈ S, placeCol C (col i) (t i j)) (∑ i ∈ S, T6.place C (col i) (x i))
      B := by
  classical
  apply rel_of_columns
  intro w
  have key : ∀ k, colOf ((∑ i ∈ S, T6.place C (col i) (x i)).get k) w =
      ∑ i ∈ S, if (w : Nat) = col i then (x i).get k else 0 := by
    intro k
    rw [T6.get_sum, colOf_sum]
    simp only [T6.place, T6.get_mk, colOf_placeCol]
  have tkey : ∀ j, colOf (∑ i ∈ S, placeCol C (col i) (t i j)) w =
      ∑ i ∈ S, if (w : Nat) = col i then t i j else 0 := by
    intro j
    rw [colOf_sum]
    simp only [colOf_placeCol]
  simp only [key, tkey]
  by_cases hex : ∃ i ∈ S, (w : Nat) = col i
  · obtain ⟨i0, hi0, hw⟩ := hex
    have hsum : ∀ {β : Type} [AddCommMonoid β] (F : ι → β),
        (∑ i ∈ S, if (w : Nat) = col i then F i else 0) = F i0 := by
      intro β _ F
      rw [Finset.sum_eq_single i0]
      · rw [if_pos hw]
      · intro i hi hne
        rw [if_neg]
        intro hw'
        exact hne (hcol i hi i0 hi0 (hw'.symm.trans hw))
      · intro hn; exact absurd hi0 hn
    simp only [hsum]
    exact h i0 hi0
  · have hzero : ∀ {β : Type} [AddCommMonoid β] (F : ι → β),
        (∑ i ∈ S, if (w : Nat) = col i then F i else 0) = 0 := by
      intro β _ F
      refine Finset.sum_eq_zero fun i hi ↦ ?_
      rw [if_neg]
      intro hw
      exact hex ⟨i, hi, hw⟩
    simp only [hzero]
    exact (Rel.zero (q := q) (m := 1) (sm := sm)).mono (Nat.zero_le B)

/-! ## Gather reads of whole families -/

/-- **A parallel Gather.** Reading the family at the table entry `tab j` for every `j` gives the
family composed with the table. -/
theorem gather_family {α : Type} {K F : Nat} (f : Fin K → α) (tab : Fin F → Fin K)
    (idx : Fin F → Int) (hidx : ∀ j, idx j = ((tab j : Nat) : Int)) {out : Fin F → α}
    (h : ∀ j, familyGetDynamic f (idx j) (out j)) : out = fun j ↦ f (tab j) := by
  funext j
  obtain ⟨pos, hpos, hout⟩ := h j
  rw [hout]
  exact congrArg f (Fin.ext (by have := hidx j; omega))

/-- **The unpack of a member's cells.** Six families read at the same table entry give the
member's six-matrix state. -/
theorem gather_t6 {α : Type} {K F : Nat} (f : Fin 6 → Fin K → α) (tab : Fin F → Fin K)
    (idx : Fin F → Int) (hidx : ∀ j, idx j = ((tab j : Nat) : Int)) {out : Fin F → T6 α}
    (h : ∀ j k, familyGetDynamic (f k) (idx j) ((out j).get k)) :
    out = fun j ↦ T6.mk fun k ↦ f k (tab j) := by
  funext j
  apply T6.ext
  intro k
  rw [T6.get_mk]
  have := gather_family (f k) tab idx hidx (out := fun j ↦ (out j).get k) (fun j ↦ h j k)
  exact congrFun this j

/-! ## The digit weights of the smudge -/

/-- The entries of a row are `K 0, K 1, …`. -/
def RowEntries {q j : Nat} (x : ExactMatrix q n 1 j) (K : Nat → ExactPoly q n) : Prop :=
  ∀ c : Fin j, x 0 c = K c

/-- The first weight. -/
theorem rowEntries_one {q : Nat} {K : Nat → ExactPoly q n} {x : ExactMatrix q n 1 1}
    (hx : x 0 0 = K 0) : RowEntries x K := by
  intro c
  rw [Subsingleton.elim c 0]
  exact hx

/-- **One step of the fold.** Appending the weight `K j` to a row of the first `j` weights gives
the first `j + 1`. -/
theorem rowEntries_concat {q j : Nat} {K : Nat → ExactPoly q n} {x : ExactMatrix q n 1 j}
    {y : ExactMatrix q n 1 1} {out : ExactMatrix q n 1 (j + 1)} (hx : RowEntries x K)
    (hy : y 0 0 = K j) (h : concatColumns x y out) : RowEntries out K := by
  intro c
  rw [h 0 c]
  split_ifs with hc
  · exact hx ⟨c, hc⟩
  · have : (c : Nat) = j := by omega
    refine (congrArg (y 0) (Subsingleton.elim _ 0)).trans ?_
    rw [hy, this]

/-- **The selected weight.** Column `k` of a row of weights is the `1 × 1` weight `K k`. -/
theorem slice_rowEntries {q d : Nat} {K : Nat → ExactPoly q n} {x : ExactMatrix q n 1 d}
    (hx : RowEntries x K) {k : Nat} (hk : k < d) {out : ExactMatrix q n 1 1}
    (h : sliceMatrix x 0 1 (k : Int) (1 + (k : Int)) out) : out = fun _ _ ↦ K k := by
  funext r c
  obtain ⟨-, -, -, -, -, -, -, -, hout⟩ := h
  rw [hout r c (by omega) (by simp; omega)]
  convert hx ⟨k, hk⟩ using 2
  ext; simp

/-- The exported digit index `Int.fmod i d`. -/
theorem fmod_ofNat (i d : Nat) : Int.fmod (Int.ofNat i) (d : Int) = ((i % d : Nat) : Int) := by
  rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
  simp

/-- The exported placement index `Int.fmod (Int.fdiv i d) N`. -/
theorem fdiv_fmod_ofNat (i d N : Nat) :
    Int.fmod (Int.fdiv (Int.ofNat i) (d : Int)) (N : Int) = ((i / d % N : Nat) : Int) := by
  rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
  rw [Int.fmod_eq_emod_of_nonneg _ (by omega)]
  simp

end RefreshTopProof.Gen

/-! ## The interface's column step (`GenIface.ColStep`) -/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-- **`GenIface.ColStep.sem`** (with `0 < k`, which the range part of the spec needs when the
member loop is empty). -/
theorem colStep_sem {q n m C F k W : Nat} (hk : 0 < k) (hW : 2 ≤ W) (i : Nat) (hC : 1 + i < C)
    {fams : Tup (Fin F → ExactMatrix q n 1 m) k} {s0 s1 init col : Tup (ExactMatrix q n 1 m) k}
    {t : Tup (ExactMatrix q n (1 * 1) (C * m)) k} {cur next : Tup (ExactMatrix q n (1 * 1) (C * m)) k}
    (hs0 : ∀ c, familyGetDynamic (fams.get c) ((W : Int) * (1 + i)) (s0.get c))
    (hs1 : ∀ c, familyGetDynamic (fams.get c) ((W : Int) * (1 + i) + 1) (s1.get c))
    (hinit : ∀ c, init.get c = s0.get c + s1.get c)
    (hloop : MemberSum.Spec fams (fun j ↦ (W : Int) * (1 + i) + 2 + j) (W - 2) init col)
    (ht : ∀ c, tensorRuns (unitRow (q := q) (n := n) (columns := C) ((1 + i : Nat) : Int))
      (col.get c) (t.get c))
    (hnext : ∀ c, next.get c = cur.get c + t.get c) :
    ColStep.Spec fams W i cur next := by
  obtain ⟨hb, hcol⟩ := hloop
  obtain ⟨W, rfl⟩ : ∃ W', W = W' + 2 := ⟨W - 2, by omega⟩
  simp only [Nat.add_sub_cancel] at hb hcol
  set a := (W + 2) * (1 + i) with ha
  have hcast : (((W + 2 : Nat) : Int) * (1 + (i : Int))) = (a : Int) := by rw [ha]; push_cast; ring
  simp only [hcast] at hs0 hs1 hb hcol
  have hF : a + (W + 2) ≤ F := by
    obtain ⟨pos, hpos, -⟩ := hs1 ⟨0, hk⟩
    have := pos.isLt
    rcases Nat.eq_zero_or_pos W with hW0 | hW0
    · omega
    · obtain ⟨-, hlt⟩ := hb (W - 1) (by omega)
      omega
  refine ⟨hF, hC, fun c ↦ ⟨t.get c, ?_, hnext c⟩⟩
  have hcol' : col.get c = ∑ j ∈ Finset.range (W + 2), famAt (fams.get c) (a + j) := by
    rw [hcol c, hinit c]
    obtain ⟨p0, hp0, e0⟩ := hs0 c
    obtain ⟨p1, hp1, e1⟩ := hs1 c
    have e0' : s0.get c = fams.get c ⟨a + 0, by omega⟩ := e0.trans (congrArg _ (Fin.ext (by
      show (p0 : Nat) = a + 0; omega)))
    have e1' : s1.get c = fams.get c ⟨a + (0 + 1), by omega⟩ := e1.trans (congrArg _ (Fin.ext (by
      show (p1 : Nat) = a + (0 + 1); omega)))
    rw [Finset.sum_range_succ', Finset.sum_range_succ', e0', e1', famAt_of_lt _ (by omega),
      famAt_of_lt _ (by omega)]
    have hsum : ∑ j ∈ Finset.range W, famAt (fams.get c) ((a : Int) + 2 + (j : Int)).toNat =
        ∑ j ∈ Finset.range W, famAt (fams.get c) (a + (j + 1 + 1)) := by
      refine Finset.sum_congr rfl fun j _ ↦ ?_
      congr 1
      omega
    rw [hsum]
    abel
  rw [← hcol']
  exact ht c

/-- The interface's `Tup` components are `T6` components. -/
theorem tup_get_eq_t6 {α : Type} (x : Tup α 6) (c : Fin 6) :
    x.get c = RefreshTopProof.Gen.T6.get (α := α) x c := by
  fin_cases c <;> rfl

/-- `EncRel` is `RelT6` in the layout `(A00, A01, A10, A11, c0, c1)`. -/
theorem encRel_iff_relT6 {n q m : Nat} (sm : ErrorPoly n) (x : Tup (ExactMatrix q n 1 m) 6)
    (t : Fin 2 → ExactMatrix q n 1 m) (B : Nat) :
    EncRel sm x t B ↔ RefreshTopProof.Gen.RelT6 sm RefreshTopProof.Gen.Layout6.enc t x B := by
  have hA : Enc.A x = fun a j ↦ RefreshTopProof.Gen.T6.get (α := ExactMatrix q n 1 m) x
      (RefreshTopProof.Gen.Layout6.enc.key a j) := by
    funext a j
    fin_cases a <;> fin_cases j <;> rfl
  have hc : Enc.c x = fun j ↦ RefreshTopProof.Gen.T6.get (α := ExactMatrix q n 1 m) x
      (RefreshTopProof.Gen.Layout6.enc.row j) := by
    funext j
    fin_cases j <;> rfl
  unfold EncRel RefreshTopProof.Gen.RelT6
  rw [hA, hc]

end RefreshTopProof.GenIface
