import MxxIR
import ExponentRuntime
import ExponentEncoding
import Mathlib

/-!
# The role interface of the generated refresh-to-the-top proof (L1)

Every scope of the exported stages (`Stage_*.lean`, `W_Requirement_0.lean`) is assigned a *role*
by `generator/xroles.py` from its atom multiset, loop shape and data flow. For each role this
file fixes

* `<Role>.Spec`: what a scope of that role computes, as a `Prop` over its inputs and outputs
  (parameters such as the ring dimension, the member count `M`, the cells per member `W`, the
  loop counts, the moduli and the matrix widths are arguments, never constants);
* `<Role>.sem`: the generic lemma deriving `Spec` from the scope's atoms. Loop roles take the
  child's `Spec` as a `∀`-quantified step hypothesis; nothing is unrolled over members, cells,
  coefficients or loop iterations. Only the components of a tuple (an encoding has six) are
  listed, through `Tup.get`.

The generated adapters (`Gen_<Stage>_Scopes.lean`, `generator/gen_adapters.py`) prove
`S.sem : S … → <Role>.Spec …` for every exported scope `S` by unfolding the scope once and
applying `<Role>.sem`; a loop parent passes its child's adapter as the step.

Tuples: an exported tuple `α × α × … × Unit` with `k` components is `Tup α k` by definition; a
tuple of `k` families `Fin F → α` is read at an index through `Tup.ev`. An encoding is a `Tup`
of six `1 × m` matrices `(A00, A01, A10, A11, c0, c1)`, read by `Enc.A` and `Enc.c`.

Ownership: the structural roles (gathers, sums, placements, decode, requirement) are proved here
or in `GenLoop`; the encoding-level roles (PRF cells, LUT rounds, samplers, smudges, key side,
encryption) are proved in `GenRel`, `GenRelLut`, `GenKey`, `GenEnc`; see `IFACE.md`.
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

/-- Index arithmetic of the export (`Int.ofNat` casts, literals): `rfl`, or `omega` after
normalizing the casts. -/
macro "gen_omega" : tactic =>
  `(tactic| first
    | rfl
    | omega
    | (simp only [Int.ofNat_eq_natCast, Nat.cast_ofNat, Nat.cast_add, Nat.cast_mul, Nat.cast_one] at *; omega)
    | (push_cast at *; omega))

/-! ## Tuples and families -/

/-- The exported tuple `α × … × α × Unit` with `k` components. -/
def Tup (α : Type) : Nat → Type
  | 0 => Unit
  | k + 1 => α × Tup α k

/-- Component `c` of a tuple. -/
def Tup.get {α : Type} : {k : Nat} → Tup α k → Fin k → α
  | 0, _, i => i.elim0
  | _ + 1, x, i => Fin.cases (motive := fun _ ↦ α) (Prod.fst x) (fun j ↦ Tup.get (Prod.snd x) j) i

/-- The tuple of the members at `j` of a tuple of families. -/
def Tup.ev {α : Type} {W : Nat} : {k : Nat} → Tup (Fin W → α) k → Fin W → Tup α k
  | 0, _, _ => ()
  | _ + 1, x, j => (Prod.fst x j, Tup.ev (Prod.snd x) j)

@[simp] theorem Tup.get_ev {α : Type} {W : Nat} :
    ∀ {k : Nat} (x : Tup (Fin W → α) k) (j : Fin W) (c : Fin k), (Tup.ev x j).get c = x.get c j
  | 0, _, _, c => c.elim0
  | _ + 1, x, j, c => by
    cases c using Fin.cases with
    | zero => rfl
    | succ c => exact Tup.get_ev (Prod.snd x) j c

/-- A family member by a natural index, zero out of range (sums over index ranges). -/
def famAt {α : Type} [Zero α] {F : Nat} (f : Fin F → α) (i : Nat) : α :=
  if h : i < F then f ⟨i, h⟩ else 0

theorem famAt_of_lt {α : Type} [Zero α] {F : Nat} (f : Fin F → α) {i : Nat} (h : i < F) :
    famAt f i = f ⟨i, h⟩ := dif_pos h

/-- The `2 × 2` public part of an encoding tuple `(A00, A01, A10, A11, c0, c1)`. -/
def Enc.A {q n m : Nat} (x : Tup (ExactMatrix q n 1 m) 6) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  ![![x.get 0, x.get 1], ![x.get 2, x.get 3]]

/-- The two ciphertext rows of an encoding tuple. -/
def Enc.c {q n m : Nat} (x : Tup (ExactMatrix q n 1 m) 6) : Fin 2 → ExactMatrix q n 1 m :=
  ![x.get 4, x.get 5]

/-- An encoding tuple encodes target `t` under mask `sm` with error bound `B`. -/
def EncRel {n q m : Nat} (sm : ErrorPoly n) (x : Tup (ExactMatrix q n 1 m) 6)
    (t : Fin 2 → ExactMatrix q n 1 m) (B : Nat) : Prop :=
  Rel sm (Enc.A x) t (Enc.c x) B

/-! ## Loop combinators (generic, proved) -/

/-- A sequential loop whose iteration `i` adds `f i` accumulates `Σ_{i < count} f i`. -/
theorem seq_sum {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {init final : α} (f : Nat → α)
    (hstep : ∀ i cur next, body i cur next → next = cur + f i)
    (h : MxxIR.IterRuns body count init final) : final = init + ∑ i ∈ Finset.range count, f i :=
  MxxIR.IterRuns.invariant (Invariant := fun k (s : α) ↦ s = init + ∑ i ∈ Finset.range k, f i)
    (by simp) (fun i cur next hcur hbody ↦ by
      show next = init + ∑ j ∈ Finset.range (i + 1), f j
      rw [hstep i cur next hbody, hcur, Finset.sum_range_succ, add_assoc]) h

/-- A sequential loop and a per-iteration invariant on the step only (`hstep` may use `i <
count`). -/
theorem seq_forall {σ : Type} {body : Nat → σ → σ → Prop} {count : Nat} {init final : σ}
    (P : Nat → σ → σ → Prop) (hstep : ∀ i cur next, body i cur next → P i cur next)
    (h : MxxIR.IterRuns body count init final) :
    MxxIR.IterRuns P count init final := by
  induction h with
  | zero => exact .zero _
  | step _ hb ih => exact .step ih (hstep _ _ _ hb)

/-- The Boolean conjunction as the export writes it. -/
theorem and_step (a b : Bool) :
    decide (((if a then 1 else 0 : Int) + (if b then 1 else 0)) = 2) = (a && b) := by
  cases a <;> cases b <;> decide

/-! ## Role `gather.unpack`: one member of each of `k` families -/

/-- Outputs are member `idx` of each input family (`idx` in range). -/
def GatherUnpack.Spec {α : Type} {F k : Nat} (fams : Tup (Fin F → α) k) (idx : Int)
    (out : Tup α k) : Prop :=
  ∃ h : idx.toNat < F, 0 ≤ idx ∧ ∀ c, out.get c = fams.get c ⟨idx.toNat, h⟩

theorem GatherUnpack.sem {α : Type} {F k : Nat} {fams : Tup (Fin F → α) k} {idx : Int}
    {out : Tup α k} (h0 : 0 ≤ idx) (h1 : idx < F)
    (hget : ∀ c, familyGetDynamic (fams.get c) idx (out.get c)) :
    GatherUnpack.Spec fams idx out := by
  refine ⟨by omega, h0, fun c ↦ ?_⟩
  obtain ⟨pos, hpos, hout⟩ := hget c
  rw [hout]
  exact congrArg _ (Fin.ext (by simp; omega))

/-! ## Role `sum.members`: a loop adding members `idx i` of `k` families -/

/-- One iteration: every component adds member `idx` of its family. -/
def MemberStep.Spec {α : Type} [Add α] {F k : Nat} (fams : Tup (Fin F → α) k) (idx : Int)
    (cur next : Tup α k) : Prop :=
  ∃ h : idx.toNat < F, 0 ≤ idx ∧ ∀ c, next.get c = cur.get c + fams.get c ⟨idx.toNat, h⟩

theorem MemberStep.sem {α : Type} [Add α] {F k : Nat} {fams : Tup (Fin F → α) k} {idx : Int}
    {cur next : Tup α k} {got : Tup α k} (h0 : 0 ≤ idx) (h1 : idx < F)
    (hget : ∀ c, familyGetDynamic (fams.get c) idx (got.get c))
    (hnext : ∀ c, next.get c = cur.get c + got.get c) :
    MemberStep.Spec fams idx cur next := by
  obtain ⟨h, -, hg⟩ := GatherUnpack.sem h0 h1 hget
  exact ⟨h, h0, fun c ↦ by rw [hnext c, hg c]⟩

/-- The loop: component `c` of the result is the initial component plus members `idx i`,
`i < count`. -/
def MemberSum.Spec {α : Type} [AddCommMonoid α] {F k : Nat} (fams : Tup (Fin F → α) k)
    (idx : Nat → Int) (count : Nat) (init final : Tup α k) : Prop :=
  (∀ i < count, 0 ≤ idx i ∧ (idx i).toNat < F) ∧
    ∀ c, final.get c = init.get c + ∑ i ∈ Finset.range count, famAt (fams.get c) (idx i).toNat

theorem MemberSum.sem {α : Type} [AddCommMonoid α] {F k : Nat} {fams : Tup (Fin F → α) k}
    {idx : Nat → Int} {count : Nat} {init final : Tup α k} {body : Nat → Tup α k → Tup α k → Prop}
    (h : MxxIR.IterRuns body count init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep.Spec fams (idx i) cur next) :
    MemberSum.Spec fams idx count init final := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun j (s : Tup α k) ↦ MemberSum.Spec fams idx j init s) ?_ ?_ h
  · exact ⟨fun i hi ↦ absurd hi (Nat.not_lt_zero _), fun c ↦ by simp⟩
  · intro i cur next ⟨hb, hs⟩ hbody
    obtain ⟨hlt, h0, hn⟩ := hstep i cur next hbody
    refine ⟨fun j hj ↦ ?_, fun c ↦ ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
      · exact hb j hj
      · exact ⟨h0, hlt⟩
    · rw [hn c, hs c, Finset.sum_range_succ, famAt_of_lt _ hlt, add_assoc]

/-- The single-family form (a plain state, e.g. the decode CRT sums `(fam, ())`). -/
def MemberStep1.Spec {α : Type} [Add α] {F : Nat} (fam : Fin F → α) (idx : Int) (cur next : α) :
    Prop :=
  ∃ h : idx.toNat < F, 0 ≤ idx ∧ next = cur + fam ⟨idx.toNat, h⟩

theorem MemberStep1.sem {α : Type} [Add α] {F : Nat} {fam : Fin F → α} {idx : Int}
    {cur next got : α} (h0 : 0 ≤ idx) (h1 : idx < F) (hget : familyGetDynamic fam idx got)
    (hnext : next = cur + got) : MemberStep1.Spec fam idx cur next := by
  obtain ⟨pos, hpos, hout⟩ := hget
  refine ⟨by omega, h0, ?_⟩
  rw [hnext, hout]
  exact congrArg _ (congrArg fam (Fin.ext (by simp; omega)))

def MemberSum1.Spec {α : Type} [AddCommMonoid α] {F : Nat} (fam : Fin F → α) (idx : Nat → Int)
    (count : Nat) (init final : α) : Prop :=
  (∀ i < count, 0 ≤ idx i ∧ (idx i).toNat < F) ∧
    final = init + ∑ i ∈ Finset.range count, famAt fam (idx i).toNat

theorem MemberSum1.sem {α : Type} [AddCommMonoid α] {F : Nat} {fam : Fin F → α}
    {idx : Nat → Int} {count : Nat} {init final : α} {body : Nat → α → α → Prop}
    (h : MxxIR.IterRuns body count init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep1.Spec fam (idx i) cur next) :
    MemberSum1.Spec fam idx count init final := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun j (s : α) ↦ MemberSum1.Spec fam idx j init s) ?_ ?_ h
  · exact ⟨fun i hi ↦ absurd hi (Nat.not_lt_zero _), by simp⟩
  · intro i cur next ⟨hb, hs⟩ hbody
    obtain ⟨hlt, h0, hn⟩ := hstep i cur next hbody
    refine ⟨fun j hj ↦ ?_, ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
      · exact hb j hj
      · exact ⟨h0, hlt⟩
    · rw [hn, hs, Finset.sum_range_succ, famAt_of_lt _ hlt, add_assoc]

/-! ## Role `prf.bucket-rowsum`: member `o`'s row of `W` cells -/

/-- Component `c` of member `o` is the sum of its `W` cells `W o + j` (every family has the
`M W` cells of a bucket). -/
def RowSum.Spec {α : Type} [AddCommMonoid α] {F k : Nat} (fams : Tup (Fin F → α) k) (W o : Nat)
    (out : Tup α k) : Prop :=
  W * o + W ≤ F ∧ ∀ c, out.get c = ∑ j ∈ Finset.range W, famAt (fams.get c) (W * o + j)

/-- The scope: a parallel unpack of the `W` cells at `cidx j = W o + j` (`hcells`), the static
reads of cells `0` and `1` (`hs0`, `hs1`), their sum as the initial state and the member loop
over cells `lidx i = 2 + i`, `i < W - 2` (`hloop`, through `MemberSum.sem`). -/
theorem rowSum_cell_index (W o j : Nat) : ((W : Int) * o + j).toNat = W * o + j := by
  rw [show (W : Int) * o + j = ((W * o + j : Nat) : Int) by push_cast; ring]
  exact Int.toNat_natCast _

theorem rowSum_core {α : Type} [AddCommMonoid α] {F k W : Nat} (o : Nat) (hW : 2 ≤ W)
    {fams : Tup (Fin F → α) k} {cells : Tup (Fin W → α) k} {s0 s1 init out : Tup α k}
    (hcells : ∀ j : Fin W, GatherUnpack.Spec fams ((W : Int) * o + j) (Tup.ev cells j))
    (hs0 : ∀ c, familyGetStatic (cells.get c) 0 (s0.get c))
    (hs1 : ∀ c, familyGetStatic (cells.get c) 1 (s1.get c))
    (hinit : ∀ c, init.get c = s0.get c + s1.get c)
    (hloop : MemberSum.Spec cells (fun i ↦ 2 + (i : Int)) (W - 2) init out) :
    RowSum.Spec fams W o out := by
  -- Cell `j` of the member is member `W o + j` of the bucket's families.
  have hcell : ∀ (j : Fin W) c, ∃ h : W * o + j < F, cells.get c j = fams.get c ⟨W * o + j, h⟩ := by
    intro j c
    obtain ⟨h, -, hg⟩ := hcells j
    have e := rowSum_cell_index W o j
    have hg' := hg c
    rw [Tup.get_ev] at hg'
    refine ⟨e ▸ h, hg'.trans (congrArg _ (Fin.ext e))⟩
  have hF : W * o + W ≤ F := by
    obtain ⟨h, -, -⟩ := hcells ⟨W - 1, by omega⟩
    rw [rowSum_cell_index] at h
    simp only at h
    omega
  refine ⟨hF, fun c ↦ ?_⟩
  have hfam : ∀ j (hj : j < W), famAt (cells.get c) j = famAt (fams.get c) (W * o + j) := by
    intro j hj
    obtain ⟨h, e⟩ := hcell ⟨j, hj⟩ c
    rw [famAt_of_lt _ hj, famAt_of_lt _ h, e]
  have hread : ∀ {x : α} (j : Nat) (hj : j < W), familyGetStatic (cells.get c) (j : Int) x →
      x = famAt (fams.get c) (W * o + j) := by
    intro x j hj hx
    obtain ⟨pos, hpos, rfl⟩ := hx
    have : pos.val = j := by exact_mod_cast hpos
    rw [← hfam j hj, famAt_of_lt _ hj]
    exact congrArg _ (Fin.ext this)
  obtain ⟨-, hsum⟩ := hloop
  rw [hsum c, hinit c, hread 0 (by omega) (hs0 c), hread 1 (by omega) (hs1 c)]
  obtain ⟨w, rfl⟩ : ∃ w, W = w + 2 := ⟨W - 2, by omega⟩
  rw [Finset.sum_range_succ', Finset.sum_range_succ', show w + 2 - 2 = w by omega,
    Finset.sum_congr rfl (fun i hi ↦ by
      rw [show (2 + (i : Int)).toNat = 2 + i by omega, hfam (2 + i) (by simp at hi; omega)])]
  have e : ∀ i, (w + 2) * o + (i + 1 + 1) = (w + 2) * o + (2 + i) := fun i ↦ by omega
  simp only [e, Nat.zero_add, add_zero]
  abel

theorem RowSum.sem {α : Type} [AddCommMonoid α] {F k W : Nat} (o : Nat) (hW : 2 ≤ W)
    {fams : Tup (Fin F → α) k} {cells : Tup (Fin W → α) k} {s0 s1 init out : Tup α k}
    {cidx lidx : Nat → Int} (hcidx : ∀ j, cidx j = (W : Int) * o + j)
    (hlidx : ∀ i, lidx i = 2 + (i : Int))
    (hcells : ∀ j : Fin W, GatherUnpack.Spec fams (cidx j) (Tup.ev cells j))
    (hs0 : ∀ c, familyGetStatic (cells.get c) 0 (s0.get c))
    (hs1 : ∀ c, familyGetStatic (cells.get c) 1 (s1.get c))
    (hinit : ∀ c, init.get c = s0.get c + s1.get c)
    (hloop : MemberSum.Spec cells lidx (W - 2) init out) :
    RowSum.Spec fams W o out := by
  have hc : cidx = fun j : Nat ↦ (W : Int) * o + j := funext hcidx
  have hl : lidx = fun i : Nat ↦ 2 + (i : Int) := funext hlidx
  subst hc hl
  exact rowSum_core o hW hcells hs0 hs1 hinit hloop

/-! ## Role `sum.columns`: the `Step::Sum` column loop -/

/-- One column `i`: the member sum of column `1 + i` (members `W (1 + i) + j`, `j < W`) placed
at block `1 + i` by the tensor with `unitRow (1 + i)` and added. -/
def ColStep.Spec {q n m C F k : Nat} (fams : Tup (Fin F → ExactMatrix q n 1 m) k) (W : Nat)
    (i : Nat) (cur next : Tup (ExactMatrix q n (1 * 1) (C * m)) k) : Prop :=
  W * (1 + i) + W ≤ F ∧ 1 + i < C ∧ ∀ c, ∃ t : ExactMatrix q n (1 * 1) (C * m),
    tensorRuns (unitRow (q := q) (n := n) (columns := C) (1 + (i : Int)))
      (∑ j ∈ Finset.range W, famAt (fams.get c) (W * (1 + i) + j)) t ∧
    next.get c = cur.get c + t

/-- The scope (one iteration `i`): reads of members `idx0 = W (1+i)` and `idx1 = W (1+i) + 1`,
their sum as the initial state of the member loop over `lidx j = W (1+i) + 2 + j`, `j < W - 2`,
and the tensor of the result with the unit row. -/
theorem ColStep.sem {q n m C F k W : Nat} (hk : 0 < k) (hW : 2 ≤ W) (i : Nat) (hC : 1 + (i : Int) < C)
    {fams : Tup (Fin F → ExactMatrix q n 1 m) k} {s0 s1 init col : Tup (ExactMatrix q n 1 m) k}
    {t : Tup (ExactMatrix q n (1 * 1) (C * m)) k} {cur next : Tup (ExactMatrix q n (1 * 1) (C * m)) k}
    {idx0 idx1 : Int} {lidx : Nat → Int} (hidx0 : idx0 = (W : Int) * (1 + i))
    (hidx1 : idx1 = (W : Int) * (1 + i) + 1) (hlidx : ∀ j, lidx j = (W : Int) * (1 + i) + 2 + j)
    (hs0 : ∀ c, familyGetDynamic (fams.get c) idx0 (s0.get c))
    (hs1 : ∀ c, familyGetDynamic (fams.get c) idx1 (s1.get c))
    (hinit : ∀ c, init.get c = s0.get c + s1.get c)
    (hloop : MemberSum.Spec fams lidx (W - 2) init col)
    (ht : ∀ c, tensorRuns (unitRow (q := q) (n := n) (columns := C) (1 + (i : Int)))
      (col.get c) (t.get c))
    (hnext : ∀ c, next.get c = cur.get c + t.get c) :
    ColStep.Spec fams W i cur next := by
  -- B2's proof (GenLoop.colStep_sem)
  have hl : lidx = fun j : Nat ↦ (W : Int) * (1 + i) + 2 + j := funext hlidx
  subst hidx0 hidx1 hl
  have hC' : 1 + i < C := by omega
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
  refine ⟨hF, hC', fun c ↦ ⟨t.get c, ?_, hnext c⟩⟩
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

/-- The slice form of a column step (the decomposition fold): block `1 + i` of `x` (columns
`m (1+i) … m (1+i) + m`) times `v`, placed at block `1 + i`. -/
def ColSliceStep.Spec {q n m C : Nat} (x : ExactMatrix q n 1 (C * m)) (v : ExactMatrix q n m 1)
    (i : Nat) (cur next : ExactMatrix q n (1 * 1) (C * 1)) : Prop :=
  1 + (i : Int) < C ∧ ∃ blk : ExactMatrix q n 1 m, ∃ t : ExactMatrix q n (1 * 1) (C * 1),
    sliceMatrix x 0 1 ((m : Int) * (1 + i)) ((m : Int) * (1 + i) + m) blk ∧
    tensorRuns (unitRow (q := q) (n := n) (columns := C) (1 + (i : Int))) (matrixMul blk v) t ∧
    next = cur + t

theorem ColSliceStep.sem {q n m C : Nat} (i : Nat) (hC : 1 + (i : Int) < C)
    {x : ExactMatrix q n 1 (C * m)} {v : ExactMatrix q n m 1} {blk : ExactMatrix q n 1 m}
    {t cur next : ExactMatrix q n (1 * 1) (C * 1)} {c0 c1 : Int}
    (hc0 : c0 = (m : Int) * (1 + i)) (hc1 : c1 = (m : Int) * (1 + i) + m)
    (hblk : sliceMatrix x 0 1 c0 c1 blk)
    (ht : tensorRuns (unitRow (q := q) (n := n) (columns := C) (1 + (i : Int))) (matrixMul blk v) t)
    (hnext : next = cur + t) : ColSliceStep.Spec x v i cur next := by
  subst hc0 hc1
  exact ⟨hC, blk, t, hblk, ht, hnext⟩

/-! ## Role `sampler.placement`: every component times `X^(o mod N)` -/

def Placement.Spec {q n r m k : Nat} (N : Int) (o : Nat) (inp out : Tup (ExactMatrix q n r m) k) :
    Prop :=
  ∀ c, out.get c = multiplyMonomial (inp.get c) (Int.fmod (o : Int) N)

theorem Placement.sem {q n r m k : Nat} (N : Int) (o : Nat) {inp out : Tup (ExactMatrix q n r m) k}
    (h : ∀ c, out.get c = multiplyMonomial (inp.get c) (Int.fmod (o : Int) N)) :
    Placement.Spec N o inp out := h

/-! ## Role `label.payload-const`: `±X^e` from member tables -/

def Payload.Spec {q n : Nat} (signs exps : List Int) (o : Nat) (out : ExactMatrix q n 1 1) : Prop :=
  o < signs.length ∧ o < exps.length ∧
    out = multiplyMonomial (liftInteger (MxxIR.selectInt (o : Int) signs)) (MxxIR.selectInt (o : Int) exps)

theorem Payload.sem {q n : Nat} {signs exps : List Int} {o : Nat} {out : ExactMatrix q n 1 1}
    (h1 : o < signs.length) (h2 : o < exps.length)
    (h : out = multiplyMonomial (liftInteger (MxxIR.selectInt (o : Int) signs))
      (MxxIR.selectInt (o : Int) exps)) : Payload.Spec signs exps o out :=
  ⟨h1, h2, h⟩

/-! ## Role `label.public-matrix`: a hash-sampled public matrix per member -/

def LabelPublic.Spec {q n r m : Nat} (model : HashModel) (tag : Blob)
    (comps : Nat → List HashTagComponent) (key : ByteArray) (o : Nat) (out : ExactMatrix q n r m) :
    Prop :=
  hashSample model tag (comps o) key out

theorem LabelPublic.sem {q n r m : Nat} {model : HashModel} {tag : Blob}
    {comps : Nat → List HashTagComponent} {key : ByteArray} {o : Nat} {out : ExactMatrix q n r m}
    (h : hashSample model tag (comps o) key out) : LabelPublic.Spec model tag comps key o out := h

/-! ## Decode roles -/

/-- `decode.coeff-round`, one coefficient `i`: add the bit `⌊(2 v_i + h) / d⌋ mod 2`. -/
def RoundStep.Spec {N : Nat} (vals : Fin N → Int) (h d : Int) (i : Nat) (cur next : Int) : Prop :=
  ∃ hi : i < N, d ≠ 0 ∧ next = cur + (vals ⟨i, hi⟩ * 2 + h) / d % 2

theorem RoundStep.sem {N : Nat} {vals : Fin N → Int} {h d : Int} (i : Nat) {cur next v : Int}
    (h0 : 0 ≤ (i : Int)) (h1 : (i : Int) < N) (hget : familyGetDynamic vals (i : Int) v)
    (hd : d ≠ 0) (hnext : next = cur + (v * 2 + h) / d % 2) : RoundStep.Spec vals h d i cur next := by
  obtain ⟨pos, hpos, hout⟩ := hget
  have hi : i < N := by omega
  refine ⟨hi, hd, ?_⟩
  have : pos = ⟨i, hi⟩ := Fin.ext (by simp only [Fin.val_mk]; omega)
  rw [hnext, hout, this]

/-- The rounded bit count of all `N` coefficients. -/
def roundSum {N : Nat} (vals : Fin N → Int) (h d : Int) : Int :=
  ∑ i : Fin N, (vals i * 2 + h) / d % 2

/-- `decode.column`, one column `i` of `x`: its coefficients' rounded bits added to the state. -/
def DecodeColumn.Spec {q n C : Nat} (x : ExactMatrix q n 1 C) (h d : Int) (i : Nat) (cur next : Int) :
    Prop :=
  i < C ∧ ∃ col : ExactMatrix q n 1 1, ∃ vals : Fin n → Int,
    sliceMatrix x 0 1 (i : Int) (1 + i) col ∧ polynomialValues false col vals ∧
    next = cur + roundSum vals h d

theorem DecodeColumn.sem {q n C : Nat} {x : ExactMatrix q n 1 C} {h d : Int} (i : Nat) (hi : i < C)
    {cur next : Int} {col : ExactMatrix q n 1 1} {vals : Fin n → Int}
    (hcol : sliceMatrix x 0 1 (i : Int) (1 + i) col) (hvals : polynomialValues false col vals)
    {body : Nat → Int → Int → Prop} (hloop : MxxIR.IterRuns body n cur next)
    (hstep : ∀ j c c', body j c c' → RoundStep.Spec vals h d j c c') :
    DecodeColumn.Spec x h d i cur next := by
  -- B5's proof (GenDecode.decodeColumn_sem)
  classical
  refine ⟨hi, col, vals, hcol, hvals, ?_⟩
  have hs := seq_sum (fun j ↦ if hj : j < n then (vals ⟨j, hj⟩ * 2 + h) / d % 2 else 0)
    (fun j c c' hb ↦ by
      obtain ⟨hj, -, rfl⟩ := hstep j c c' hb
      simp [hj]) hloop
  rw [hs, roundSum, Finset.sum_range (fun j ↦ if hj : j < n then (vals ⟨j, hj⟩ * 2 + h) / d % 2 else 0)]
  congr 1
  exact Finset.sum_congr rfl fun j _ ↦ by simp

/-! ## Requirement roles: Boolean all-loops -/

/-- One iteration of a requirement loop: the state stays true only if `P i` holds. -/
def ReqStep.Spec (P : Nat → Prop) (i : Nat) (cur next : Bool) : Prop :=
  next = true → cur = true ∧ P i

/-- A requirement loop: a true result means a true initial state and `P i` for all `i`. -/
def ReqAll.Spec (P : Nat → Prop) (count : Nat) (init final : Bool) : Prop :=
  final = true → init = true ∧ ∀ i < count, P i

theorem ReqAll.sem {P : Nat → Prop} {count : Nat} {init final : Bool}
    {body : Nat → Bool → Bool → Prop} (h : MxxIR.IterRuns body count init final)
    (hstep : ∀ i cur next, body i cur next → ReqStep.Spec P i cur next) : ReqAll.Spec P count init final := by
  refine MxxIR.IterRuns.invariant (Invariant := fun j s ↦ ReqAll.Spec P j init s) ?_ ?_ h
  · exact fun hs ↦ ⟨hs, fun i hi ↦ absurd hi (Nat.not_lt_zero _)⟩
  · intro i cur next hinv hb hn
    obtain ⟨hc, hp⟩ := hstep i cur next hb hn
    obtain ⟨hi, hall⟩ := hinv hc
    refine ⟨hi, fun j hj ↦ ?_⟩
    rcases Nat.lt_succ_iff_lt_or_eq.1 hj with hj | rfl
    · exact hall j hj
    · exact hp

/-- The export's conjunction step: `next = decide ((cur:1/0) + (b:1/0) = 2)`. -/
theorem ReqStep.of_and {P : Nat → Prop} {i : Nat} {cur next b : Bool} (hb : b = true → P i)
    (hnext : next = decide (((if cur then 1 else 0 : Int) + (if b then 1 else 0)) = 2)) :
    ReqStep.Spec P i cur next := by
  intro hn
  rw [and_step] at hnext
  subst hnext
  simp only [Bool.and_eq_true] at hn
  exact ⟨hn.1, hb hn.2⟩

-- BEGIN GENERATED COMP (generator/gen_adapters.py --iface; do not edit by hand)

/-- Computation bundle of role `enc.label-entry` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def EncLabelEntry.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (inputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 1 × ExactMatrix q0 n 1 1 × Int × Int × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 1 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 d0),
    MxxRuntime.gadgetMatrixRuns backend p0 p1 v0 ∧
    MxxRuntime.gaussianSample (p2) (p3) v1 ∧
    MxxRuntime.gaussianSample (p4) (p5) v2 ∧
    outputs = (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.1 inputs.1 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.2.1 inputs.2.2.1 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.1 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.multiplyMonomial ((MxxRuntime.liftInteger inputs.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)) inputs.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)) v0 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.1 inputs.2.1 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.2.1 inputs.2.2.2.1 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.2.2.2.1 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.multiplyMonomial ((MxxRuntime.liftInteger inputs.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)) inputs.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)) v0 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v2 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.liftInteger inputs.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)) inputs.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 1)), ())))))))

/-- Computation bundle of role `enc.lut-branch-key` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def EncLutBranchKey.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ExactMatrix q0 n 1 1 × ByteArray × ExactMatrix q0 n 1 1 × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 1) (v3 : ExactMatrix q1 n 1 d0) (v4 : ExactMatrix q0 n 1 d0) (v5 : ExactMatrix q0 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 1) (v9 : ExactMatrix q0 n 1 d0),
    MxxRuntime.hashSample (hashModel) p1 ([.u64Le (p0), .integer ((MxxIR.selectInt ((Int.ofNat i)) p2)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.1) v0 ∧
    MxxRuntime.hashSample (hashModel) p4 ([.u64Le (p3), .integer ((MxxIR.selectInt ((Int.ofNat i)) p5)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.1) v1 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p6)) inputs.1 v2 ∧
    MxxRuntime.gadgetMatrixRuns backend p7 p8 v3 ∧
    MxxRuntime.centeredRebaseRuns v3 v4 ∧
    MxxRuntime.gaussianSample (p9) (p10) v5 ∧
    MxxRuntime.hashSample (hashModel) p12 ([.u64Le (p11), .integer ((MxxIR.selectInt ((Int.ofNat i)) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.1) v6 ∧
    MxxRuntime.hashSample (hashModel) p15 ([.u64Le (p14), .integer ((MxxIR.selectInt ((Int.ofNat i)) p16)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.1) v7 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p17)) inputs.2.2.1 v8 ∧
    MxxRuntime.gaussianSample (p18) (p19) v9 ∧
    outputs = (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft inputs.1 v0 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.1 v1 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p20 : ExactMatrix q0 n 1 1)) v2 : ExactMatrix q0 n 1 1)) v4 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v5 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft inputs.1 v6 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft inputs.2.2.1 v7 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p21 : ExactMatrix q0 n 1 1)) v8 : ExactMatrix q0 n 1 1)) v4 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v9 : ExactMatrix q0 n 1 d0)), ()))

/-- Computation bundle of role `enc.lut-branch-key` (variant 1): the scope relation with dimensions, moduli and constants as parameters. -/
def EncLutBranchKey.Comp_1 (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (q2 : Nat) (q3 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ExactMatrix q1 n 1 1 × ByteArray × ExactMatrix q2 n 1 1 × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 1) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 1) (v3 : ExactMatrix q0 n 1 d0) (v4 : ExactMatrix q0 n 1 1) (v5 : ExactMatrix q3 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 d0) (v9 : ExactMatrix q0 n 1 d0) (v10 : ExactMatrix q0 n 1 1) (v11 : ExactMatrix q0 n 1 d0),
    MxxRuntime.modulusReduceRuns inputs.1 v0 ∧
    MxxRuntime.hashSample (hashModel) p1 ([.u64Le (p0), .integer ((MxxIR.selectInt ((Int.ofNat i)) p2)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.1) v1 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.1 v2 ∧
    MxxRuntime.hashSample (hashModel) p4 ([.u64Le (p3), .integer ((MxxIR.selectInt ((Int.ofNat i)) p5)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.1) v3 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p6)) v0 v4 ∧
    MxxRuntime.gadgetMatrixRuns backend p7 p8 v5 ∧
    MxxRuntime.centeredRebaseRuns v5 v6 ∧
    MxxRuntime.gaussianSample (p9) (p10) v7 ∧
    MxxRuntime.hashSample (hashModel) p12 ([.u64Le (p11), .integer ((MxxIR.selectInt ((Int.ofNat i)) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.1) v8 ∧
    MxxRuntime.hashSample (hashModel) p15 ([.u64Le (p14), .integer ((MxxIR.selectInt ((Int.ofNat i)) p16)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.1) v9 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p17)) v2 v10 ∧
    MxxRuntime.gaussianSample (p18) (p19) v11 ∧
    outputs = (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft v0 v1 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft v2 v3 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p20 : ExactMatrix q0 n 1 1)) v4 : ExactMatrix q0 n 1 1)) v6 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v7 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft v0 v8 : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft v2 v9 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p21 : ExactMatrix q0 n 1 1)) v10 : ExactMatrix q0 n 1 1)) v6 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v11 : ExactMatrix q0 n 1 d0)), ()))

/-- Computation bundle of role `lut.key-member.carrier` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyMemberCarrier.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (n2 : Nat) (d2 : Nat) (d3 : Nat) (q3 : Nat) (n3 : Nat) (d4 : Nat) (q4 : Nat) (n4 : Nat) (d5 : Nat) (q5 : Nat) (n5 : Nat) (d6 : Nat) (d7 : Nat) (q6 : Nat) (n6 : Nat) (d8 : Nat) (d9 : Nat) (q7 : Nat) (n7 : Nat) (d10 : Nat) (d11 : Nat) (q8 : Nat) (n8 : Nat) (d12 : Nat) (p0 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit → Prop) (inputs : ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 d0) (v3 : ExactMatrix q0 n 1 d0),
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, ()))))))))))))) next) (Int.toNat p0) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), ())))) (v0, (v1, (v2, (v3, ())))) ∧
    outputs = (v0, (v1, (v2, (v3, ()))))

/-- Computation bundle of role `lut.key-member.readout` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyMemberReadout.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (n2 : Nat) (d2 : Nat) (d3 : Nat) (q3 : Nat) (n3 : Nat) (d4 : Nat) (q4 : Nat) (n4 : Nat) (d5 : Nat) (q5 : Nat) (n5 : Nat) (d6 : Nat) (d7 : Nat) (q6 : Nat) (n6 : Nat) (d8 : Nat) (d9 : Nat) (q7 : Nat) (n7 : Nat) (d10 : Nat) (d11 : Nat) (q8 : Nat) (n8 : Nat) (d12 : Nat) (q9 : Nat) (q10 : Nat) (q11 : Nat) (q12 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit → Prop) (inputs : ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit) (outputs : ExactMatrix q9 n 1 d0 × ExactMatrix q10 n 1 d0 × ExactMatrix q11 n 1 d0 × ExactMatrix q12 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 d0) (v3 : ExactMatrix q0 n 1 d0) (v4 : ExactMatrix q9 n 1 d0) (v5 : ExactMatrix q10 n 1 d0) (v6 : ExactMatrix q11 n 1 d0) (v7 : ExactMatrix q12 n 1 d0),
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, ()))))))))))))) next) (Int.toNat p0) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), ())))) (v0, (v1, (v2, (v3, ())))) ∧
    MxxRuntime.blockModSwitchRuns p1 (1) v0 v4 ∧
    MxxRuntime.blockModSwitchRuns p2 (1) v1 v5 ∧
    MxxRuntime.blockModSwitchRuns p3 (1) v2 v6 ∧
    MxxRuntime.blockModSwitchRuns p4 (1) v3 v7 ∧
    outputs = (v4, (v5, (v6, (v7, ()))))

/-- Computation bundle of role `lut.key-member.sampler` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyMemberSampler.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (n2 : Nat) (d2 : Nat) (d3 : Nat) (q3 : Nat) (n3 : Nat) (d4 : Nat) (q4 : Nat) (n4 : Nat) (d5 : Nat) (q5 : Nat) (n5 : Nat) (d6 : Nat) (d7 : Nat) (q6 : Nat) (n6 : Nat) (d8 : Nat) (d9 : Nat) (q7 : Nat) (n7 : Nat) (d10 : Nat) (d11 : Nat) (q8 : Nat) (n8 : Nat) (d12 : Nat) (p0 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit → Prop) (inputs : ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 d0) (v3 : ExactMatrix q0 n 1 d0),
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, ()))))))))))))) next) (Int.toNat p0) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), ())))) (v0, (v1, (v2, (v3, ())))) ∧
    outputs = (((MxxRuntime.matrixAdd v0 inputs.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v1 inputs.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v2 inputs.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v3 inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), ()))))

/-- Computation bundle of role `lut.key-round.step` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyRoundStep.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (q2 : Nat) (d10 : Nat) (d11 : Nat) (q3 : Nat) (d12 : Nat) (q4 : Nat) (d13 : Nat) (q5 : Nat) (d14 : Nat) (q6 : Nat) (d15 : Nat) (q7 : Nat) (q8 : Nat) (q9 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ExactMatrix q1 n 1 d4 × ExactMatrix q1 n 1 d5 × ExactMatrix q1 n 1 d6 × ExactMatrix q1 n 1 d7 × ByteArray × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × (Fin d9 → ExactMatrix q2 n 1 d0) × ExactMatrix q0 n 1 d10 × ExactMatrix q0 n 1 d10 × (Fin d11 → ExactMatrix q3 n 1 d0) × (Fin d12 → ExactMatrix q4 n 1 d2) × (Fin d13 → ExactMatrix q5 n 1 d2) × Unit) (outputs : ExactMatrix q1 n 1 d4 × ExactMatrix q1 n 1 d5 × ExactMatrix q1 n 1 d6 × ExactMatrix q1 n 1 d7 × Unit) :=
  ∃ (v0 : ExactMatrix q1 n 1 d14) (v1 : ExactMatrix q1 n 1 d14) (v2 : ExactMatrix q1 n (1 + 1) d14) (v3 : ExactMatrix q0 n 1 d8) (v4 : ExactMatrix q0 n 1 d8) (v5 : ExactMatrix q0 n (1 + 1) d8) (v6 : ExactMatrix q2 n 1 d0) (v7 : ExactMatrix q2 n d8 d0) (v8 : ExactMatrix q0 n d8 d0) (v9 : ExactMatrix q0 n 1 d10) (v10 : ExactMatrix q0 n 1 d10) (v11 : ExactMatrix q0 n (1 + 1) d10) (v12 : ExactMatrix q3 n 1 d0) (v13 : ExactMatrix q3 n d10 d0) (v14 : ExactMatrix q0 n d10 d0) (v15 : ExactMatrix q0 n 1 d1) (v16 : ExactMatrix q6 n 1 d1) (v17 : ExactMatrix q6 n d14 d1) (v18 : ExactMatrix q1 n d14 d1) (v19 : ExactMatrix q1 n 1 d15) (v20 : ExactMatrix q1 n 1 d15) (v21 : ExactMatrix q1 n (1 + 1) d15) (v22 : ExactMatrix q0 n 1 d1) (v23 : ExactMatrix q7 n 1 d1) (v24 : ExactMatrix q7 n d15 d1) (v25 : ExactMatrix q1 n d15 d1) (v26 : ExactMatrix q1 n 1 d4) (v27 : ExactMatrix q4 n 1 d2) (v28 : ExactMatrix q4 n d8 d2) (v29 : ExactMatrix q0 n d8 d2) (v30 : ExactMatrix q5 n 1 d2) (v31 : ExactMatrix q5 n d10 d2) (v32 : ExactMatrix q0 n d10 d2) (v33 : ExactMatrix q0 n 1 d3) (v34 : ExactMatrix q8 n 1 d3) (v35 : ExactMatrix q8 n d14 d3) (v36 : ExactMatrix q1 n d14 d3) (v37 : ExactMatrix q0 n 1 d3) (v38 : ExactMatrix q9 n 1 d3) (v39 : ExactMatrix q9 n d15 d3) (v40 : ExactMatrix q1 n d15 d3) (v41 : ExactMatrix q1 n 1 d5) (v42 : ExactMatrix q1 n 1 d6) (v43 : ExactMatrix q1 n 1 d7),
    MxxRuntime.hashSample (hashModel) p1 ([.u64Le (p0), .integer ((MxxIR.selectInt ((Int.ofNat i)) p2)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.1) v0 ∧
    MxxRuntime.hashSample (hashModel) p4 ([.u64Le (p3), .integer ((MxxIR.selectInt ((Int.ofNat i)) p5)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.1) v1 ∧
    MxxRuntime.concatRows v0 v1 v2 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p6)) inputs.2.2.2.2.2.1 v3 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p7)) inputs.2.2.2.2.2.2.1 v4 ∧
    MxxRuntime.concatRows v3 v4 v5 ∧
    p8 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p9 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v6 ∧
    MxxRuntime.gadgetDecomposeRuns backend p10 p11 v6 v7 ∧
    MxxRuntime.compactCenteredRebaseRuns p12 v7 v8 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p13)) inputs.2.2.2.2.2.2.2.2.1 v9 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p14)) inputs.2.2.2.2.2.2.2.2.2.1 v10 ∧
    MxxRuntime.concatRows v9 v10 v11 ∧
    p15 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p16 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v12 ∧
    MxxRuntime.gadgetDecomposeRuns backend p17 p18 v12 v13 ∧
    MxxRuntime.compactCenteredRebaseRuns p19 v13 v14 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v5 v8 : ExactMatrix q0 n (1 + 1) d0)) ((MxxRuntime.matrixMul v11 v14 : ExactMatrix q0 n (1 + 1) d0)) : ExactMatrix q0 n (1 + 1) d0)) 0 1 0 p20 v15 ∧
    MxxRuntime.blockModSwitchRuns p21 (1) v15 v16 ∧
    MxxRuntime.gadgetDecomposeRuns backend p22 p23 v16 v17 ∧
    MxxRuntime.compactCenteredRebaseRuns p24 v17 v18 ∧
    MxxRuntime.hashSample (hashModel) p26 ([.u64Le (p25), .integer ((MxxIR.selectInt ((Int.ofNat i)) p27)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.1) v19 ∧
    MxxRuntime.hashSample (hashModel) p29 ([.u64Le (p28), .integer ((MxxIR.selectInt ((Int.ofNat i)) p30)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.1) v20 ∧
    MxxRuntime.concatRows v19 v20 v21 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v5 v8 : ExactMatrix q0 n (1 + 1) d0)) ((MxxRuntime.matrixMul v11 v14 : ExactMatrix q0 n (1 + 1) d0)) : ExactMatrix q0 n (1 + 1) d0)) 1 2 0 p31 v22 ∧
    MxxRuntime.blockModSwitchRuns p32 (1) v22 v23 ∧
    MxxRuntime.gadgetDecomposeRuns backend p33 p34 v23 v24 ∧
    MxxRuntime.compactCenteredRebaseRuns p35 v24 v25 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v18 : ExactMatrix q1 n (1 + 1) d1)) ((MxxRuntime.matrixMul v21 v25 : ExactMatrix q1 n (1 + 1) d1)) : ExactMatrix q1 n (1 + 1) d1)) 0 1 0 p36 v26 ∧
    p37 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p38 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v27 ∧
    MxxRuntime.gadgetDecomposeRuns backend p39 p40 v27 v28 ∧
    MxxRuntime.compactCenteredRebaseRuns p41 v28 v29 ∧
    p42 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p43 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v30 ∧
    MxxRuntime.gadgetDecomposeRuns backend p44 p45 v30 v31 ∧
    MxxRuntime.compactCenteredRebaseRuns p46 v31 v32 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v5 v29 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v11 v32 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 0 1 0 p47 v33 ∧
    MxxRuntime.blockModSwitchRuns p48 (1) v33 v34 ∧
    MxxRuntime.gadgetDecomposeRuns backend p49 p50 v34 v35 ∧
    MxxRuntime.compactCenteredRebaseRuns p51 v35 v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v5 v29 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v11 v32 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 1 2 0 p52 v37 ∧
    MxxRuntime.blockModSwitchRuns p53 (1) v37 v38 ∧
    MxxRuntime.gadgetDecomposeRuns backend p54 p55 v38 v39 ∧
    MxxRuntime.compactCenteredRebaseRuns p56 v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v36 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v21 v40 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 0 1 0 p57 v41 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v18 : ExactMatrix q1 n (1 + 1) d1)) ((MxxRuntime.matrixMul v21 v25 : ExactMatrix q1 n (1 + 1) d1)) : ExactMatrix q1 n (1 + 1) d1)) 1 2 0 p58 v42 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v36 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v21 v40 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 1 2 0 p59 v43 ∧
    outputs = (((MxxRuntime.matrixAdd inputs.1 v26 : ExactMatrix q1 n 1 d4)), (((MxxRuntime.matrixAdd inputs.2.1 v41 : ExactMatrix q1 n 1 d5)), (((MxxRuntime.matrixAdd inputs.2.2.1 v42 : ExactMatrix q1 n 1 d6)), (((MxxRuntime.matrixAdd inputs.2.2.2.1 v43 : ExactMatrix q1 n 1 d7)), ()))))

/-- Computation bundle of role `lut.key-table` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyTable.Comp (d0 : Nat) (q0 : Nat) (n : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (q3 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 1 × (Fin d0 → Int) × ExactMatrix q1 n1 1 1 × Int × Unit → ExactMatrix q0 n 1 1 → Prop) (table : Nat → Int → Prop) (inputs : ByteArray × ExactMatrix q2 n 1 1 × ExactMatrix q3 n 1 1 × Unit) (outputs : ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d1) (v1 : ExactMatrix q0 n 1 d1) (v2 : ExactMatrix q0 n 1 1) (v3 : ExactMatrix q0 n 1 1) (v4 : (Fin d0 → Int)) (v5 : ExactMatrix q0 n 1 1) (v6 : ExactMatrix q0 n 1 d1) (v7 : ExactMatrix q0 n 1 d1) (v8 : ExactMatrix q0 n 1 1) (v9 : ExactMatrix q0 n 1 d1),
    MxxRuntime.hashSample (hashModel) p1 ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i)) p0)), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v0 ∧
    MxxRuntime.hashSample (hashModel) p3 ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i)) p2)), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v1 ∧
    MxxRuntime.modulusReduceRuns inputs.2.1 v2 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p4)) v2 v3 ∧
    (∀ i : Fin d0, table i (v4 i)) ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 1) => child i (current, (v4, (((MxxRuntime.matrixPolynomial p5 : ExactMatrix q1 n1 1 1)), ((-p6), ())))) next) (Int.toNat p7) (((0 : ExactMatrix q0 n 1 1))) (v5) ∧
    MxxRuntime.gadgetMatrixRuns backend p8 p9 v6 ∧
    MxxRuntime.gaussianSample (p10) (p11) v7 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.1 v8 ∧
    MxxRuntime.gaussianSample (p12) (p13) v9 ∧
    outputs = (v0, (v1, (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixNeg v3 : ExactMatrix q0 n 1 1)) v0 : ExactMatrix q0 n 1 d1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p14 : ExactMatrix q0 n 1 1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p15 : ExactMatrix q0 n 1 1)) v5 : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1)) v2 : ExactMatrix q0 n 1 1)) v6 : ExactMatrix q0 n 1 d1)) : ExactMatrix q0 n 1 d1)) v7 : ExactMatrix q0 n 1 d1)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixNeg v3 : ExactMatrix q0 n 1 1)) v1 : ExactMatrix q0 n 1 d1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p16 : ExactMatrix q0 n 1 1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p17 : ExactMatrix q0 n 1 1)) v5 : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1)) v8 : ExactMatrix q0 n 1 1)) v6 : ExactMatrix q0 n 1 d1)) : ExactMatrix q0 n 1 d1)) v9 : ExactMatrix q0 n 1 d1)), ()))))

/-- Computation bundle of role `lut.key-table` (variant 1): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyTable.Comp_1 (d0 : Nat) (q0 : Nat) (n : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (q3 : Nat) (q4 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 1 × ExactMatrix q1 n1 1 1 × Int × (Fin d0 → Int) × Unit → ExactMatrix q0 n 1 1 → Prop) (table : Nat → Int → Prop) (inputs : ByteArray × ExactMatrix q2 n 1 1 × ExactMatrix q3 n 1 1 × Unit) (outputs : ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d1) (v1 : ExactMatrix q0 n 1 d1) (v2 : ExactMatrix q0 n 1 1) (v3 : ExactMatrix q0 n 1 1) (v4 : (Fin d0 → Int)) (v5 : ExactMatrix q0 n 1 1) (v6 : ExactMatrix q4 n 1 d1) (v7 : ExactMatrix q0 n 1 d1) (v8 : ExactMatrix q0 n 1 d1) (v9 : ExactMatrix q0 n 1 1) (v10 : ExactMatrix q0 n 1 d1),
    MxxRuntime.hashSample (hashModel) p1 ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i)) p0)), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v0 ∧
    MxxRuntime.hashSample (hashModel) p3 ([.bytes [115, 105, 103, 109, 97], .u64Le ((MxxIR.selectInt ((Int.ofNat i)) p2)), .bytes [98, 114, 97, 110, 99, 104], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v1 ∧
    MxxRuntime.modulusReduceRuns inputs.2.1 v2 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p4)) v2 v3 ∧
    (∀ i : Fin d0, table i (v4 i)) ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 1) => child i (current, (((MxxRuntime.matrixPolynomial p5 : ExactMatrix q1 n1 1 1)), ((-p6), (v4, ())))) next) (Int.toNat p7) (((0 : ExactMatrix q0 n 1 1))) (v5) ∧
    MxxRuntime.gadgetMatrixRuns backend p8 p9 v6 ∧
    MxxRuntime.centeredRebaseRuns v6 v7 ∧
    MxxRuntime.gaussianSample (p10) (p11) v8 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.1 v9 ∧
    MxxRuntime.gaussianSample (p12) (p13) v10 ∧
    outputs = (v0, (v1, (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixNeg v3 : ExactMatrix q0 n 1 1)) v0 : ExactMatrix q0 n 1 d1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p14 : ExactMatrix q0 n 1 1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p15 : ExactMatrix q0 n 1 1)) v5 : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1)) v2 : ExactMatrix q0 n 1 1)) v7 : ExactMatrix q0 n 1 d1)) : ExactMatrix q0 n 1 d1)) v8 : ExactMatrix q0 n 1 d1)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixNeg v3 : ExactMatrix q0 n 1 1)) v1 : ExactMatrix q0 n 1 d1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p16 : ExactMatrix q0 n 1 1)) ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p17 : ExactMatrix q0 n 1 1)) v5 : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1)) v9 : ExactMatrix q0 n 1 1)) v7 : ExactMatrix q0 n 1 d1)) : ExactMatrix q0 n 1 d1)) v10 : ExactMatrix q0 n 1 d1)), ()))))

/-- Computation bundle of role `lut.key-table.index` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyTableIndex.Comp  (p0 : _) (p1 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : Unit) (outputs : Int) :=
  outputs = ((((MxxIR.selectInt ((Int.ofNat i)) p0)) + (p1)))

/-- Computation bundle of role `lut.key-table.step` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyTableStep.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (p0 : _) (p1 : _) (p2 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (j1 : Nat) (i : Nat) (inputs : ExactMatrix q0 n 1 1 × (Fin d0 → Int) × ExactMatrix q0 n 1 1 × Int × Unit) (outputs : ExactMatrix q0 n 1 1) :=
  ∃ (v0 : Int) (v1 : ExactMatrix q0 n 1 1),
    p0 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p1 ∧
    MxxRuntime.familyGetDynamic inputs.2.1 ((Int.ofNat i)) v0 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat j1)) p2)) ((MxxRuntime.multiplyMonomial inputs.2.2.1 ((((Int.ofNat i)) * inputs.2.2.2.1)) : ExactMatrix q0 n 1 1)) v1 ∧
    outputs = ((MxxRuntime.matrixAdd inputs.1 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.liftInteger v0 : ExactMatrix q0 n 1 1)) v1 : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1))

/-- Computation bundle of role `lut.key-table.step` (variant 1): the scope relation with dimensions, moduli and constants as parameters. -/
def LutKeyTableStep.Comp_1 (q0 : Nat) (n : Nat) (d0 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (j1 : Nat) (i : Nat) (inputs : ExactMatrix q0 n 1 1 × ExactMatrix q0 n 1 1 × Int × (Fin d0 → Int) × Unit) (outputs : ExactMatrix q0 n 1 1) :=
  ∃ (v0 : ExactMatrix q0 n 1 1) (v1 : Int),
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat j1)) p0)) ((MxxRuntime.multiplyMonomial inputs.2.1 ((((Int.ofNat i)) * inputs.2.2.1)) : ExactMatrix q0 n 1 1)) v0 ∧
    p1 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p2 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.1 ((Int.ofNat i)) v1 ∧
    outputs = ((MxxRuntime.matrixAdd inputs.1 ((MxxRuntime.multiplyMonomial v0 ((v1 * (p3))) : ExactMatrix q0 n 1 1)) : ExactMatrix q0 n 1 1))

/-- Computation bundle of role `lut.member.readout` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutMemberReadout.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (q2 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (d13 : Nat) (d14 : Nat) (q3 : Nat) (q4 : Nat) (q5 : Nat) (q6 : Nat) (q7 : Nat) (q8 : Nat) (q9 : Nat) (q10 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (p63 : _) (p64 : _) (p65 : _) (p66 : _) (p67 : _) (p68 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit → Prop) (inputs : (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × Unit) (outputs : ExactMatrix q3 n 1 d0 × ExactMatrix q4 n 1 d0 × ExactMatrix q5 n 1 d0 × ExactMatrix q6 n 1 d0 × ExactMatrix q7 n 1 d0 × ExactMatrix q8 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q1 n 1 d1) (v1 : ExactMatrix q2 n 1 d2) (v2 : ExactMatrix q1 n 1 d1) (v3 : ExactMatrix q2 n 1 d2) (v4 : ExactMatrix q0 n 1 d0) (v5 : ExactMatrix q0 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 d0) (v9 : ExactMatrix q0 n 1 d0) (v10 : ExactMatrix q1 n 1 d1) (v11 : ExactMatrix q2 n 1 d2) (v12 : ExactMatrix q1 n 1 d1) (v13 : ExactMatrix q2 n 1 d2) (v14 : ExactMatrix q0 n 1 d4) (v15 : ExactMatrix q0 n 1 d4) (v16 : ExactMatrix q0 n 1 d4) (v17 : ExactMatrix q0 n ((1 + 1) + 1) d4) (v18 : ExactMatrix q0 n (1 + 1) d4) (v19 : ExactMatrix q0 n 1 d6) (v20 : ExactMatrix q0 n 1 d6) (v21 : ExactMatrix q0 n 1 d6) (v22 : ExactMatrix q0 n ((1 + 1) + 1) d6) (v23 : ExactMatrix q0 n (1 + 1) d6) (v24 : ExactMatrix q0 n ((1 + 1) + 1) (d4 + d6)) (v25 : ExactMatrix q0 n 1 d7) (v26 : ExactMatrix q0 n 1 d7) (v27 : ExactMatrix q0 n 1 d7) (v28 : ExactMatrix q0 n ((1 + 1) + 1) d7) (v29 : ExactMatrix q0 n (1 + 1) d7) (v30 : ExactMatrix q0 n 1 d8) (v31 : ExactMatrix q0 n 1 d8) (v32 : ExactMatrix q0 n 1 d8) (v33 : ExactMatrix q0 n ((1 + 1) + 1) d8) (v34 : ExactMatrix q0 n (1 + 1) d8) (v35 : ExactMatrix q0 n ((1 + 1) + 1) (d7 + d8)) (v36 : ExactMatrix q1 n (1 + 1) d1) (v37 : ExactMatrix q1 n (d7 + d8) d1) (v38 : ExactMatrix q0 n (d7 + d8) d1) (v39 : ExactMatrix q0 n 1 d13) (v40 : ExactMatrix q9 n 1 d13) (v41 : ExactMatrix q0 n 1 d13) (v42 : ExactMatrix q9 n 1 d13) (v43 : ExactMatrix q9 n (1 + 1) d13) (v44 : ExactMatrix q9 n (d4 + d6) d13) (v45 : ExactMatrix q0 n (d4 + d6) d13) (v46 : ExactMatrix q0 n 1 d0) (v47 : ExactMatrix q3 n 1 d0) (v48 : ExactMatrix q2 n (1 + 1) d2) (v49 : ExactMatrix q2 n (d7 + d8) d2) (v50 : ExactMatrix q0 n (d7 + d8) d2) (v51 : ExactMatrix q0 n 1 d14) (v52 : ExactMatrix q10 n 1 d14) (v53 : ExactMatrix q0 n 1 d14) (v54 : ExactMatrix q10 n 1 d14) (v55 : ExactMatrix q10 n (1 + 1) d14) (v56 : ExactMatrix q10 n (d4 + d6) d14) (v57 : ExactMatrix q0 n (d4 + d6) d14) (v58 : ExactMatrix q0 n 1 d0) (v59 : ExactMatrix q4 n 1 d0) (v60 : ExactMatrix q0 n 1 d0) (v61 : ExactMatrix q5 n 1 d0) (v62 : ExactMatrix q0 n 1 d0) (v63 : ExactMatrix q6 n 1 d0) (v64 : ExactMatrix q0 n 1 d0) (v65 : ExactMatrix q0 n 1 d0) (v66 : ExactMatrix q7 n 1 d0) (v67 : ExactMatrix q0 n 1 d0) (v68 : ExactMatrix q0 n 1 d0) (v69 : ExactMatrix q8 n 1 d0),
    MxxRuntime.familyGetStatic inputs.1 (p0) v0 ∧
    MxxRuntime.familyGetStatic inputs.2.1 (p1) v1 ∧
    MxxRuntime.familyGetStatic inputs.2.2.1 (p2) v2 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.1 (p3) v3 ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, ()))))))))))))))))))))))) next) (Int.toNat p4) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (v0, (v1, (v2, (v3, ())))))))))) (v4, (v5, (v6, (v7, (v8, (v9, (v10, (v11, (v12, (v13, ())))))))))) ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.1 (p5) v14 ∧
    MxxRuntime.hashSample (hashModel) p7 ([.u64Le (p6), .integer ((MxxIR.selectInt (p8) p9)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.1) v15 ∧
    MxxRuntime.hashSample (hashModel) p11 ([.u64Le (p10), .integer ((MxxIR.selectInt (p12) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.2.1) v16 ∧
    MxxRuntime.concatRows v14 v15 v18 ∧
    MxxRuntime.concatRows v18 v16 v17 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.2.2.1 (p14) v19 ∧
    MxxRuntime.hashSample (hashModel) p16 ([.u64Le (p15), .integer ((MxxIR.selectInt (p17) p18)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.1) v20 ∧
    MxxRuntime.hashSample (hashModel) p20 ([.u64Le (p19), .integer ((MxxIR.selectInt (p21) p22)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.2.1) v21 ∧
    MxxRuntime.concatRows v19 v20 v23 ∧
    MxxRuntime.concatRows v23 v21 v22 ∧
    MxxRuntime.concatColumns v17 v22 v24 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p23) p24)) inputs.2.2.2.2.2.2.2.1 v25 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p25) p26)) inputs.2.2.2.2.2.2.2.2.1 v26 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p27) p28)) inputs.2.2.2.2.2.2.2.2.2.1 v27 ∧
    MxxRuntime.concatRows v25 v26 v29 ∧
    MxxRuntime.concatRows v29 v27 v28 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p29) p30)) inputs.2.2.2.2.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p31) p32)) inputs.2.2.2.2.2.2.2.2.2.2.2.1 v31 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p33) p34)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 v32 ∧
    MxxRuntime.concatRows v30 v31 v34 ∧
    MxxRuntime.concatRows v34 v32 v33 ∧
    MxxRuntime.concatColumns v28 v33 v35 ∧
    MxxRuntime.concatRows v10 v12 v36 ∧
    MxxRuntime.gadgetDecomposeRuns backend p35 p36 v36 v37 ∧
    MxxRuntime.compactCenteredRebaseRuns p37 v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 1 2 0 p38 v39 ∧
    MxxRuntime.blockModSwitchRuns p39 (1) v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 2 3 0 p40 v41 ∧
    MxxRuntime.blockModSwitchRuns p41 (1) v41 v42 ∧
    MxxRuntime.concatRows v40 v42 v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p42 p43 v43 v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p44 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 1 2 0 p45 v46 ∧
    MxxRuntime.blockModSwitchRuns p46 (1) ((MxxRuntime.matrixAdd v6 v46 : ExactMatrix q0 n 1 d0)) v47 ∧
    MxxRuntime.concatRows v11 v13 v48 ∧
    MxxRuntime.gadgetDecomposeRuns backend p47 p48 v48 v49 ∧
    MxxRuntime.compactCenteredRebaseRuns p49 v49 v50 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v50 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p50 v51 ∧
    MxxRuntime.blockModSwitchRuns p51 (1) v51 v52 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v50 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p52 v53 ∧
    MxxRuntime.blockModSwitchRuns p53 (1) v53 v54 ∧
    MxxRuntime.concatRows v52 v54 v55 ∧
    MxxRuntime.gadgetDecomposeRuns backend p54 p55 v55 v56 ∧
    MxxRuntime.compactCenteredRebaseRuns p56 v56 v57 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v57 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 1 2 0 p57 v58 ∧
    MxxRuntime.blockModSwitchRuns p58 (1) ((MxxRuntime.matrixAdd v7 v58 : ExactMatrix q0 n 1 d0)) v59 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 2 3 0 p59 v60 ∧
    MxxRuntime.blockModSwitchRuns p60 (1) ((MxxRuntime.matrixAdd v8 v60 : ExactMatrix q0 n 1 d0)) v61 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v57 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 2 3 0 p61 v62 ∧
    MxxRuntime.blockModSwitchRuns p62 (1) ((MxxRuntime.matrixAdd v9 v62 : ExactMatrix q0 n 1 d0)) v63 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 0 1 0 p63 v64 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 0 1 0 p64 v65 ∧
    MxxRuntime.blockModSwitchRuns p65 (1) ((MxxRuntime.matrixAdd v4 ((MxxRuntime.matrixAdd v64 v65 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v66 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v57 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 0 1 0 p66 v67 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v50 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p67 v68 ∧
    MxxRuntime.blockModSwitchRuns p68 (1) ((MxxRuntime.matrixAdd v5 ((MxxRuntime.matrixAdd v67 v68 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v69 ∧
    outputs = (v47, (v59, (v61, (v63, (v66, (v69, ()))))))

/-- Computation bundle of role `lut.member.sampler` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutMemberSampler.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (q2 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (d13 : Nat) (d14 : Nat) (q3 : Nat) (q4 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit → Prop) (inputs : (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q1 n 1 d1) (v1 : ExactMatrix q2 n 1 d2) (v2 : ExactMatrix q1 n 1 d1) (v3 : ExactMatrix q2 n 1 d2) (v4 : ExactMatrix q0 n 1 d0) (v5 : ExactMatrix q0 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 d0) (v9 : ExactMatrix q0 n 1 d0) (v10 : ExactMatrix q1 n 1 d1) (v11 : ExactMatrix q2 n 1 d2) (v12 : ExactMatrix q1 n 1 d1) (v13 : ExactMatrix q2 n 1 d2) (v14 : ExactMatrix q0 n 1 d4) (v15 : ExactMatrix q0 n 1 d4) (v16 : ExactMatrix q0 n 1 d4) (v17 : ExactMatrix q0 n ((1 + 1) + 1) d4) (v18 : ExactMatrix q0 n (1 + 1) d4) (v19 : ExactMatrix q0 n 1 d6) (v20 : ExactMatrix q0 n 1 d6) (v21 : ExactMatrix q0 n 1 d6) (v22 : ExactMatrix q0 n ((1 + 1) + 1) d6) (v23 : ExactMatrix q0 n (1 + 1) d6) (v24 : ExactMatrix q0 n ((1 + 1) + 1) (d4 + d6)) (v25 : ExactMatrix q0 n 1 d7) (v26 : ExactMatrix q0 n 1 d7) (v27 : ExactMatrix q0 n 1 d7) (v28 : ExactMatrix q0 n ((1 + 1) + 1) d7) (v29 : ExactMatrix q0 n (1 + 1) d7) (v30 : ExactMatrix q0 n 1 d8) (v31 : ExactMatrix q0 n 1 d8) (v32 : ExactMatrix q0 n 1 d8) (v33 : ExactMatrix q0 n ((1 + 1) + 1) d8) (v34 : ExactMatrix q0 n (1 + 1) d8) (v35 : ExactMatrix q0 n ((1 + 1) + 1) (d7 + d8)) (v36 : ExactMatrix q1 n (1 + 1) d1) (v37 : ExactMatrix q1 n (d7 + d8) d1) (v38 : ExactMatrix q0 n (d7 + d8) d1) (v39 : ExactMatrix q0 n 1 d13) (v40 : ExactMatrix q3 n 1 d13) (v41 : ExactMatrix q0 n 1 d13) (v42 : ExactMatrix q3 n 1 d13) (v43 : ExactMatrix q3 n (1 + 1) d13) (v44 : ExactMatrix q3 n (d4 + d6) d13) (v45 : ExactMatrix q0 n (d4 + d6) d13) (v46 : ExactMatrix q0 n 1 d0) (v47 : ExactMatrix q2 n (1 + 1) d2) (v48 : ExactMatrix q2 n (d7 + d8) d2) (v49 : ExactMatrix q0 n (d7 + d8) d2) (v50 : ExactMatrix q0 n 1 d14) (v51 : ExactMatrix q4 n 1 d14) (v52 : ExactMatrix q0 n 1 d14) (v53 : ExactMatrix q4 n 1 d14) (v54 : ExactMatrix q4 n (1 + 1) d14) (v55 : ExactMatrix q4 n (d4 + d6) d14) (v56 : ExactMatrix q0 n (d4 + d6) d14) (v57 : ExactMatrix q0 n 1 d0) (v58 : ExactMatrix q0 n 1 d0) (v59 : ExactMatrix q0 n 1 d0) (v60 : ExactMatrix q0 n 1 d0) (v61 : ExactMatrix q0 n 1 d0) (v62 : ExactMatrix q0 n 1 d0) (v63 : ExactMatrix q0 n 1 d0),
    MxxRuntime.familyGetStatic inputs.1 (p0) v0 ∧
    MxxRuntime.familyGetStatic inputs.2.1 (p1) v1 ∧
    MxxRuntime.familyGetStatic inputs.2.2.1 (p2) v2 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.1 (p3) v3 ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, ()))))))))))))))))))))))) next) (Int.toNat p4) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (v0, (v1, (v2, (v3, ())))))))))) (v4, (v5, (v6, (v7, (v8, (v9, (v10, (v11, (v12, (v13, ())))))))))) ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.1 (p5) v14 ∧
    MxxRuntime.hashSample (hashModel) p7 ([.u64Le (p6), .integer ((MxxIR.selectInt (p8) p9)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.1) v15 ∧
    MxxRuntime.hashSample (hashModel) p11 ([.u64Le (p10), .integer ((MxxIR.selectInt (p12) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.2.1) v16 ∧
    MxxRuntime.concatRows v14 v15 v18 ∧
    MxxRuntime.concatRows v18 v16 v17 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.2.2.1 (p14) v19 ∧
    MxxRuntime.hashSample (hashModel) p16 ([.u64Le (p15), .integer ((MxxIR.selectInt (p17) p18)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.1) v20 ∧
    MxxRuntime.hashSample (hashModel) p20 ([.u64Le (p19), .integer ((MxxIR.selectInt (p21) p22)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.2.1) v21 ∧
    MxxRuntime.concatRows v19 v20 v23 ∧
    MxxRuntime.concatRows v23 v21 v22 ∧
    MxxRuntime.concatColumns v17 v22 v24 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p23) p24)) inputs.2.2.2.2.2.2.2.1 v25 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p25) p26)) inputs.2.2.2.2.2.2.2.2.1 v26 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p27) p28)) inputs.2.2.2.2.2.2.2.2.2.1 v27 ∧
    MxxRuntime.concatRows v25 v26 v29 ∧
    MxxRuntime.concatRows v29 v27 v28 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p29) p30)) inputs.2.2.2.2.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p31) p32)) inputs.2.2.2.2.2.2.2.2.2.2.2.1 v31 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p33) p34)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 v32 ∧
    MxxRuntime.concatRows v30 v31 v34 ∧
    MxxRuntime.concatRows v34 v32 v33 ∧
    MxxRuntime.concatColumns v28 v33 v35 ∧
    MxxRuntime.concatRows v10 v12 v36 ∧
    MxxRuntime.gadgetDecomposeRuns backend p35 p36 v36 v37 ∧
    MxxRuntime.compactCenteredRebaseRuns p37 v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 1 2 0 p38 v39 ∧
    MxxRuntime.blockModSwitchRuns p39 (1) v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 2 3 0 p40 v41 ∧
    MxxRuntime.blockModSwitchRuns p41 (1) v41 v42 ∧
    MxxRuntime.concatRows v40 v42 v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p42 p43 v43 v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p44 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 1 2 0 p45 v46 ∧
    MxxRuntime.concatRows v11 v13 v47 ∧
    MxxRuntime.gadgetDecomposeRuns backend p46 p47 v47 v48 ∧
    MxxRuntime.compactCenteredRebaseRuns p48 v48 v49 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p49 v50 ∧
    MxxRuntime.blockModSwitchRuns p50 (1) v50 v51 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p51 v52 ∧
    MxxRuntime.blockModSwitchRuns p52 (1) v52 v53 ∧
    MxxRuntime.concatRows v51 v53 v54 ∧
    MxxRuntime.gadgetDecomposeRuns backend p53 p54 v54 v55 ∧
    MxxRuntime.compactCenteredRebaseRuns p55 v55 v56 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 1 2 0 p56 v57 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 2 3 0 p57 v58 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 2 3 0 p58 v59 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 0 1 0 p59 v60 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 0 1 0 p60 v61 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 0 1 0 p61 v62 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p62 v63 ∧
    outputs = (((MxxRuntime.matrixAdd v6 v46 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v7 v57 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v8 v58 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v9 v59 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v4 ((MxxRuntime.matrixAdd v60 v61 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd v5 ((MxxRuntime.matrixAdd v62 v63 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)), ()))))))

/-- Computation bundle of role `lut.member.sampler-offset` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutMemberSamplerOffset.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (q2 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (d13 : Nat) (d14 : Nat) (q3 : Nat) (q4 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit → Prop) (inputs : (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) (outputs : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) :=
  ∃ (v0 : ExactMatrix q1 n 1 d1) (v1 : ExactMatrix q2 n 1 d2) (v2 : ExactMatrix q1 n 1 d1) (v3 : ExactMatrix q2 n 1 d2) (v4 : ExactMatrix q0 n 1 d0) (v5 : ExactMatrix q0 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 d0) (v9 : ExactMatrix q0 n 1 d0) (v10 : ExactMatrix q1 n 1 d1) (v11 : ExactMatrix q2 n 1 d2) (v12 : ExactMatrix q1 n 1 d1) (v13 : ExactMatrix q2 n 1 d2) (v14 : ExactMatrix q0 n 1 d4) (v15 : ExactMatrix q0 n 1 d4) (v16 : ExactMatrix q0 n 1 d4) (v17 : ExactMatrix q0 n ((1 + 1) + 1) d4) (v18 : ExactMatrix q0 n (1 + 1) d4) (v19 : ExactMatrix q0 n 1 d6) (v20 : ExactMatrix q0 n 1 d6) (v21 : ExactMatrix q0 n 1 d6) (v22 : ExactMatrix q0 n ((1 + 1) + 1) d6) (v23 : ExactMatrix q0 n (1 + 1) d6) (v24 : ExactMatrix q0 n ((1 + 1) + 1) (d4 + d6)) (v25 : ExactMatrix q0 n 1 d7) (v26 : ExactMatrix q0 n 1 d7) (v27 : ExactMatrix q0 n 1 d7) (v28 : ExactMatrix q0 n ((1 + 1) + 1) d7) (v29 : ExactMatrix q0 n (1 + 1) d7) (v30 : ExactMatrix q0 n 1 d8) (v31 : ExactMatrix q0 n 1 d8) (v32 : ExactMatrix q0 n 1 d8) (v33 : ExactMatrix q0 n ((1 + 1) + 1) d8) (v34 : ExactMatrix q0 n (1 + 1) d8) (v35 : ExactMatrix q0 n ((1 + 1) + 1) (d7 + d8)) (v36 : ExactMatrix q1 n (1 + 1) d1) (v37 : ExactMatrix q1 n (d7 + d8) d1) (v38 : ExactMatrix q0 n (d7 + d8) d1) (v39 : ExactMatrix q0 n 1 d13) (v40 : ExactMatrix q3 n 1 d13) (v41 : ExactMatrix q0 n 1 d13) (v42 : ExactMatrix q3 n 1 d13) (v43 : ExactMatrix q3 n (1 + 1) d13) (v44 : ExactMatrix q3 n (d4 + d6) d13) (v45 : ExactMatrix q0 n (d4 + d6) d13) (v46 : ExactMatrix q0 n 1 d0) (v47 : ExactMatrix q2 n (1 + 1) d2) (v48 : ExactMatrix q2 n (d7 + d8) d2) (v49 : ExactMatrix q0 n (d7 + d8) d2) (v50 : ExactMatrix q0 n 1 d14) (v51 : ExactMatrix q4 n 1 d14) (v52 : ExactMatrix q0 n 1 d14) (v53 : ExactMatrix q4 n 1 d14) (v54 : ExactMatrix q4 n (1 + 1) d14) (v55 : ExactMatrix q4 n (d4 + d6) d14) (v56 : ExactMatrix q0 n (d4 + d6) d14) (v57 : ExactMatrix q0 n 1 d0) (v58 : ExactMatrix q0 n 1 d0) (v59 : ExactMatrix q0 n 1 d0) (v60 : ExactMatrix q0 n 1 d0) (v61 : ExactMatrix q0 n 1 d0) (v62 : ExactMatrix q0 n 1 d0) (v63 : ExactMatrix q0 n 1 d0),
    MxxRuntime.familyGetStatic inputs.1 (p0) v0 ∧
    MxxRuntime.familyGetStatic inputs.2.1 (p1) v1 ∧
    MxxRuntime.familyGetStatic inputs.2.2.1 (p2) v2 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.1 (p3) v3 ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, ()))))))))))))))))))))))) next) (Int.toNat p4) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (v0, (v1, (v2, (v3, ())))))))))) (v4, (v5, (v6, (v7, (v8, (v9, (v10, (v11, (v12, (v13, ())))))))))) ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.1 (p5) v14 ∧
    MxxRuntime.hashSample (hashModel) p7 ([.u64Le (p6), .integer ((MxxIR.selectInt (p8) p9)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.1) v15 ∧
    MxxRuntime.hashSample (hashModel) p11 ([.u64Le (p10), .integer ((MxxIR.selectInt (p12) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.2.1) v16 ∧
    MxxRuntime.concatRows v14 v15 v18 ∧
    MxxRuntime.concatRows v18 v16 v17 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.2.2.1 (p14) v19 ∧
    MxxRuntime.hashSample (hashModel) p16 ([.u64Le (p15), .integer ((MxxIR.selectInt (p17) p18)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.1) v20 ∧
    MxxRuntime.hashSample (hashModel) p20 ([.u64Le (p19), .integer ((MxxIR.selectInt (p21) p22)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.2.1) v21 ∧
    MxxRuntime.concatRows v19 v20 v23 ∧
    MxxRuntime.concatRows v23 v21 v22 ∧
    MxxRuntime.concatColumns v17 v22 v24 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p23) p24)) inputs.2.2.2.2.2.2.2.1 v25 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p25) p26)) inputs.2.2.2.2.2.2.2.2.1 v26 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p27) p28)) inputs.2.2.2.2.2.2.2.2.2.1 v27 ∧
    MxxRuntime.concatRows v25 v26 v29 ∧
    MxxRuntime.concatRows v29 v27 v28 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p29) p30)) inputs.2.2.2.2.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p31) p32)) inputs.2.2.2.2.2.2.2.2.2.2.2.1 v31 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p33) p34)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 v32 ∧
    MxxRuntime.concatRows v30 v31 v34 ∧
    MxxRuntime.concatRows v34 v32 v33 ∧
    MxxRuntime.concatColumns v28 v33 v35 ∧
    MxxRuntime.concatRows v10 v12 v36 ∧
    MxxRuntime.gadgetDecomposeRuns backend p35 p36 v36 v37 ∧
    MxxRuntime.compactCenteredRebaseRuns p37 v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 1 2 0 p38 v39 ∧
    MxxRuntime.blockModSwitchRuns p39 (1) v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 2 3 0 p40 v41 ∧
    MxxRuntime.blockModSwitchRuns p41 (1) v41 v42 ∧
    MxxRuntime.concatRows v40 v42 v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p42 p43 v43 v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p44 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 1 2 0 p45 v46 ∧
    MxxRuntime.concatRows v11 v13 v47 ∧
    MxxRuntime.gadgetDecomposeRuns backend p46 p47 v47 v48 ∧
    MxxRuntime.compactCenteredRebaseRuns p48 v48 v49 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p49 v50 ∧
    MxxRuntime.blockModSwitchRuns p50 (1) v50 v51 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p51 v52 ∧
    MxxRuntime.blockModSwitchRuns p52 (1) v52 v53 ∧
    MxxRuntime.concatRows v51 v53 v54 ∧
    MxxRuntime.gadgetDecomposeRuns backend p53 p54 v54 v55 ∧
    MxxRuntime.compactCenteredRebaseRuns p55 v55 v56 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 1 2 0 p56 v57 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 2 3 0 p57 v58 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 2 3 0 p58 v59 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 0 1 0 p59 v60 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 0 1 0 p60 v61 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v56 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 0 1 0 p61 v62 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v49 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p62 v63 ∧
    outputs = (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v6 v46 : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v7 v57 : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v8 v58 : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v9 v59 : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v4 ((MxxRuntime.matrixAdd v60 v61 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixAdd v5 ((MxxRuntime.matrixAdd v62 v63 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 : ExactMatrix q0 n 1 d0)), ()))))))

/-- Computation bundle of role `lut.round.step` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def LutRoundStep.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (d1 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (q1 : Nat) (q2 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (d13 : Nat) (d14 : Nat) (d15 : Nat) (d16 : Nat) (q3 : Nat) (n1 : Nat) (d17 : Nat) (d18 : Nat) (q4 : Nat) (n2 : Nat) (d19 : Nat) (d20 : Nat) (q5 : Nat) (n3 : Nat) (d21 : Nat) (d22 : Nat) (q6 : Nat) (n4 : Nat) (d23 : Nat) (q7 : Nat) (q8 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ExactMatrix q0 n 1 d4 × ExactMatrix q0 n 1 d5 × ExactMatrix q0 n 1 d6 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d9 × ExactMatrix q1 n 1 d0 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d0 × ExactMatrix q2 n 1 d2 × (Fin d10 → ExactMatrix q0 n 1 d11) × ByteArray × (Fin d12 → ExactMatrix q0 n 1 d13) × ExactMatrix q0 n 1 d14 × ExactMatrix q0 n 1 d14 × ExactMatrix q0 n 1 d14 × ExactMatrix q0 n 1 d15 × ExactMatrix q0 n 1 d15 × ExactMatrix q0 n 1 d15 × (Fin d16 → ExactMatrix q3 n1 1 d17) × (Fin d18 → ExactMatrix q4 n2 1 d19) × (Fin d20 → ExactMatrix q5 n3 1 d21) × (Fin d22 → ExactMatrix q6 n4 1 d23) × Unit) (outputs : ExactMatrix q0 n 1 d4 × ExactMatrix q0 n 1 d5 × ExactMatrix q0 n 1 d6 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d9 × ExactMatrix q3 n1 1 d17 × ExactMatrix q4 n2 1 d19 × ExactMatrix q5 n3 1 d21 × ExactMatrix q6 n4 1 d23 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d11) (v1 : ExactMatrix q0 n 1 d11) (v2 : ExactMatrix q0 n 1 d11) (v3 : ExactMatrix q0 n ((1 + 1) + 1) d11) (v4 : ExactMatrix q0 n (1 + 1) d11) (v5 : ExactMatrix q0 n 1 d13) (v6 : ExactMatrix q0 n 1 d13) (v7 : ExactMatrix q0 n 1 d13) (v8 : ExactMatrix q0 n ((1 + 1) + 1) d13) (v9 : ExactMatrix q0 n (1 + 1) d13) (v10 : ExactMatrix q0 n ((1 + 1) + 1) (d11 + d13)) (v11 : ExactMatrix q0 n 1 d14) (v12 : ExactMatrix q0 n 1 d14) (v13 : ExactMatrix q0 n 1 d14) (v14 : ExactMatrix q0 n ((1 + 1) + 1) d14) (v15 : ExactMatrix q0 n (1 + 1) d14) (v16 : ExactMatrix q0 n 1 d15) (v17 : ExactMatrix q0 n 1 d15) (v18 : ExactMatrix q0 n 1 d15) (v19 : ExactMatrix q0 n ((1 + 1) + 1) d15) (v20 : ExactMatrix q0 n (1 + 1) d15) (v21 : ExactMatrix q0 n ((1 + 1) + 1) (d14 + d15)) (v22 : ExactMatrix q1 n (1 + 1) d0) (v23 : ExactMatrix q1 n (d14 + d15) d0) (v24 : ExactMatrix q0 n (d14 + d15) d0) (v25 : ExactMatrix q0 n 1 d1) (v26 : ExactMatrix q7 n 1 d1) (v27 : ExactMatrix q0 n 1 d1) (v28 : ExactMatrix q7 n 1 d1) (v29 : ExactMatrix q7 n (1 + 1) d1) (v30 : ExactMatrix q7 n (d11 + d13) d1) (v31 : ExactMatrix q0 n (d11 + d13) d1) (v32 : ExactMatrix q0 n 1 d4) (v33 : ExactMatrix q0 n 1 d4) (v34 : ExactMatrix q2 n (1 + 1) d2) (v35 : ExactMatrix q2 n (d14 + d15) d2) (v36 : ExactMatrix q0 n (d14 + d15) d2) (v37 : ExactMatrix q0 n 1 d3) (v38 : ExactMatrix q8 n 1 d3) (v39 : ExactMatrix q0 n 1 d3) (v40 : ExactMatrix q8 n 1 d3) (v41 : ExactMatrix q8 n (1 + 1) d3) (v42 : ExactMatrix q8 n (d11 + d13) d3) (v43 : ExactMatrix q0 n (d11 + d13) d3) (v44 : ExactMatrix q0 n 1 d5) (v45 : ExactMatrix q0 n 1 d5) (v46 : ExactMatrix q0 n 1 d6) (v47 : ExactMatrix q0 n 1 d7) (v48 : ExactMatrix q0 n 1 d8) (v49 : ExactMatrix q0 n 1 d9) (v50 : ExactMatrix q3 n1 1 d17) (v51 : ExactMatrix q4 n2 1 d19) (v52 : ExactMatrix q5 n3 1 d21) (v53 : ExactMatrix q6 n4 1 d23),
    p0 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p1 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v0 ∧
    MxxRuntime.hashSample (hashModel) p3 ([.u64Le (p2), .integer ((MxxIR.selectInt ((Int.ofNat i)) p4)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.1) v1 ∧
    MxxRuntime.hashSample (hashModel) p6 ([.u64Le (p5), .integer ((MxxIR.selectInt ((Int.ofNat i)) p7)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.1) v2 ∧
    MxxRuntime.concatRows v0 v1 v4 ∧
    MxxRuntime.concatRows v4 v2 v3 ∧
    p8 ≤ ((Int.ofNat i)) ∧
    ((Int.ofNat i)) < p9 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.ofNat i)) v5 ∧
    MxxRuntime.hashSample (hashModel) p11 ([.u64Le (p10), .integer ((MxxIR.selectInt ((Int.ofNat i)) p12)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.1) v6 ∧
    MxxRuntime.hashSample (hashModel) p14 ([.u64Le (p13), .integer ((MxxIR.selectInt ((Int.ofNat i)) p15)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.1) v7 ∧
    MxxRuntime.concatRows v5 v6 v9 ∧
    MxxRuntime.concatRows v9 v7 v8 ∧
    MxxRuntime.concatColumns v3 v8 v10 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p16)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v11 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p17)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v12 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p18)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v13 ∧
    MxxRuntime.concatRows v11 v12 v15 ∧
    MxxRuntime.concatRows v15 v13 v14 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p19)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v16 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p20)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v17 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt ((Int.ofNat i)) p21)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v18 ∧
    MxxRuntime.concatRows v16 v17 v20 ∧
    MxxRuntime.concatRows v20 v18 v19 ∧
    MxxRuntime.concatColumns v14 v19 v21 ∧
    MxxRuntime.concatRows inputs.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 v22 ∧
    MxxRuntime.gadgetDecomposeRuns backend p22 p23 v22 v23 ∧
    MxxRuntime.compactCenteredRebaseRuns p24 v23 v24 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v24 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 1 2 0 p25 v25 ∧
    MxxRuntime.blockModSwitchRuns p26 (1) v25 v26 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v24 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 2 3 0 p27 v27 ∧
    MxxRuntime.blockModSwitchRuns p28 (1) v27 v28 ∧
    MxxRuntime.concatRows v26 v28 v29 ∧
    MxxRuntime.gadgetDecomposeRuns backend p29 p30 v29 v30 ∧
    MxxRuntime.compactCenteredRebaseRuns p31 v30 v31 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v31 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 0 1 0 p32 v32 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v24 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 0 1 0 p33 v33 ∧
    MxxRuntime.concatRows inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.2.1 v34 ∧
    MxxRuntime.gadgetDecomposeRuns backend p34 p35 v34 v35 ∧
    MxxRuntime.compactCenteredRebaseRuns p36 v35 v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v36 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p37 v37 ∧
    MxxRuntime.blockModSwitchRuns p38 (1) v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v36 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p39 v39 ∧
    MxxRuntime.blockModSwitchRuns p40 (1) v39 v40 ∧
    MxxRuntime.concatRows v38 v40 v41 ∧
    MxxRuntime.gadgetDecomposeRuns backend p41 p42 v41 v42 ∧
    MxxRuntime.compactCenteredRebaseRuns p43 v42 v43 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v43 : ExactMatrix q0 n ((1 + 1) + 1) d3)) 0 1 0 p44 v44 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v21 v36 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p45 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v31 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 1 2 0 p46 v46 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v43 : ExactMatrix q0 n ((1 + 1) + 1) d3)) 1 2 0 p47 v47 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v31 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 2 3 0 p48 v48 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v10 v43 : ExactMatrix q0 n ((1 + 1) + 1) d3)) 2 3 0 p49 v49 ∧
    p50 ≤ ((((Int.ofNat i)) + (1))) ∧
    ((((Int.ofNat i)) + (1))) < p51 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ((((Int.ofNat i)) + (1))) v50 ∧
    p52 ≤ ((((Int.ofNat i)) + (1))) ∧
    ((((Int.ofNat i)) + (1))) < p53 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ((((Int.ofNat i)) + (1))) v51 ∧
    p54 ≤ ((((Int.ofNat i)) + (1))) ∧
    ((((Int.ofNat i)) + (1))) < p55 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ((((Int.ofNat i)) + (1))) v52 ∧
    p56 ≤ ((((Int.ofNat i)) + (1))) ∧
    ((((Int.ofNat i)) + (1))) < p57 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ((((Int.ofNat i)) + (1))) v53 ∧
    outputs = (((MxxRuntime.matrixAdd inputs.1 ((MxxRuntime.matrixAdd v32 v33 : ExactMatrix q0 n 1 d4)) : ExactMatrix q0 n 1 d4)), (((MxxRuntime.matrixAdd inputs.2.1 ((MxxRuntime.matrixAdd v44 v45 : ExactMatrix q0 n 1 d5)) : ExactMatrix q0 n 1 d5)), (((MxxRuntime.matrixAdd inputs.2.2.1 v46 : ExactMatrix q0 n 1 d6)), (((MxxRuntime.matrixAdd inputs.2.2.2.1 v47 : ExactMatrix q0 n 1 d7)), (((MxxRuntime.matrixAdd inputs.2.2.2.2.1 v48 : ExactMatrix q0 n 1 d8)), (((MxxRuntime.matrixAdd inputs.2.2.2.2.2.1 v49 : ExactMatrix q0 n 1 d9)), (v50, (v51, (v52, (v53, ()))))))))))

/-- Computation bundle of role `prf.cell0` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def PrfCell0.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (d1 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (q1 : Nat) (d8 : Nat) (q2 : Nat) (q3 : Nat) (q4 : Nat) (q5 : Nat) (d9 : Nat) (q6 : Nat) (d10 : Nat) (q7 : Nat) (d11 : Nat) (q8 : Nat) (d12 : Nat) (q9 : Nat) (d13 : Nat) (q10 : Nat) (d14 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : Int × ExactMatrix q0 n 1 d6 × ExactMatrix q0 n 1 d6 × ExactMatrix q0 n 1 d6 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q1 n 1 d0 × ExactMatrix q1 n 1 d0 × ExactMatrix q0 n 1 d1 × (Fin d8 → ExactMatrix q0 n 1 1) × ExactMatrix q2 n 1 d2 × ExactMatrix q2 n 1 d2 × ExactMatrix q0 n 1 d3 × ByteArray × ExactMatrix q3 n 1 d4 × ExactMatrix q4 n 1 d5 × Unit) (outputs : ExactMatrix q5 n 1 d9 × ExactMatrix q6 n 1 d10 × ExactMatrix q7 n 1 d11 × ExactMatrix q8 n 1 d12 × ExactMatrix q9 n 1 d13 × ExactMatrix q10 n 1 d14 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n ((1 + 1) + 1) d6) (v1 : ExactMatrix q0 n (1 + 1) d6) (v2 : ExactMatrix q0 n ((1 + 1) + 1) d7) (v3 : ExactMatrix q0 n (1 + 1) d7) (v4 : ExactMatrix q0 n ((1 + 1) + 1) (d6 + d7)) (v5 : ExactMatrix q1 n (1 + 1) d0) (v6 : ExactMatrix q1 n (d6 + d7) d0) (v7 : ExactMatrix q0 n (d6 + d7) d0) (v8 : ExactMatrix q0 n 1 d1) (v9 : ExactMatrix q0 n 1 1) (v10 : ExactMatrix q0 n 1 d1) (v11 : ExactMatrix q0 n 1 d1) (v12 : ExactMatrix q0 n ((1 + 1) + 1) d1) (v13 : ExactMatrix q0 n (1 + 1) d1) (v14 : ExactMatrix q2 n (1 + 1) d2) (v15 : ExactMatrix q2 n (d6 + d7) d2) (v16 : ExactMatrix q0 n (d6 + d7) d2) (v17 : ExactMatrix q0 n 1 d3) (v18 : ExactMatrix q0 n 1 d3) (v19 : ExactMatrix q0 n 1 d3) (v20 : ExactMatrix q0 n ((1 + 1) + 1) d3) (v21 : ExactMatrix q0 n (1 + 1) d3) (v22 : ExactMatrix q0 n ((1 + 1) + 1) (d1 + d3)) (v23 : ExactMatrix q3 n 1 d4) (v24 : ExactMatrix q3 n (1 + 1) d4) (v25 : ExactMatrix q3 n (d1 + d3) d4) (v26 : ExactMatrix q0 n (d1 + d3) d4) (v27 : ExactMatrix q0 n 1 d9) (v28 : ExactMatrix q5 n 1 d9) (v29 : ExactMatrix q4 n 1 d5) (v30 : ExactMatrix q4 n (1 + 1) d5) (v31 : ExactMatrix q4 n (d1 + d3) d5) (v32 : ExactMatrix q0 n (d1 + d3) d5) (v33 : ExactMatrix q0 n 1 d10) (v34 : ExactMatrix q6 n 1 d10) (v35 : ExactMatrix q0 n 1 d11) (v36 : ExactMatrix q7 n 1 d11) (v37 : ExactMatrix q0 n 1 d12) (v38 : ExactMatrix q8 n 1 d12) (v39 : ExactMatrix q0 n 1 d13) (v40 : ExactMatrix q9 n 1 d13) (v41 : ExactMatrix q0 n 1 d14) (v42 : ExactMatrix q10 n 1 d14),
    MxxRuntime.concatRows inputs.2.1 inputs.2.2.1 v1 ∧
    MxxRuntime.concatRows v1 inputs.2.2.2.1 v0 ∧
    MxxRuntime.concatRows inputs.2.2.2.2.1 inputs.2.2.2.2.2.1 v3 ∧
    MxxRuntime.concatRows v3 inputs.2.2.2.2.2.2.1 v2 ∧
    MxxRuntime.concatColumns v0 v2 v4 ∧
    MxxRuntime.concatRows inputs.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.1 v5 ∧
    MxxRuntime.gadgetDecomposeRuns backend p0 p1 v5 v6 ∧
    MxxRuntime.compactCenteredRebaseRuns p2 v6 v7 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v7 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 0 1 0 p3 v8 ∧
    p4 ≤ inputs.1 ∧
    inputs.1 < p5 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.1 inputs.1 v9 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v7 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 1 2 0 p6 v10 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v7 : ExactMatrix q0 n ((1 + 1) + 1) d0)) 2 3 0 p7 v11 ∧
    MxxRuntime.concatRows ((MxxRuntime.matrixAdd v8 ((MxxRuntime.matrixMulScalarRight inputs.2.2.2.2.2.2.2.2.2.1 v9 : ExactMatrix q0 n 1 d1)) : ExactMatrix q0 n 1 d1)) v10 v13 ∧
    MxxRuntime.concatRows v13 v11 v12 ∧
    MxxRuntime.concatRows inputs.2.2.2.2.2.2.2.2.2.2.2.1 inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 v14 ∧
    MxxRuntime.gadgetDecomposeRuns backend p8 p9 v14 v15 ∧
    MxxRuntime.compactCenteredRebaseRuns p10 v15 v16 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v16 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p11 v17 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v16 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p12 v18 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v4 v16 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p13 v19 ∧
    MxxRuntime.concatRows ((MxxRuntime.matrixAdd v17 ((MxxRuntime.matrixMulScalarRight inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v9 : ExactMatrix q0 n 1 d3)) : ExactMatrix q0 n 1 d3)) v18 v21 ∧
    MxxRuntime.concatRows v21 v19 v20 ∧
    MxxRuntime.concatColumns v12 v20 v22 ∧
    MxxRuntime.hashSample (hashModel) p16 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p14), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p15), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) v23 ∧
    MxxRuntime.concatRows v23 inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v24 ∧
    MxxRuntime.gadgetDecomposeRuns backend p17 p18 v24 v25 ∧
    MxxRuntime.compactCenteredRebaseRuns p19 v25 v26 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v26 : ExactMatrix q0 n ((1 + 1) + 1) d4)) 1 2 0 p20 v27 ∧
    MxxRuntime.blockModSwitchRuns p21 (1) v27 v28 ∧
    MxxRuntime.hashSample (hashModel) p24 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p22), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p23), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) v29 ∧
    MxxRuntime.concatRows v29 inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.gadgetDecomposeRuns backend p25 p26 v30 v31 ∧
    MxxRuntime.compactCenteredRebaseRuns p27 v31 v32 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v32 : ExactMatrix q0 n ((1 + 1) + 1) d5)) 1 2 0 p28 v33 ∧
    MxxRuntime.blockModSwitchRuns p29 (1) v33 v34 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v26 : ExactMatrix q0 n ((1 + 1) + 1) d4)) 2 3 0 p30 v35 ∧
    MxxRuntime.blockModSwitchRuns p31 (1) v35 v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v32 : ExactMatrix q0 n ((1 + 1) + 1) d5)) 2 3 0 p32 v37 ∧
    MxxRuntime.blockModSwitchRuns p33 (1) v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v26 : ExactMatrix q0 n ((1 + 1) + 1) d4)) 0 1 0 p34 v39 ∧
    MxxRuntime.blockModSwitchRuns p35 (1) v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v22 v32 : ExactMatrix q0 n ((1 + 1) + 1) d5)) 0 1 0 p36 v41 ∧
    MxxRuntime.blockModSwitchRuns p37 (1) v41 v42 ∧
    outputs = (v28, (v34, (v36, (v38, (v40, (v42, ()))))))

/-- Computation bundle of role `prf.cell1` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def PrfCell1.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (q2 : Nat) (d1 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (q3 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (q4 : Nat) (d13 : Nat) (q5 : Nat) (d14 : Nat) (d15 : Nat) (q6 : Nat) (q7 : Nat) (q8 : Nat) (d16 : Nat) (q9 : Nat) (d17 : Nat) (q10 : Nat) (d18 : Nat) (q11 : Nat) (d19 : Nat) (q12 : Nat) (d20 : Nat) (q13 : Nat) (d21 : Nat) (q14 : Nat) (q15 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (p63 : _) (p64 : _) (p65 : _) (p66 : _) (p67 : _) (p68 : _) (p69 : _) (p70 : _) (p71 : _) (p72 : _) (p73 : _) (p74 : _) (p75 : _) (p76 : _) (p77 : _) (p78 : _) (p79 : _) (p80 : _) (p81 : _) (p82 : _) (p83 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : Int × ExactMatrix q0 n 1 d7 × ExactMatrix q1 n 1 d8 × ExactMatrix q1 n 1 d8 × ExactMatrix q0 n 1 d7 × ExactMatrix q1 n 1 d8 × ExactMatrix q1 n 1 d8 × (Fin d9 → ExactMatrix q3 n 1 d1) × (Fin d10 → ExactMatrix q3 n 1 d1) × (Fin d11 → ExactMatrix q2 n 1 d2) × (Fin d12 → ExactMatrix q4 n 1 1) × (Fin d13 → ExactMatrix q5 n 1 d3) × (Fin d14 → ExactMatrix q5 n 1 d3) × (Fin d15 → ExactMatrix q2 n 1 d4) × ByteArray × ExactMatrix q6 n 1 d5 × ExactMatrix q7 n 1 d6 × Unit) (outputs : ExactMatrix q8 n 1 d16 × ExactMatrix q9 n 1 d17 × ExactMatrix q10 n 1 d18 × ExactMatrix q11 n 1 d19 × ExactMatrix q12 n 1 d20 × ExactMatrix q13 n 1 d21 × Unit) :=
  ∃ (v0 : ExactMatrix q14 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n d7 d0) (v3 : ExactMatrix q2 n 1 d0) (v4 : ExactMatrix q15 n 1 d0) (v5 : ExactMatrix q1 n 1 d0) (v6 : ExactMatrix q1 n d8 d0) (v7 : ExactMatrix q2 n 1 d0) (v8 : ExactMatrix q2 n 1 d0) (v9 : ExactMatrix q2 n ((1 + 1) + 1) d0) (v10 : ExactMatrix q2 n (1 + 1) d0) (v11 : ExactMatrix q2 n 1 d0) (v12 : ExactMatrix q2 n 1 d0) (v13 : ExactMatrix q2 n 1 d0) (v14 : ExactMatrix q2 n ((1 + 1) + 1) d0) (v15 : ExactMatrix q2 n (1 + 1) d0) (v16 : ExactMatrix q2 n ((1 + 1) + 1) (d0 + d0)) (v17 : ExactMatrix q3 n 1 d1) (v18 : ExactMatrix q3 n 1 d1) (v19 : ExactMatrix q3 n (1 + 1) d1) (v20 : ExactMatrix q3 n (d0 + d0) d1) (v21 : ExactMatrix q2 n (d0 + d0) d1) (v22 : ExactMatrix q2 n 1 d2) (v23 : ExactMatrix q2 n 1 d2) (v24 : ExactMatrix q4 n 1 1) (v25 : ExactMatrix q2 n 1 1) (v26 : ExactMatrix q2 n 1 d2) (v27 : ExactMatrix q2 n 1 d2) (v28 : ExactMatrix q2 n ((1 + 1) + 1) d2) (v29 : ExactMatrix q2 n (1 + 1) d2) (v30 : ExactMatrix q5 n 1 d3) (v31 : ExactMatrix q5 n 1 d3) (v32 : ExactMatrix q5 n (1 + 1) d3) (v33 : ExactMatrix q5 n (d0 + d0) d3) (v34 : ExactMatrix q2 n (d0 + d0) d3) (v35 : ExactMatrix q2 n 1 d4) (v36 : ExactMatrix q2 n 1 d4) (v37 : ExactMatrix q2 n 1 d4) (v38 : ExactMatrix q2 n 1 d4) (v39 : ExactMatrix q2 n ((1 + 1) + 1) d4) (v40 : ExactMatrix q2 n (1 + 1) d4) (v41 : ExactMatrix q2 n ((1 + 1) + 1) (d2 + d4)) (v42 : ExactMatrix q6 n 1 d5) (v43 : ExactMatrix q6 n (1 + 1) d5) (v44 : ExactMatrix q6 n (d2 + d4) d5) (v45 : ExactMatrix q2 n (d2 + d4) d5) (v46 : ExactMatrix q2 n 1 d16) (v47 : ExactMatrix q8 n 1 d16) (v48 : ExactMatrix q7 n 1 d6) (v49 : ExactMatrix q7 n (1 + 1) d6) (v50 : ExactMatrix q7 n (d2 + d4) d6) (v51 : ExactMatrix q2 n (d2 + d4) d6) (v52 : ExactMatrix q2 n 1 d17) (v53 : ExactMatrix q9 n 1 d17) (v54 : ExactMatrix q2 n 1 d18) (v55 : ExactMatrix q10 n 1 d18) (v56 : ExactMatrix q2 n 1 d19) (v57 : ExactMatrix q11 n 1 d19) (v58 : ExactMatrix q2 n 1 d20) (v59 : ExactMatrix q12 n 1 d20) (v60 : ExactMatrix q2 n 1 d21) (v61 : ExactMatrix q13 n 1 d21),
    MxxRuntime.gadgetMatrixRuns backend p0 p1 v0 ∧
    MxxRuntime.centeredRebaseRuns v0 v1 ∧
    MxxRuntime.gadgetDecomposeRuns backend p2 p3 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p4 : ExactMatrix q0 n 1 1)) v1 : ExactMatrix q0 n 1 d0)) v2 ∧
    MxxRuntime.blockModSwitchRuns p5 (1) ((MxxRuntime.matrixMul inputs.2.1 v2 : ExactMatrix q0 n 1 d0)) v3 ∧
    MxxRuntime.gadgetMatrixRuns backend p6 p7 v4 ∧
    MxxRuntime.centeredRebaseRuns v4 v5 ∧
    MxxRuntime.gadgetDecomposeRuns backend p8 p9 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p10 : ExactMatrix q1 n 1 1)) v5 : ExactMatrix q1 n 1 d0)) v6 ∧
    MxxRuntime.blockModSwitchRuns p11 (1) ((MxxRuntime.matrixMul inputs.2.2.1 v6 : ExactMatrix q1 n 1 d0)) v7 ∧
    MxxRuntime.blockModSwitchRuns p12 (1) ((MxxRuntime.matrixMul inputs.2.2.2.1 v6 : ExactMatrix q1 n 1 d0)) v8 ∧
    MxxRuntime.concatRows v3 v7 v10 ∧
    MxxRuntime.concatRows v10 v8 v9 ∧
    MxxRuntime.blockModSwitchRuns p13 (1) ((MxxRuntime.matrixMul inputs.2.2.2.2.1 v2 : ExactMatrix q0 n 1 d0)) v11 ∧
    MxxRuntime.blockModSwitchRuns p14 (1) ((MxxRuntime.matrixMul inputs.2.2.2.2.2.1 v6 : ExactMatrix q1 n 1 d0)) v12 ∧
    MxxRuntime.blockModSwitchRuns p15 (1) ((MxxRuntime.matrixMul inputs.2.2.2.2.2.2.1 v6 : ExactMatrix q1 n 1 d0)) v13 ∧
    MxxRuntime.concatRows v11 v12 v15 ∧
    MxxRuntime.concatRows v15 v13 v14 ∧
    MxxRuntime.concatColumns v9 v14 v16 ∧
    p16 ≤ ((Int.fdiv ((Int.ofNat i)) (p17))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p19))) < p18 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p20))) v17 ∧
    p21 ≤ ((Int.fdiv ((Int.ofNat i)) (p22))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p24))) < p23 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p25))) v18 ∧
    MxxRuntime.concatRows v17 v18 v19 ∧
    MxxRuntime.gadgetDecomposeRuns backend p26 p27 v19 v20 ∧
    MxxRuntime.compactCenteredRebaseRuns p28 v20 v21 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v21 : ExactMatrix q2 n ((1 + 1) + 1) d1)) 0 1 0 p29 v22 ∧
    p30 ≤ ((Int.fdiv ((Int.ofNat i)) (p31))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p33))) < p32 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p34))) v23 ∧
    p35 ≤ inputs.1 ∧
    inputs.1 < p36 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.1 inputs.1 v24 ∧
    MxxRuntime.modulusReduceRuns v24 v25 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v21 : ExactMatrix q2 n ((1 + 1) + 1) d1)) 1 2 0 p37 v26 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v21 : ExactMatrix q2 n ((1 + 1) + 1) d1)) 2 3 0 p38 v27 ∧
    MxxRuntime.concatRows ((MxxRuntime.matrixAdd v22 ((MxxRuntime.matrixMulScalarRight v23 v25 : ExactMatrix q2 n 1 d2)) : ExactMatrix q2 n 1 d2)) v26 v29 ∧
    MxxRuntime.concatRows v29 v27 v28 ∧
    p39 ≤ ((Int.fdiv ((Int.ofNat i)) (p40))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p42))) < p41 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p43))) v30 ∧
    p44 ≤ ((Int.fdiv ((Int.ofNat i)) (p45))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p47))) < p46 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p48))) v31 ∧
    MxxRuntime.concatRows v30 v31 v32 ∧
    MxxRuntime.gadgetDecomposeRuns backend p49 p50 v32 v33 ∧
    MxxRuntime.compactCenteredRebaseRuns p51 v33 v34 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v34 : ExactMatrix q2 n ((1 + 1) + 1) d3)) 0 1 0 p52 v35 ∧
    p53 ≤ ((Int.fdiv ((Int.ofNat i)) (p54))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p56))) < p55 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p57))) v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v34 : ExactMatrix q2 n ((1 + 1) + 1) d3)) 1 2 0 p58 v37 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v16 v34 : ExactMatrix q2 n ((1 + 1) + 1) d3)) 2 3 0 p59 v38 ∧
    MxxRuntime.concatRows ((MxxRuntime.matrixAdd v35 ((MxxRuntime.matrixMulScalarRight v36 v25 : ExactMatrix q2 n 1 d4)) : ExactMatrix q2 n 1 d4)) v37 v40 ∧
    MxxRuntime.concatRows v40 v38 v39 ∧
    MxxRuntime.concatColumns v28 v39 v41 ∧
    MxxRuntime.hashSample (hashModel) p62 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p60), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p61), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) v42 ∧
    MxxRuntime.concatRows v42 inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p63 p64 v43 v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p65 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v45 : ExactMatrix q2 n ((1 + 1) + 1) d5)) 1 2 0 p66 v46 ∧
    MxxRuntime.blockModSwitchRuns p67 (1) v46 v47 ∧
    MxxRuntime.hashSample (hashModel) p70 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p68), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p69), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1) v48 ∧
    MxxRuntime.concatRows v48 inputs.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.2.1 v49 ∧
    MxxRuntime.gadgetDecomposeRuns backend p71 p72 v49 v50 ∧
    MxxRuntime.compactCenteredRebaseRuns p73 v50 v51 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v51 : ExactMatrix q2 n ((1 + 1) + 1) d6)) 1 2 0 p74 v52 ∧
    MxxRuntime.blockModSwitchRuns p75 (1) v52 v53 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v45 : ExactMatrix q2 n ((1 + 1) + 1) d5)) 2 3 0 p76 v54 ∧
    MxxRuntime.blockModSwitchRuns p77 (1) v54 v55 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v51 : ExactMatrix q2 n ((1 + 1) + 1) d6)) 2 3 0 p78 v56 ∧
    MxxRuntime.blockModSwitchRuns p79 (1) v56 v57 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v45 : ExactMatrix q2 n ((1 + 1) + 1) d5)) 0 1 0 p80 v58 ∧
    MxxRuntime.blockModSwitchRuns p81 (1) v58 v59 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v41 v51 : ExactMatrix q2 n ((1 + 1) + 1) d6)) 0 1 0 p82 v60 ∧
    MxxRuntime.blockModSwitchRuns p83 (1) v60 v61 ∧
    outputs = (v47, (v53, (v55, (v57, (v59, (v61, ()))))))

/-- Computation bundle of role `prf.key-cell0` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def PrfKeyCell0.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (d1 : Nat) (q1 : Nat) (q2 : Nat) (d2 : Nat) (q3 : Nat) (q4 : Nat) (q5 : Nat) (d3 : Nat) (q6 : Nat) (d4 : Nat) (q7 : Nat) (d5 : Nat) (q8 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (d10 : Nat) (q9 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (p63 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ByteArray × Int × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d1 × ExactMatrix q0 n 1 d1 × ExactMatrix q3 n 1 1 × Int × Int × ExactMatrix q4 n 1 1 × Unit) (outputs : ExactMatrix q5 n 1 d3 × ExactMatrix q6 n 1 d4 × ExactMatrix q7 n 1 d5 × ExactMatrix q8 n 1 d6 × ExactMatrix q2 n 1 d2 × ExactMatrix q2 n 1 d2 × ExactMatrix q2 n 1 d2 × ExactMatrix q2 n 1 d2 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d7) (v1 : ExactMatrix q0 n 1 d7) (v2 : ExactMatrix q0 n (1 + 1) d7) (v3 : ExactMatrix q0 n d7 d0) (v4 : ExactMatrix q0 n 1 d8) (v5 : ExactMatrix q0 n 1 d8) (v6 : ExactMatrix q0 n (1 + 1) d8) (v7 : ExactMatrix q0 n d8 d0) (v8 : ExactMatrix q0 n 1 d9) (v9 : ExactMatrix q0 n 1 d9) (v10 : ExactMatrix q0 n (1 + 1) d9) (v11 : ExactMatrix q2 n 1 d2) (v12 : ExactMatrix q2 n d9 d2) (v13 : ExactMatrix q0 n d9 d2) (v14 : ExactMatrix q0 n d7 d1) (v15 : ExactMatrix q0 n d8 d1) (v16 : ExactMatrix q0 n 1 d10) (v17 : ExactMatrix q0 n 1 d10) (v18 : ExactMatrix q0 n (1 + 1) d10) (v19 : ExactMatrix q2 n 1 1) (v20 : ExactMatrix q9 n 1 d2) (v21 : ExactMatrix q2 n 1 d2) (v22 : ExactMatrix q2 n 1 1) (v23 : ExactMatrix q2 n 1 d2) (v24 : ExactMatrix q2 n d10 d2) (v25 : ExactMatrix q0 n d10 d2) (v26 : ExactMatrix q0 n 1 d3) (v27 : ExactMatrix q5 n 1 d3) (v28 : ExactMatrix q2 n 1 d2) (v29 : ExactMatrix q2 n d9 d2) (v30 : ExactMatrix q0 n d9 d2) (v31 : ExactMatrix q2 n 1 1) (v32 : ExactMatrix q2 n 1 d2) (v33 : ExactMatrix q2 n d10 d2) (v34 : ExactMatrix q0 n d10 d2) (v35 : ExactMatrix q0 n 1 d4) (v36 : ExactMatrix q6 n 1 d4) (v37 : ExactMatrix q0 n 1 d5) (v38 : ExactMatrix q7 n 1 d5) (v39 : ExactMatrix q0 n 1 d6) (v40 : ExactMatrix q8 n 1 d6),
    p0 ≤ inputs.2.1 ∧
    inputs.2.1 < p1 ^ p2 ∧
    MxxRuntime.hashSample (hashModel) p3 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v0 ∧
    p4 ≤ inputs.2.1 ∧
    inputs.2.1 < p5 ^ p6 ∧
    MxxRuntime.hashSample (hashModel) p7 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.1) v1 ∧
    MxxRuntime.concatRows v0 v1 v2 ∧
    MxxRuntime.gadgetDecomposeRuns backend p8 p9 inputs.2.2.1 v3 ∧
    p10 ≤ inputs.2.1 ∧
    inputs.2.1 < p11 ^ p12 ∧
    MxxRuntime.hashSample (hashModel) p13 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v4 ∧
    p14 ≤ inputs.2.1 ∧
    inputs.2.1 < p15 ^ p16 ∧
    MxxRuntime.hashSample (hashModel) p17 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.1) v5 ∧
    MxxRuntime.concatRows v4 v5 v6 ∧
    MxxRuntime.gadgetDecomposeRuns backend p18 p19 inputs.2.2.2.1 v7 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v3 : ExactMatrix q0 n (1 + 1) d0)) ((MxxRuntime.matrixMul v6 v7 : ExactMatrix q0 n (1 + 1) d0)) : ExactMatrix q0 n (1 + 1) d0)) 0 1 0 p20 v8 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v3 : ExactMatrix q0 n (1 + 1) d0)) ((MxxRuntime.matrixMul v6 v7 : ExactMatrix q0 n (1 + 1) d0)) : ExactMatrix q0 n (1 + 1) d0)) 1 2 0 p21 v9 ∧
    MxxRuntime.concatRows v8 v9 v10 ∧
    MxxRuntime.hashSample (hashModel) p24 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p22), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p23), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v11 ∧
    MxxRuntime.gadgetDecomposeRuns backend p25 p26 v11 v12 ∧
    MxxRuntime.compactCenteredRebaseRuns p27 v12 v13 ∧
    MxxRuntime.gadgetDecomposeRuns backend p28 p29 inputs.2.2.2.2.1 v14 ∧
    MxxRuntime.gadgetDecomposeRuns backend p30 p31 inputs.2.2.2.2.2.1 v15 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v14 : ExactMatrix q0 n (1 + 1) d1)) ((MxxRuntime.matrixMul v6 v15 : ExactMatrix q0 n (1 + 1) d1)) : ExactMatrix q0 n (1 + 1) d1)) 0 1 0 p32 v16 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v2 v14 : ExactMatrix q0 n (1 + 1) d1)) ((MxxRuntime.matrixMul v6 v15 : ExactMatrix q0 n (1 + 1) d1)) : ExactMatrix q0 n (1 + 1) d1)) 1 2 0 p33 v17 ∧
    MxxRuntime.concatRows v16 v17 v18 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.2.2.2.2.1 v19 ∧
    MxxRuntime.gadgetMatrixRuns backend p34 p35 v20 ∧
    MxxRuntime.centeredRebaseRuns v20 v21 ∧
    MxxRuntime.modulusReduceRuns ((MxxRuntime.multiplyMonomial ((MxxRuntime.liftInteger inputs.2.2.2.2.2.2.2.1 : ExactMatrix q1 n 1 1)) inputs.2.2.2.2.2.2.2.2.1 : ExactMatrix q1 n 1 1)) v22 ∧
    MxxRuntime.gaussianSample (p36) (p37) v23 ∧
    MxxRuntime.gadgetDecomposeRuns backend p38 p39 ((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v19 v21 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixPolynomial p40 : ExactMatrix q2 n 1 1)) : ExactMatrix q2 n 1 d2)) v22 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixMulScalarLeft v19 v11 : ExactMatrix q2 n 1 d2)) : ExactMatrix q2 n 1 d2)) v23 : ExactMatrix q2 n 1 d2)) v24 ∧
    MxxRuntime.compactCenteredRebaseRuns p41 v24 v25 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v10 v13 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v18 v25 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 0 1 0 p42 v26 ∧
    MxxRuntime.blockModSwitchRuns p43 (1) v26 v27 ∧
    MxxRuntime.hashSample (hashModel) p46 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p44), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p45), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v28 ∧
    MxxRuntime.gadgetDecomposeRuns backend p47 p48 v28 v29 ∧
    MxxRuntime.compactCenteredRebaseRuns p49 v29 v30 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.2.2.2.2.2.2.2.1 v31 ∧
    MxxRuntime.gaussianSample (p50) (p51) v32 ∧
    MxxRuntime.gadgetDecomposeRuns backend p52 p53 ((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v31 v21 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixPolynomial p54 : ExactMatrix q2 n 1 1)) : ExactMatrix q2 n 1 d2)) v22 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixMulScalarLeft v19 v28 : ExactMatrix q2 n 1 d2)) : ExactMatrix q2 n 1 d2)) v32 : ExactMatrix q2 n 1 d2)) v33 ∧
    MxxRuntime.compactCenteredRebaseRuns p55 v33 v34 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v10 v30 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v18 v34 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 0 1 0 p56 v35 ∧
    MxxRuntime.blockModSwitchRuns p57 (1) v35 v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v10 v13 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v18 v25 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 1 2 0 p58 v37 ∧
    MxxRuntime.blockModSwitchRuns p59 (1) v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v10 v30 : ExactMatrix q0 n (1 + 1) d2)) ((MxxRuntime.matrixMul v18 v34 : ExactMatrix q0 n (1 + 1) d2)) : ExactMatrix q0 n (1 + 1) d2)) 1 2 0 p60 v39 ∧
    MxxRuntime.blockModSwitchRuns p61 (1) v39 v40 ∧
    outputs = (v27, (v36, (v38, (v40, (v11, (v28, (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v19 v21 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixPolynomial p62 : ExactMatrix q2 n 1 1)) : ExactMatrix q2 n 1 d2)) v22 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixMulScalarLeft v19 v11 : ExactMatrix q2 n 1 d2)) : ExactMatrix q2 n 1 d2)) v23 : ExactMatrix q2 n 1 d2)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v31 v21 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixPolynomial p63 : ExactMatrix q2 n 1 1)) : ExactMatrix q2 n 1 d2)) v22 : ExactMatrix q2 n 1 d2)) ((MxxRuntime.matrixMulScalarLeft v19 v28 : ExactMatrix q2 n 1 d2)) : ExactMatrix q2 n 1 d2)) v32 : ExactMatrix q2 n 1 d2)), ()))))))))

/-- Computation bundle of role `prf.key-cell1` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def PrfKeyCell1.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (d2 : Nat) (q2 : Nat) (q3 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (q4 : Nat) (q5 : Nat) (q6 : Nat) (d8 : Nat) (q7 : Nat) (d9 : Nat) (q8 : Nat) (d10 : Nat) (q9 : Nat) (d11 : Nat) (d12 : Nat) (q10 : Nat) (d13 : Nat) (d14 : Nat) (q11 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (p63 : _) (p64 : _) (p65 : _) (p66 : _) (p67 : _) (p68 : _) (p69 : _) (p70 : _) (p71 : _) (p72 : _) (p73 : _) (p74 : _) (p75 : _) (p76 : _) (p77 : _) (p78 : _) (p79 : _) (p80 : _) (p81 : _) (p82 : _) (p83 : _) (p84 : _) (p85 : _) (p86 : _) (p87 : _) (p88 : _) (p89 : _) (p90 : _) (p91 : _) (p92 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (inputs : ByteArray × Int × (Fin d4 → ExactMatrix q1 n 1 d1) × (Fin d5 → ExactMatrix q1 n 1 d1) × (Fin d6 → ExactMatrix q1 n 1 d2) × (Fin d7 → ExactMatrix q1 n 1 d2) × ExactMatrix q4 n 1 1 × Int × Int × ExactMatrix q5 n 1 1 × Unit) (outputs : ExactMatrix q6 n 1 d8 × ExactMatrix q7 n 1 d9 × ExactMatrix q8 n 1 d10 × ExactMatrix q9 n 1 d11 × ExactMatrix q3 n 1 d3 × ExactMatrix q3 n 1 d3 × ExactMatrix q3 n 1 d3 × ExactMatrix q3 n 1 d3 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d12) (v1 : ExactMatrix q10 n 1 d0) (v2 : ExactMatrix q0 n 1 d0) (v3 : ExactMatrix q0 n d12 d0) (v4 : ExactMatrix q1 n 1 d0) (v5 : ExactMatrix q0 n 1 d12) (v6 : ExactMatrix q1 n 1 d0) (v7 : ExactMatrix q1 n (1 + 1) d0) (v8 : ExactMatrix q1 n 1 d1) (v9 : ExactMatrix q1 n d0 d1) (v10 : ExactMatrix q0 n 1 d12) (v11 : ExactMatrix q1 n 1 d0) (v12 : ExactMatrix q0 n 1 d12) (v13 : ExactMatrix q1 n 1 d0) (v14 : ExactMatrix q1 n (1 + 1) d0) (v15 : ExactMatrix q1 n 1 d1) (v16 : ExactMatrix q1 n d0 d1) (v17 : ExactMatrix q1 n 1 d13) (v18 : ExactMatrix q1 n 1 d13) (v19 : ExactMatrix q1 n (1 + 1) d13) (v20 : ExactMatrix q3 n 1 d3) (v21 : ExactMatrix q3 n d13 d3) (v22 : ExactMatrix q1 n d13 d3) (v23 : ExactMatrix q1 n 1 d2) (v24 : ExactMatrix q1 n d0 d2) (v25 : ExactMatrix q1 n 1 d2) (v26 : ExactMatrix q1 n d0 d2) (v27 : ExactMatrix q1 n 1 d14) (v28 : ExactMatrix q1 n 1 d14) (v29 : ExactMatrix q1 n (1 + 1) d14) (v30 : ExactMatrix q3 n 1 1) (v31 : ExactMatrix q11 n 1 d3) (v32 : ExactMatrix q3 n 1 d3) (v33 : ExactMatrix q3 n 1 1) (v34 : ExactMatrix q3 n 1 d3) (v35 : ExactMatrix q3 n d14 d3) (v36 : ExactMatrix q1 n d14 d3) (v37 : ExactMatrix q1 n 1 d8) (v38 : ExactMatrix q6 n 1 d8) (v39 : ExactMatrix q3 n 1 d3) (v40 : ExactMatrix q3 n d13 d3) (v41 : ExactMatrix q1 n d13 d3) (v42 : ExactMatrix q3 n 1 1) (v43 : ExactMatrix q3 n 1 d3) (v44 : ExactMatrix q3 n d14 d3) (v45 : ExactMatrix q1 n d14 d3) (v46 : ExactMatrix q1 n 1 d9) (v47 : ExactMatrix q7 n 1 d9) (v48 : ExactMatrix q1 n 1 d10) (v49 : ExactMatrix q8 n 1 d10) (v50 : ExactMatrix q1 n 1 d11) (v51 : ExactMatrix q9 n 1 d11),
    p0 ≤ inputs.2.1 ∧
    inputs.2.1 < p1 ^ p2 ∧
    MxxRuntime.hashSample (hashModel) p3 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v0 ∧
    MxxRuntime.gadgetMatrixRuns backend p4 p5 v1 ∧
    MxxRuntime.centeredRebaseRuns v1 v2 ∧
    MxxRuntime.gadgetDecomposeRuns backend p6 p7 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p8 : ExactMatrix q0 n 1 1)) v2 : ExactMatrix q0 n 1 d0)) v3 ∧
    MxxRuntime.blockModSwitchRuns p9 (1) ((MxxRuntime.matrixMul v0 v3 : ExactMatrix q0 n 1 d0)) v4 ∧
    p10 ≤ inputs.2.1 ∧
    inputs.2.1 < p11 ^ p12 ∧
    MxxRuntime.hashSample (hashModel) p13 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.1) v5 ∧
    MxxRuntime.blockModSwitchRuns p14 (1) ((MxxRuntime.matrixMul v5 v3 : ExactMatrix q0 n 1 d0)) v6 ∧
    MxxRuntime.concatRows v4 v6 v7 ∧
    p15 ≤ ((Int.fdiv ((Int.ofNat i)) (p16))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p18))) < p17 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p19))) v8 ∧
    MxxRuntime.gadgetDecomposeRuns backend p20 p21 v8 v9 ∧
    p22 ≤ inputs.2.1 ∧
    inputs.2.1 < p23 ^ p24 ∧
    MxxRuntime.hashSample (hashModel) p25 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v10 ∧
    MxxRuntime.blockModSwitchRuns p26 (1) ((MxxRuntime.matrixMul v10 v3 : ExactMatrix q0 n 1 d0)) v11 ∧
    p27 ≤ inputs.2.1 ∧
    inputs.2.1 < p28 ^ p29 ∧
    MxxRuntime.hashSample (hashModel) p30 ([.bytes [109, 101, 109, 98, 101, 114], .u64Le inputs.2.1, .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.1) v12 ∧
    MxxRuntime.blockModSwitchRuns p31 (1) ((MxxRuntime.matrixMul v12 v3 : ExactMatrix q0 n 1 d0)) v13 ∧
    MxxRuntime.concatRows v11 v13 v14 ∧
    p32 ≤ ((Int.fdiv ((Int.ofNat i)) (p33))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p35))) < p34 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p36))) v15 ∧
    MxxRuntime.gadgetDecomposeRuns backend p37 p38 v15 v16 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v7 v9 : ExactMatrix q1 n (1 + 1) d1)) ((MxxRuntime.matrixMul v14 v16 : ExactMatrix q1 n (1 + 1) d1)) : ExactMatrix q1 n (1 + 1) d1)) 0 1 0 p39 v17 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v7 v9 : ExactMatrix q1 n (1 + 1) d1)) ((MxxRuntime.matrixMul v14 v16 : ExactMatrix q1 n (1 + 1) d1)) : ExactMatrix q1 n (1 + 1) d1)) 1 2 0 p40 v18 ∧
    MxxRuntime.concatRows v17 v18 v19 ∧
    MxxRuntime.hashSample (hashModel) p43 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p41), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p42), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.1) v20 ∧
    MxxRuntime.gadgetDecomposeRuns backend p44 p45 v20 v21 ∧
    MxxRuntime.compactCenteredRebaseRuns p46 v21 v22 ∧
    p47 ≤ ((Int.fdiv ((Int.ofNat i)) (p48))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p50))) < p49 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p51))) v23 ∧
    MxxRuntime.gadgetDecomposeRuns backend p52 p53 v23 v24 ∧
    p54 ≤ ((Int.fdiv ((Int.ofNat i)) (p55))) ∧
    ((Int.fdiv ((Int.ofNat i)) (p57))) < p56 ∧
    MxxRuntime.familyGetDynamic inputs.2.2.2.2.2.1 ((Int.fdiv ((Int.ofNat i)) (p58))) v25 ∧
    MxxRuntime.gadgetDecomposeRuns backend p59 p60 v25 v26 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v7 v24 : ExactMatrix q1 n (1 + 1) d2)) ((MxxRuntime.matrixMul v14 v26 : ExactMatrix q1 n (1 + 1) d2)) : ExactMatrix q1 n (1 + 1) d2)) 0 1 0 p61 v27 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v7 v24 : ExactMatrix q1 n (1 + 1) d2)) ((MxxRuntime.matrixMul v14 v26 : ExactMatrix q1 n (1 + 1) d2)) : ExactMatrix q1 n (1 + 1) d2)) 1 2 0 p62 v28 ∧
    MxxRuntime.concatRows v27 v28 v29 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.gadgetMatrixRuns backend p63 p64 v31 ∧
    MxxRuntime.centeredRebaseRuns v31 v32 ∧
    MxxRuntime.modulusReduceRuns ((MxxRuntime.multiplyMonomial ((MxxRuntime.liftInteger inputs.2.2.2.2.2.2.2.1 : ExactMatrix q2 n 1 1)) inputs.2.2.2.2.2.2.2.2.1 : ExactMatrix q2 n 1 1)) v33 ∧
    MxxRuntime.gaussianSample (p65) (p66) v34 ∧
    MxxRuntime.gadgetDecomposeRuns backend p67 p68 ((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v30 v32 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixPolynomial p69 : ExactMatrix q3 n 1 1)) : ExactMatrix q3 n 1 d3)) v33 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixMulScalarLeft v30 v20 : ExactMatrix q3 n 1 d3)) : ExactMatrix q3 n 1 d3)) v34 : ExactMatrix q3 n 1 d3)) v35 ∧
    MxxRuntime.compactCenteredRebaseRuns p70 v35 v36 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v19 v22 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v29 v36 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 0 1 0 p71 v37 ∧
    MxxRuntime.blockModSwitchRuns p72 (1) v37 v38 ∧
    MxxRuntime.hashSample (hashModel) p75 ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal (p73), .bytes [112, 114, 111, 103, 114, 97, 109], .decimal (p74), .bytes [109, 101, 109, 98, 101, 114], .u64Le ((Int.ofNat i)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.1) v39 ∧
    MxxRuntime.gadgetDecomposeRuns backend p76 p77 v39 v40 ∧
    MxxRuntime.compactCenteredRebaseRuns p78 v40 v41 ∧
    MxxRuntime.modulusReduceRuns inputs.2.2.2.2.2.2.2.2.2.1 v42 ∧
    MxxRuntime.gaussianSample (p79) (p80) v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p81 p82 ((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v42 v32 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixPolynomial p83 : ExactMatrix q3 n 1 1)) : ExactMatrix q3 n 1 d3)) v33 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixMulScalarLeft v30 v39 : ExactMatrix q3 n 1 d3)) : ExactMatrix q3 n 1 d3)) v43 : ExactMatrix q3 n 1 d3)) v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p84 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v19 v41 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v29 v45 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 0 1 0 p85 v46 ∧
    MxxRuntime.blockModSwitchRuns p86 (1) v46 v47 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v19 v22 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v29 v36 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 1 2 0 p87 v48 ∧
    MxxRuntime.blockModSwitchRuns p88 (1) v48 v49 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixAdd ((MxxRuntime.matrixMul v19 v41 : ExactMatrix q1 n (1 + 1) d3)) ((MxxRuntime.matrixMul v29 v45 : ExactMatrix q1 n (1 + 1) d3)) : ExactMatrix q1 n (1 + 1) d3)) 1 2 0 p89 v50 ∧
    MxxRuntime.blockModSwitchRuns p90 (1) v50 v51 ∧
    outputs = (v38, (v47, (v49, (v51, (v20, (v39, (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v30 v32 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixPolynomial p91 : ExactMatrix q3 n 1 1)) : ExactMatrix q3 n 1 d3)) v33 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixMulScalarLeft v30 v20 : ExactMatrix q3 n 1 d3)) : ExactMatrix q3 n 1 d3)) v34 : ExactMatrix q3 n 1 d3)), (((MxxRuntime.matrixAdd ((MxxRuntime.matrixSub ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarRight ((MxxRuntime.matrixMulScalarLeft v42 v32 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixPolynomial p92 : ExactMatrix q3 n 1 1)) : ExactMatrix q3 n 1 d3)) v33 : ExactMatrix q3 n 1 d3)) ((MxxRuntime.matrixMulScalarLeft v30 v39 : ExactMatrix q3 n 1 d3)) : ExactMatrix q3 n 1 d3)) v43 : ExactMatrix q3 n 1 d3)), ()))))))))

/-- Computation bundle of role `smudge.key-member` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def SmudgeKeyMember.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (n1 : Nat) (d1 : Nat) (q2 : Nat) (n2 : Nat) (d2 : Nat) (d3 : Nat) (q3 : Nat) (n3 : Nat) (d4 : Nat) (q4 : Nat) (n4 : Nat) (d5 : Nat) (q5 : Nat) (n5 : Nat) (d6 : Nat) (d7 : Nat) (q6 : Nat) (n6 : Nat) (d8 : Nat) (d9 : Nat) (q7 : Nat) (n7 : Nat) (d10 : Nat) (d11 : Nat) (q8 : Nat) (n8 : Nat) (d12 : Nat) (q9 : Nat) (q10 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit → Prop) (inputs : ByteArray × ExactMatrix q1 n1 1 d1 × ExactMatrix q2 n2 1 d2 × (Fin d3 → ExactMatrix q3 n3 1 d4) × ExactMatrix q4 n4 1 d5 × ExactMatrix q5 n5 1 d6 × (Fin d7 → ExactMatrix q6 n6 1 d8) × (Fin d9 → ExactMatrix q7 n7 1 d10) × (Fin d11 → ExactMatrix q8 n8 1 d12) × Unit) (outputs : ExactMatrix q10 n 1 1 × ExactMatrix q10 n 1 1 × ExactMatrix q10 n 1 1 × ExactMatrix q10 n 1 1 × Unit) :=
  ∃ (v0 : ExactMatrix q0 n 1 d0) (v1 : ExactMatrix q0 n 1 d0) (v2 : ExactMatrix q0 n 1 d0) (v3 : ExactMatrix q0 n 1 d0) (v4 : ExactMatrix q10 n 1 d0) (v5 : ExactMatrix q9 n d0 1) (v6 : ExactMatrix q10 n d0 1) (v7 : ExactMatrix q10 n 1 d0) (v8 : ExactMatrix q10 n 1 d0) (v9 : ExactMatrix q10 n 1 d0),
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, ()))))))))))))) next) (Int.toNat p0) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), ())))) (v0, (v1, (v2, (v3, ())))) ∧
    MxxRuntime.blockModSwitchRuns p1 (1) v0 v4 ∧
    MxxRuntime.gadgetDecomposeRuns backend p2 p3 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p4 : ExactMatrix q9 n 1 1)) ((MxxRuntime.liftInteger ((p5) ^ ((Int.fmod ((Int.ofNat i)) (p6))).toNat) : ExactMatrix q9 n 1 1)) : ExactMatrix q9 n 1 1)) v5 ∧
    MxxRuntime.compactCenteredRebaseRuns p7 v5 v6 ∧
    MxxRuntime.blockModSwitchRuns p8 (1) v1 v7 ∧
    MxxRuntime.blockModSwitchRuns p9 (1) v2 v8 ∧
    MxxRuntime.blockModSwitchRuns p10 (1) v3 v9 ∧
    outputs = (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v4 v6 : ExactMatrix q10 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p11))) (p12))) : ExactMatrix q10 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v7 v6 : ExactMatrix q10 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p13))) (p14))) : ExactMatrix q10 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v8 v6 : ExactMatrix q10 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p15))) (p16))) : ExactMatrix q10 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v9 v6 : ExactMatrix q10 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p17))) (p18))) : ExactMatrix q10 n 1 1)), ()))))

/-- Computation bundle of role `smudge.member` (variant 0): the scope relation with dimensions, moduli and constants as parameters. -/
def SmudgeMember.Comp (q0 : Nat) (n : Nat) (d0 : Nat) (q1 : Nat) (d1 : Nat) (q2 : Nat) (d2 : Nat) (d3 : Nat) (d4 : Nat) (d5 : Nat) (d6 : Nat) (d7 : Nat) (d8 : Nat) (d9 : Nat) (d10 : Nat) (d11 : Nat) (d12 : Nat) (d13 : Nat) (q3 : Nat) (d14 : Nat) (q4 : Nat) (q5 : Nat) (q6 : Nat) (p0 : _) (p1 : _) (p2 : _) (p3 : _) (p4 : _) (p5 : _) (p6 : _) (p7 : _) (p8 : _) (p9 : _) (p10 : _) (p11 : _) (p12 : _) (p13 : _) (p14 : _) (p15 : _) (p16 : _) (p17 : _) (p18 : _) (p19 : _) (p20 : _) (p21 : _) (p22 : _) (p23 : _) (p24 : _) (p25 : _) (p26 : _) (p27 : _) (p28 : _) (p29 : _) (p30 : _) (p31 : _) (p32 : _) (p33 : _) (p34 : _) (p35 : _) (p36 : _) (p37 : _) (p38 : _) (p39 : _) (p40 : _) (p41 : _) (p42 : _) (p43 : _) (p44 : _) (p45 : _) (p46 : _) (p47 : _) (p48 : _) (p49 : _) (p50 : _) (p51 : _) (p52 : _) (p53 : _) (p54 : _) (p55 : _) (p56 : _) (p57 : _) (p58 : _) (p59 : _) (p60 : _) (p61 : _) (p62 : _) (p63 : _) (p64 : _) (p65 : _) (p66 : _) (p67 : _) (p68 : _) (p69 : _) (p70 : _) (p71 : _) (p72 : _) (p73 : _) (p74 : _) (p75 : _) (p76 : _) (p77 : _) (p78 : _) (p79 : _) (p80 : _) (p81 : _) (p82 : _) (p83 : _) (p84 : _) (p85 : _) (p86 : _) (backend : MxxRuntime.BackendContext) (hashModel : MxxRuntime.HashModel) (i : Nat) (child : Nat → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × Unit → ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit → Prop) (inputs : (Fin d9 → ExactMatrix q1 n 1 d1) × (Fin d10 → ExactMatrix q2 n 1 d2) × (Fin d11 → ExactMatrix q1 n 1 d1) × (Fin d12 → ExactMatrix q2 n 1 d2) × (Fin d3 → ExactMatrix q0 n 1 d4) × ByteArray × (Fin d5 → ExactMatrix q0 n 1 d6) × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d7 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × ExactMatrix q0 n 1 d8 × Unit) (outputs : ExactMatrix q4 n 1 1 × ExactMatrix q4 n 1 1 × ExactMatrix q4 n 1 1 × ExactMatrix q4 n 1 1 × ExactMatrix q4 n 1 1 × ExactMatrix q4 n 1 1 × Unit) :=
  ∃ (v0 : ExactMatrix q1 n 1 d1) (v1 : ExactMatrix q2 n 1 d2) (v2 : ExactMatrix q1 n 1 d1) (v3 : ExactMatrix q2 n 1 d2) (v4 : ExactMatrix q0 n 1 d0) (v5 : ExactMatrix q0 n 1 d0) (v6 : ExactMatrix q0 n 1 d0) (v7 : ExactMatrix q0 n 1 d0) (v8 : ExactMatrix q0 n 1 d0) (v9 : ExactMatrix q0 n 1 d0) (v10 : ExactMatrix q1 n 1 d1) (v11 : ExactMatrix q2 n 1 d2) (v12 : ExactMatrix q1 n 1 d1) (v13 : ExactMatrix q2 n 1 d2) (v14 : ExactMatrix q0 n 1 d4) (v15 : ExactMatrix q0 n 1 d4) (v16 : ExactMatrix q0 n 1 d4) (v17 : ExactMatrix q0 n ((1 + 1) + 1) d4) (v18 : ExactMatrix q0 n (1 + 1) d4) (v19 : ExactMatrix q0 n 1 d6) (v20 : ExactMatrix q0 n 1 d6) (v21 : ExactMatrix q0 n 1 d6) (v22 : ExactMatrix q0 n ((1 + 1) + 1) d6) (v23 : ExactMatrix q0 n (1 + 1) d6) (v24 : ExactMatrix q0 n ((1 + 1) + 1) (d4 + d6)) (v25 : ExactMatrix q0 n 1 d7) (v26 : ExactMatrix q0 n 1 d7) (v27 : ExactMatrix q0 n 1 d7) (v28 : ExactMatrix q0 n ((1 + 1) + 1) d7) (v29 : ExactMatrix q0 n (1 + 1) d7) (v30 : ExactMatrix q0 n 1 d8) (v31 : ExactMatrix q0 n 1 d8) (v32 : ExactMatrix q0 n 1 d8) (v33 : ExactMatrix q0 n ((1 + 1) + 1) d8) (v34 : ExactMatrix q0 n (1 + 1) d8) (v35 : ExactMatrix q0 n ((1 + 1) + 1) (d7 + d8)) (v36 : ExactMatrix q1 n (1 + 1) d1) (v37 : ExactMatrix q1 n (d7 + d8) d1) (v38 : ExactMatrix q0 n (d7 + d8) d1) (v39 : ExactMatrix q0 n 1 d13) (v40 : ExactMatrix q5 n 1 d13) (v41 : ExactMatrix q0 n 1 d13) (v42 : ExactMatrix q5 n 1 d13) (v43 : ExactMatrix q5 n (1 + 1) d13) (v44 : ExactMatrix q5 n (d4 + d6) d13) (v45 : ExactMatrix q0 n (d4 + d6) d13) (v46 : ExactMatrix q0 n 1 d0) (v47 : ExactMatrix q4 n 1 d0) (v48 : ExactMatrix q3 n d0 1) (v49 : ExactMatrix q4 n d0 1) (v50 : ExactMatrix q2 n (1 + 1) d2) (v51 : ExactMatrix q2 n (d7 + d8) d2) (v52 : ExactMatrix q0 n (d7 + d8) d2) (v53 : ExactMatrix q0 n 1 d14) (v54 : ExactMatrix q6 n 1 d14) (v55 : ExactMatrix q0 n 1 d14) (v56 : ExactMatrix q6 n 1 d14) (v57 : ExactMatrix q6 n (1 + 1) d14) (v58 : ExactMatrix q6 n (d4 + d6) d14) (v59 : ExactMatrix q0 n (d4 + d6) d14) (v60 : ExactMatrix q0 n 1 d0) (v61 : ExactMatrix q4 n 1 d0) (v62 : ExactMatrix q0 n 1 d0) (v63 : ExactMatrix q4 n 1 d0) (v64 : ExactMatrix q0 n 1 d0) (v65 : ExactMatrix q4 n 1 d0) (v66 : ExactMatrix q0 n 1 d0) (v67 : ExactMatrix q0 n 1 d0) (v68 : ExactMatrix q4 n 1 d0) (v69 : ExactMatrix q0 n 1 d0) (v70 : ExactMatrix q0 n 1 d0) (v71 : ExactMatrix q4 n 1 d0),
    MxxRuntime.familyGetStatic inputs.1 (p0) v0 ∧
    MxxRuntime.familyGetStatic inputs.2.1 (p1) v1 ∧
    MxxRuntime.familyGetStatic inputs.2.2.1 (p2) v2 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.1 (p3) v3 ∧
    MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q0 n 1 d0 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × ExactMatrix q1 n 1 d1 × ExactMatrix q2 n 1 d2 × Unit) => child i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (current.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.1, (current.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.1, (inputs.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.2.2.2.2.2.2.2.2.2.2.2.2.1, (inputs.1, (inputs.2.1, (inputs.2.2.1, (inputs.2.2.2.1, ()))))))))))))))))))))))) next) (Int.toNat p4) (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (((0 : ExactMatrix q0 n 1 d0)), (v0, (v1, (v2, (v3, ())))))))))) (v4, (v5, (v6, (v7, (v8, (v9, (v10, (v11, (v12, (v13, ())))))))))) ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.1 (p5) v14 ∧
    MxxRuntime.hashSample (hashModel) p7 ([.u64Le (p6), .integer ((MxxIR.selectInt (p8) p9)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (0)]) (inputs.2.2.2.2.2.1) v15 ∧
    MxxRuntime.hashSample (hashModel) p11 ([.u64Le (p10), .integer ((MxxIR.selectInt (p12) p13)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (2)]) (inputs.2.2.2.2.2.1) v16 ∧
    MxxRuntime.concatRows v14 v15 v18 ∧
    MxxRuntime.concatRows v18 v16 v17 ∧
    MxxRuntime.familyGetStatic inputs.2.2.2.2.2.2.1 (p14) v19 ∧
    MxxRuntime.hashSample (hashModel) p16 ([.u64Le (p15), .integer ((MxxIR.selectInt (p17) p18)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (1)]) (inputs.2.2.2.2.2.1) v20 ∧
    MxxRuntime.hashSample (hashModel) p20 ([.u64Le (p19), .integer ((MxxIR.selectInt (p21) p22)), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le (3)]) (inputs.2.2.2.2.2.1) v21 ∧
    MxxRuntime.concatRows v19 v20 v23 ∧
    MxxRuntime.concatRows v23 v21 v22 ∧
    MxxRuntime.concatColumns v17 v22 v24 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p23) p24)) inputs.2.2.2.2.2.2.2.1 v25 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p25) p26)) inputs.2.2.2.2.2.2.2.2.1 v26 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p27) p28)) inputs.2.2.2.2.2.2.2.2.2.1 v27 ∧
    MxxRuntime.concatRows v25 v26 v29 ∧
    MxxRuntime.concatRows v29 v27 v28 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p29) p30)) inputs.2.2.2.2.2.2.2.2.2.2.1 v30 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p31) p32)) inputs.2.2.2.2.2.2.2.2.2.2.2.1 v31 ∧
    MxxRuntime.ringAutomorphismRuns ((MxxIR.selectInt (p33) p34)) inputs.2.2.2.2.2.2.2.2.2.2.2.2.1 v32 ∧
    MxxRuntime.concatRows v30 v31 v34 ∧
    MxxRuntime.concatRows v34 v32 v33 ∧
    MxxRuntime.concatColumns v28 v33 v35 ∧
    MxxRuntime.concatRows v10 v12 v36 ∧
    MxxRuntime.gadgetDecomposeRuns backend p35 p36 v36 v37 ∧
    MxxRuntime.compactCenteredRebaseRuns p37 v37 v38 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 1 2 0 p38 v39 ∧
    MxxRuntime.blockModSwitchRuns p39 (1) v39 v40 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 2 3 0 p40 v41 ∧
    MxxRuntime.blockModSwitchRuns p41 (1) v41 v42 ∧
    MxxRuntime.concatRows v40 v42 v43 ∧
    MxxRuntime.gadgetDecomposeRuns backend p42 p43 v43 v44 ∧
    MxxRuntime.compactCenteredRebaseRuns p44 v44 v45 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 1 2 0 p45 v46 ∧
    MxxRuntime.blockModSwitchRuns p46 (1) ((MxxRuntime.matrixAdd v6 v46 : ExactMatrix q0 n 1 d0)) v47 ∧
    MxxRuntime.gadgetDecomposeRuns backend p47 p48 ((MxxRuntime.matrixMulScalarLeft ((MxxRuntime.matrixPolynomial p49 : ExactMatrix q3 n 1 1)) ((MxxRuntime.liftInteger ((p50) ^ ((Int.fmod ((Int.ofNat i)) (p51))).toNat) : ExactMatrix q3 n 1 1)) : ExactMatrix q3 n 1 1)) v48 ∧
    MxxRuntime.compactCenteredRebaseRuns p52 v48 v49 ∧
    MxxRuntime.concatRows v11 v13 v50 ∧
    MxxRuntime.gadgetDecomposeRuns backend p53 p54 v50 v51 ∧
    MxxRuntime.compactCenteredRebaseRuns p55 v51 v52 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v52 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 1 2 0 p56 v53 ∧
    MxxRuntime.blockModSwitchRuns p57 (1) v53 v54 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v52 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 2 3 0 p58 v55 ∧
    MxxRuntime.blockModSwitchRuns p59 (1) v55 v56 ∧
    MxxRuntime.concatRows v54 v56 v57 ∧
    MxxRuntime.gadgetDecomposeRuns backend p60 p61 v57 v58 ∧
    MxxRuntime.compactCenteredRebaseRuns p62 v58 v59 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v59 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 1 2 0 p63 v60 ∧
    MxxRuntime.blockModSwitchRuns p64 (1) ((MxxRuntime.matrixAdd v7 v60 : ExactMatrix q0 n 1 d0)) v61 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 2 3 0 p65 v62 ∧
    MxxRuntime.blockModSwitchRuns p66 (1) ((MxxRuntime.matrixAdd v8 v62 : ExactMatrix q0 n 1 d0)) v63 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v59 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 2 3 0 p67 v64 ∧
    MxxRuntime.blockModSwitchRuns p68 (1) ((MxxRuntime.matrixAdd v9 v64 : ExactMatrix q0 n 1 d0)) v65 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v45 : ExactMatrix q0 n ((1 + 1) + 1) d13)) 0 1 0 p69 v66 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v38 : ExactMatrix q0 n ((1 + 1) + 1) d1)) 0 1 0 p70 v67 ∧
    MxxRuntime.blockModSwitchRuns p71 (1) ((MxxRuntime.matrixAdd v4 ((MxxRuntime.matrixAdd v66 v67 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v68 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v24 v59 : ExactMatrix q0 n ((1 + 1) + 1) d14)) 0 1 0 p72 v69 ∧
    MxxRuntime.sliceMatrix ((MxxRuntime.matrixMul v35 v52 : ExactMatrix q0 n ((1 + 1) + 1) d2)) 0 1 0 p73 v70 ∧
    MxxRuntime.blockModSwitchRuns p74 (1) ((MxxRuntime.matrixAdd v5 ((MxxRuntime.matrixAdd v69 v70 : ExactMatrix q0 n 1 d0)) : ExactMatrix q0 n 1 d0)) v71 ∧
    outputs = (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v47 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p75))) (p76))) : ExactMatrix q4 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v61 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p77))) (p78))) : ExactMatrix q4 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v63 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p79))) (p80))) : ExactMatrix q4 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v65 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p81))) (p82))) : ExactMatrix q4 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v68 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p83))) (p84))) : ExactMatrix q4 n 1 1)), (((MxxRuntime.multiplyMonomial ((MxxRuntime.matrixMul v71 v49 : ExactMatrix q4 n 1 1)) ((Int.fmod ((Int.fdiv ((Int.ofNat i)) (p85))) (p86))) : ExactMatrix q4 n 1 1)), ()))))))

-- END GENERATED COMP

end RefreshTopProof.GenIface
