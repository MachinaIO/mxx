import GenIface
import GenRel
import GenLoop
import ExponentDynamic

/-!
# Glue lemmas of the instance theorems (WP D)

Generic over every parameter: the bridges between the structural role facts of GenIface
(`RowSum.Spec`, …) and the encoding-level lemmas (GenRel, GenLoop).
-/

namespace RefreshTopProof.GenInst

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface RefreshTopProof.Gen
open scoped BigOperators

variable {n : Nat}

/-- The steps of an iteration satisfy `P` when every body step does. -/
theorem steps_of {σ : Type} {body : Nat → σ → σ → Prop} {count : Nat} {init final : σ}
    (h : MxxIR.IterRuns body count init final) {P : Nat → σ → σ → Prop}
    (hs : ∀ i cur next, body i cur next → P i cur next) : MxxIR.IterRuns P count init final :=
  RefreshTopProof.GenIface.seq_forall P hs h

/-- A six-family Gather loop whose child relation `P` implies `Q`: every iteration's `Q` holds at
the table entry's members. -/
theorem gather_sem {α : Type} {F K : Nat} {f0 f1 f2 f3 f4 f5 : Fin K → α} {t : Fin F → Int}
    (P : Fin F → Int → α → α → α → α → α → α → Prop)
    (h : ∀ i : Fin F, ∃ x0 x1 x2 x3 x4 x5 : α,
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f0 (t i) x0 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f1 (t i) x1 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f2 (t i) x2 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f3 (t i) x3 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f4 (t i) x4 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f5 (t i) x5 ∧
      P i (t i) x0 x1 x2 x3 x4 x5)
    {Q : Fin F → Int → α → α → α → α → α → α → Prop}
    (hs : ∀ i ti x0 x1 x2 x3 x4 x5, P i ti x0 x1 x2 x3 x4 x5 → Q i ti x0 x1 x2 x3 x4 x5)
    (i : Fin F) : ∃ k : Fin K, ((k : Nat) : Int) = t i ∧ Q i (t i) (f0 k) (f1 k) (f2 k) (f3 k) (f4 k) (f5 k) :=
  (Gen.gather_loop6 P h i).imp fun _ hk ↦ ⟨hk.1, hs _ _ _ _ _ _ _ _ hk.2⟩

/-- `gather_sem` for a three-family Gather loop (`Gen.gather_loop3`). -/
theorem gather_sem3 {α β γ : Type} {F K : Nat} {f0 : Fin K → α} {f1 : Fin K → β} {f2 : Fin K → γ}
    {t : Fin F → Int} (P : Fin F → Int → α → β → γ → Prop)
    (h : ∀ i : Fin F, ∃ (x0 : α) (x1 : β) (x2 : γ),
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f0 (t i) x0 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f1 (t i) x1 ∧
      0 ≤ t i ∧ t i < K ∧ MxxRuntime.familyGetDynamic f2 (t i) x2 ∧
      P i (t i) x0 x1 x2)
    {Q : Fin F → Int → α → β → γ → Prop} (hs : ∀ i ti x0 x1 x2, P i ti x0 x1 x2 → Q i ti x0 x1 x2)
    (i : Fin F) : ∃ k : Fin K, ((k : Nat) : Int) = t i ∧ Q i (t i) (f0 k) (f1 k) (f2 k) :=
  (Gen.gather_loop3 P h i).imp fun _ hk ↦ ⟨hk.1, hs _ _ _ _ _ hk.2⟩

/-- A row sum is the sum of the member's cells. -/
theorem rowSum_get {α : Type} [AddCommMonoid α] {M W k : Nat} {fams : Tup (Fin (M * W) → α) k}
    {o : Fin M} {out : Tup α k} (h : RowSum.Spec fams W o out) (c : Fin k) :
    out.get c = ∑ j : Fin W, fams.get c (cellAt o j) := by
  rw [h.2 c, Finset.sum_range]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  rw [famAt_of_lt _ (by have := j.isLt; have := o.isLt; nlinarith)]
  rfl

theorem enc_sum_A {q m W : Nat} (x : Fin W → Tup (ExactMatrix q n 1 m) 6) (out : Tup (ExactMatrix q n 1 m) 6)
    (h : ∀ c, out.get c = ∑ j, (x j).get c) : Enc.A out = ∑ j, Enc.A (x j) := by
  funext a b
  fin_cases a <;> fin_cases b <;> simp [Enc.A, Finset.sum_apply, h]

theorem enc_sum_c {q m W : Nat} (x : Fin W → Tup (ExactMatrix q n 1 m) 6) (out : Tup (ExactMatrix q n 1 m) 6)
    (h : ∀ c, out.get c = ∑ j, (x j).get c) : Enc.c out = ∑ j, Enc.c (x j) := by
  funext a
  fin_cases a <;> simp [Enc.c, Finset.sum_apply, h]

end RefreshTopProof.GenInst

namespace RefreshTopProof.GenInst

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.GenIface RefreshTopProof.Gen
open scoped BigOperators

variable {n : Nat}

/-- **One bucket of member `o`** from per-cell facts at the gathered table entries: cell `(o, j)`
encodes `p (β k μ_o s)` at its table entry `k = t (o, j)`; with one-hot selectors at `sel` the
row sum encodes `p (β key μ_o)` for the entry `key = t (o, sel)`. -/
theorem bucket_of_cells {q m M W K p : Nat} {sm sp : ErrorPoly n} {hT : ExactMatrix q n 1 m}
    (β : Fin K → ErrorPoly n) (μ : Fin M → ErrorPoly n) (t : Fin (M * W) → Int)
    (s : Fin (M * W) → ErrorPoly n) {fams : Tup (Fin (M * W) → ExactMatrix q n 1 m) 6} {Bc : Nat}
    (hcell : ∀ (o : Fin M) (j : Fin W), ∃ k : Fin K, ((k : Nat) : Int) = t (cellAt o j) ∧
      EncRel sm (Tup.ev fams (cellAt o j))
        (fun i ↦ reducePoly q n ((p : Int) * (β k * μ o * s (cellAt o j)) *
          unitPair sp i) • hT) Bc)
    (o : Fin M) {out : Tup (ExactMatrix q n 1 m) 6} (hrow : RowSum.Spec fams W o out)
    (sel : Fin W) (hsel : ∀ j, s (cellAt o j) = if j = sel then (1 : ErrorPoly n) else 0)
    (key : Fin K) (hkey : ((key : Nat) : Int) = t (cellAt o sel)) :
    EncRel sm out (fun i ↦ reducePoly q n ((p : Int) * (β key * μ o) * unitPair sp i) • hT)
      (W * Bc) := by
  choose tab htab hrel using hcell o
  have hsum := Rel.sum Finset.univ (fun j (_ : j ∈ Finset.univ) ↦ hrel j)
  have hg : ∀ c, out.get c = ∑ j, (Tup.ev fams (cellAt o j)).get c := fun c ↦ by
    rw [rowSum_get hrow c]; simp
  unfold EncRel
  rw [enc_sum_A _ out hg, enc_sum_c _ out hg]
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  have hk : tab sel = key := Fin.ext (by have := htab sel; omega)
  rw [Finset.sum_eq_single sel]
  · funext i
    simp [hsel, hk]
  · intro j _ hj
    funext i
    simp [hsel, hj]
  · simp

/-- A bucket-1 cell of member `o` (`⌊(W o + j) / W⌋ = o`): the member index of the cell facts. -/
theorem cells_at_member {M W K : Nat} {t : Fin (M * W) → Int} {P : Fin (M * W) → Fin K → Fin M → Prop}
    (h : ∀ i, ∃ k : Fin K, ((k : Nat) : Int) = t i ∧ ∃ o : Fin M, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) W ∧ P i k o)
    (o : Fin M) (j : Fin W) : ∃ k : Fin K, ((k : Nat) : Int) = t (cellAt o j) ∧ P (cellAt o j) k o := by
  obtain ⟨k, hk, o', ho', hp⟩ := h (cellAt o j)
  have : o' = o := by
    apply Fin.ext
    have hj := j.isLt
    have hW : 0 < W := by omega
    have e : Int.fdiv (Int.ofNat (cellAt o j : Nat)) W = (o : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), Int.ofNat_eq_natCast, ← Int.natCast_div]
      congr 1
      show (W * o.val + j.val) / W = o.val
      rw [Nat.mul_add_div hW, Nat.div_eq_of_lt hj, Nat.add_zero]
    omega
  subst this
  exact ⟨k, hk, hp⟩

/-- **Member total** (`sum.members` at the root): reads of members `0`, `1` as the initial state
and the member loop over `2 + i`, `i < M - 2`: every component is the sum of all `M` members. -/
theorem member_total {α : Type} [AddCommMonoid α] {M k : Nat} (hM : 2 ≤ M)
    {fams : Tup (Fin M → α) k} {a b init final : Tup α k} {body : Nat → Tup α k → Tup α k → Prop}
    (ha : ∀ c, familyGetStatic (fams.get c) 0 (a.get c))
    (hb : ∀ c, familyGetStatic (fams.get c) 1 (b.get c))
    (hinit : ∀ c, init.get c = a.get c + b.get c)
    (h : MxxIR.IterRuns body (M - 2) init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep.Spec fams (2 + (i : Int)) cur next) :
    ∀ c, final.get c = ∑ j : Fin M, fams.get c j := by
  have hs := MemberSum.sem (idx := fun i ↦ 2 + (i : Int)) h hstep
  intro c
  rw [hs.2 c, hinit c]
  obtain ⟨_, ea⟩ := familyGet_eq (k := 0) (ha c)
  obtain ⟨_, eb⟩ := familyGet_eq (k := 1) (hb c)
  rw [ea, eb]
  obtain ⟨M', rfl⟩ : ∃ M', M = M' + 2 := ⟨M - 2, by omega⟩
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ, Finset.sum_range, add_assoc]
  congr 2
  simp only [Nat.add_sub_cancel]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  have hj := j.isLt
  rw [show ((2 + ((j : Nat) : Int)).toNat) = j + 2 by omega, famAt_of_lt _ (by omega)]
  rfl

/-- The encoding relation of a member total. -/
theorem member_total_rel {q m M : Nat} (hM : 2 ≤ M) {sm : ErrorPoly n}
    {fams : Tup (Fin M → ExactMatrix q n 1 m) 6} {a b init final : Tup (ExactMatrix q n 1 m) 6}
    {body : Nat → Tup (ExactMatrix q n 1 m) 6 → Tup (ExactMatrix q n 1 m) 6 → Prop}
    (ha : ∀ c, familyGetStatic (fams.get c) 0 (a.get c))
    (hb : ∀ c, familyGetStatic (fams.get c) 1 (b.get c))
    (hinit : ∀ c, init.get c = a.get c + b.get c)
    (h : MxxIR.IterRuns body (M - 2) init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep.Spec fams (2 + (i : Int)) cur next)
    {t : Fin M → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hmem : ∀ j, EncRel sm (Tup.ev fams j) (t j) B) :
    EncRel sm final (∑ j, t j) (M * B) := by
  have hg : ∀ c, final.get c = ∑ j, (Tup.ev fams j).get c := fun c ↦ by
    rw [member_total hM ha hb hinit h hstep c]; simp
  have hsum := Rel.sum Finset.univ (fun j (_ : j ∈ Finset.univ) ↦ hmem j)
  unfold EncRel
  rw [enc_sum_A _ final hg, enc_sum_c _ final hg]
  exact hsum.mono (le_of_eq (by simp))

/-- **Placement** of an encoding by `X^(o mod N)`. -/
theorem place_rel (hn : 1 < n) {q m : Nat} {sm : ErrorPoly n} {N : Int} {o : Nat}
    {inp out : Tup (ExactMatrix q n 1 m) 6} (h : Placement.Spec N o inp out)
    {t : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (hin : EncRel sm inp t B) {e : Nat}
    (he : Int.fmod (o : Int) N = (e : Int)) (he2 : e < 2 * n) :
    EncRel sm out (fun j ↦ reducePoly q n (Xe (n := n) e) • t j) B := by
  have hr := rel_monomial_gen hn hin he2
  have hA : Enc.A out = fun a j ↦ multiplyMonomial (Enc.A inp a j) (e : Int) := by
    funext a j; fin_cases a <;> fin_cases j <;> simp [Enc.A, h _, he] <;> rfl
  have hc : Enc.c out = fun j ↦ multiplyMonomial (Enc.c inp j) (e : Int) := by
    funext j; fin_cases j <;> simp [Enc.c, h _, he] <;> rfl
  unfold EncRel
  rw [hA, hc]
  convert hr using 2

/-! ## Column rows (`Step::Sum`) -/

/-- Column block `c` (`m` entries) of a row of `C` blocks. -/
def blk {q r C m : Nat} (x : ExactMatrix q n r (C * m)) (c : Fin C) : ExactMatrix q n r m :=
  fun i k ↦ x i ⟨c.val * m + k.val, by have := c.isLt; have := k.isLt; nlinarith⟩

theorem blk_add {q r C m : Nat} (x y : ExactMatrix q n r (C * m)) (c : Fin C) :
    blk (x + y) c = blk x c + blk y c := rfl

theorem blk_zero {q r C m : Nat} (c : Fin C) : blk (0 : ExactMatrix q n r (C * m)) c = 0 := rfl

theorem blk_sum {q r C m : Nat} {ι : Type} (S : Finset ι) (f : ι → ExactMatrix q n r (C * m))
    (c : Fin C) : blk (∑ i ∈ S, f i) c = ∑ i ∈ S, blk (f i) c := by
  classical
  induction S using Finset.induction_on with
  | empty => simp [blk_zero]
  | insert a S ha ih => rw [Finset.sum_insert ha, Finset.sum_insert ha, blk_add, ih]

/-- The block `c` of `unitRow s ⊗ x` is `x` at `c = s` and `0` elsewhere. -/
theorem blk_tensor_unitRow {q C m : Nat} {s : Nat} {x : ExactMatrix q n 1 m}
    {out : ExactMatrix q n (1 * 1) (C * m)}
    (h : tensorRuns (unitRow (s : Int) : ExactMatrix q n 1 C) x out) (c : Fin C) :
    blk out c = if (c : Nat) = s then x else 0 := by
  funext i k
  have hi : i = 0 := Fin.ext (by have := i.isLt; omega)
  subst hi
  have hk := k.isLt
  have e := tensorRuns_unitRow h ⟨c.val * m + k.val, by have := c.isLt; nlinarith⟩
  simp only [blk]
  rw [e]
  have hm : 0 < m := by omega
  have h1 : (c.val * m + k.val) / m = c.val := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hm, Nat.div_eq_of_lt hk, Nat.zero_add]
  have h2 : (c.val * m + k.val) % m = k.val := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hk]
  simp only [h1]
  split_ifs with hc
  · congr 1; exact Fin.ext h2
  · rfl

/-- **A row of columns** (`Step::Sum` at the root): the initial state places column `0` (the
member total `col0`), iteration `i` places column `1 + i` (`ColStep.Spec`): block `c` of every
component is the sum of the `W` members of column `c`. -/
theorem column_total {q m C F W k : Nat} (hC : 0 < C) {fams : Tup (Fin F → ExactMatrix q n 1 m) k}
    {init final : Tup (ExactMatrix q n (1 * 1) (C * m)) k}
    {x0 : Tup (ExactMatrix q n 1 m) k}
    (hx0 : ∀ c, x0.get c = ∑ j ∈ Finset.range W, famAt (fams.get c) (W * 0 + j))
    (hinit : ∀ c, tensorRuns (unitRow ((0 : Nat) : Int) : ExactMatrix q n 1 C) (x0.get c) (init.get c))
    {body : Nat → Tup (ExactMatrix q n (1 * 1) (C * m)) k → Tup (ExactMatrix q n (1 * 1) (C * m)) k → Prop}
    (h : MxxIR.IterRuns body (C - 1) init final)
    (hstep : ∀ i cur next, body i cur next → ColStep.Spec fams W i cur next) :
    ∀ c (cc : Fin C), blk (final.get c) cc = ∑ j ∈ Finset.range W, famAt (fams.get c) (W * cc + j) := by
  have inv := MxxIR.IterRuns.invariant
    (Invariant := fun s (st : Tup (ExactMatrix q n (1 * 1) (C * m)) k) ↦ ∀ c (cc : Fin C),
      blk (st.get c) cc = if (cc : Nat) ≤ s then ∑ j ∈ Finset.range W, famAt (fams.get c) (W * cc + j) else 0)
    ?_ ?_ h
  · intro c cc
    rw [inv c cc, if_pos (by omega)]
  · intro c cc
    rw [blk_tensor_unitRow (hinit c) cc, hx0 c]
    by_cases h0 : (cc : Nat) = 0
    · rw [if_pos h0, if_pos (by omega), h0]
    · rw [if_neg h0, if_neg (by omega)]
  · intro i cur next hinv hb c cc
    obtain ⟨-, -, hc⟩ := hstep i cur next hb
    obtain ⟨t, ht, hn⟩ := hc c
    rw [hn, blk_add, hinv c cc]
    have e := blk_tensor_unitRow (s := 1 + i) (by exact_mod_cast ht) cc
    rw [e]
    by_cases h1 : (cc : Nat) = 1 + i
    · rw [if_neg (by omega), if_pos h1, if_pos (by omega), zero_add, h1]
    · rw [if_neg h1, add_zero]
      by_cases h2 : (cc : Nat) ≤ i
      · rw [if_pos h2, if_pos (by omega)]
      · rw [if_neg h2, if_neg (by omega)]

/-- Componentwise column block of a tuple. -/
def blkT {q r C m : Nat} (x : Tup (ExactMatrix q n r (C * m)) 6) (c : Fin C) : Tup (ExactMatrix q n r m) 6 :=
  (blk (x.get 0) c, blk (x.get 1) c, blk (x.get 2) c, blk (x.get 3) c, blk (x.get 4) c, blk (x.get 5) c, ())

theorem blkT_get {q r C m : Nat} (x : Tup (ExactMatrix q n r (C * m)) 6) (c : Fin C) (k : Fin 6) :
    (blkT x c).get k = blk (x.get k) c := by
  fin_cases k <;> rfl

/-- The encoding relation of the column blocks of a `Step::Sum` row. -/
theorem column_total_rel {q m C F W : Nat} (hC : 0 < C) (hF : W * C ≤ F) {sm : ErrorPoly n}
    {fams : Tup (Fin F → ExactMatrix q n 1 m) 6}
    {init final : Tup (ExactMatrix q n (1 * 1) (C * m)) 6} {x0 : Tup (ExactMatrix q n 1 m) 6}
    (hx0 : ∀ c, x0.get c = ∑ j ∈ Finset.range W, famAt (fams.get c) (W * 0 + j))
    (hinit : ∀ c, tensorRuns (unitRow ((0 : Nat) : Int) : ExactMatrix q n 1 C) (x0.get c) (init.get c))
    {body : Nat → Tup (ExactMatrix q n (1 * 1) (C * m)) 6 → Tup (ExactMatrix q n (1 * 1) (C * m)) 6 → Prop}
    (h : MxxIR.IterRuns body (C - 1) init final)
    (hstep : ∀ i cur next, body i cur next → ColStep.Spec fams W i cur next)
    {t : Fin F → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hmem : ∀ f, EncRel sm (Tup.ev fams f) (t f) B) (cc : Fin C) :
    EncRel sm (blkT final cc)
      (∑ j : Fin W, t ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩) (W * B) := by
  have hcol := column_total hC hx0 hinit h hstep
  have hg : ∀ c, (blkT final cc).get c =
      ∑ j : Fin W, (Tup.ev fams ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩).get c := by
    intro c
    rw [blkT_get, hcol c cc, Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    rw [Tup.get_ev, famAt_of_lt _ (by have := cc.isLt; have := j.isLt; nlinarith)]
  let g : Fin W → Fin F := fun j ↦ ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩
  have hsum := Rel.sum (Finset.univ : Finset (Fin W)) (A := fun j ↦ Enc.A (Tup.ev fams (g j)))
    (t := fun j ↦ t (g j)) (c := fun j ↦ Enc.c (Tup.ev fams (g j))) (B := fun _ ↦ B)
    (fun j _ ↦ hmem (g j))
  unfold EncRel
  rw [enc_sum_A _ _ hg, enc_sum_c _ _ hg]
  exact hsum.mono (le_of_eq (by simp))

/-- **Member prefix total**: the static reads of members `0`, `1` and the member loop over
`2 + i`, `i < Wm - 2` sum the first `Wm` members (as in column `0` of a `Step::Sum` row). -/
theorem member_prefix {α : Type} [AddCommMonoid α] {F k Wm : Nat} (hW : 2 ≤ Wm) (hF : Wm ≤ F)
    {fams : Tup (Fin F → α) k} {a b init final : Tup α k} {body : Nat → Tup α k → Tup α k → Prop}
    (ha : ∀ c, familyGetStatic (fams.get c) 0 (a.get c))
    (hb : ∀ c, familyGetStatic (fams.get c) 1 (b.get c))
    (hinit : ∀ c, init.get c = a.get c + b.get c)
    (h : MxxIR.IterRuns body (Wm - 2) init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep.Spec fams (2 + (i : Int)) cur next) :
    ∀ c, final.get c = ∑ j ∈ Finset.range Wm, famAt (fams.get c) (Wm * 0 + j) := by
  have hs := MemberSum.sem (idx := fun i ↦ 2 + (i : Int)) h hstep
  intro c
  rw [hs.2 c, hinit c]
  obtain ⟨_, ea⟩ := familyGet_eq (k := 0) (ha c)
  obtain ⟨_, eb⟩ := familyGet_eq (k := 1) (hb c)
  rw [ea, eb]
  obtain ⟨W', rfl⟩ : ∃ W', Wm = W' + 2 := ⟨Wm - 2, by omega⟩
  rw [Finset.sum_range_succ', Finset.sum_range_succ', Nat.add_sub_cancel, add_assoc, add_comm]
  simp only [Nat.mul_zero, Nat.zero_add]
  rw [famAt_of_lt _ (by omega), famAt_of_lt _ (by omega)]
  have e : ∀ j ∈ Finset.range W', famAt (fams.get c) (2 + ((j : Nat) : Int)).toNat =
      famAt (fams.get c) (j + 1 + 1) := fun j _ ↦ by
    rw [show (2 + ((j : Nat) : Int)).toNat = j + 1 + 1 by omega]
  rw [Finset.sum_congr rfl e, add_comm (fams.get c ⟨1, _⟩)]

/-- A one-row slice at column `m c` of width `m` is the block `c`. -/
theorem slice_blk {q C m : Nat} {x : ExactMatrix q n 1 (C * m)} {s e : Int} {y : ExactMatrix q n 1 m}
    (c : Fin C) (h : sliceMatrix x 0 1 s e y) (hs : s = (m : Int) * (c : Nat)) : y = blk x c := by
  obtain ⟨-, -, -, -, -, -, -, hm, hentry⟩ := h
  funext r l
  have hr : r = 0 := Fin.ext (by have := r.isLt; omega)
  subst hr
  have hl := l.isLt
  have hc := c.isLt
  subst hs
  have e1 : ((m : Int) * ((c : Nat) : Int)).toNat = m * c := by
    rw [show ((m : Int) * ((c : Nat) : Int)) = ((m * c : Nat) : Int) by push_cast; ring]; rfl
  rw [hentry 0 l (by omega) (by rw [e1]; nlinarith)]
  simp only [blk]
  congr 1
  apply Fin.ext
  show ((m : Int) * ((c : Nat) : Int)).toNat + l.val = c.val * m + l.val
  rw [e1, Nat.mul_comm]

/-- **The slice column loop** (`sum.columns.slice-step` at the root): block `c` of the row is
block `c` of `x` times `v`. -/
theorem slice_column_total {q C m : Nat} (hC : 0 < C) {x : ExactMatrix q n 1 (C * m)}
    {v : ExactMatrix q n m 1} {b0 : ExactMatrix q n 1 m} {s0 e0 : Int}
    (hb0 : sliceMatrix x 0 1 s0 e0 b0) (hs0 : s0 = 0)
    {init final : ExactMatrix q n (1 * 1) (C * 1)}
    (hinit : tensorRuns (unitRow ((0 : Nat) : Int) : ExactMatrix q n 1 C) (matrixMul b0 v) init)
    {body : Nat → ExactMatrix q n (1 * 1) (C * 1) → ExactMatrix q n (1 * 1) (C * 1) → Prop}
    (h : MxxIR.IterRuns body (C - 1) init final)
    (hstep : ∀ i cur next, body i cur next → ColSliceStep.Spec x v i cur next) :
    ∀ cc : Fin C, blk final cc = matrixMul (blk x cc) v := by
  have inv := MxxIR.IterRuns.invariant
    (Invariant := fun s (st : ExactMatrix q n (1 * 1) (C * 1)) ↦ ∀ cc : Fin C,
      blk st cc = if (cc : Nat) ≤ s then matrixMul (blk x cc) v else 0) ?_ ?_ h
  · intro cc
    rw [inv cc, if_pos (by omega)]
  · intro cc
    rw [blk_tensor_unitRow hinit cc]
    have e0 : b0 = blk x ⟨0, hC⟩ := slice_blk ⟨0, hC⟩ hb0 (by simp [hs0])
    by_cases h0 : (cc : Nat) = 0
    · rw [if_pos h0, if_pos (by omega), e0]
      congr 2
      exact Fin.ext h0.symm
    · rw [if_neg h0, if_neg (by omega)]
  · intro i cur next hinv hb cc
    obtain ⟨hlt, b, t, hs, ht, hn⟩ := hstep i cur next hb
    rw [hn, blk_add, hinv cc]
    have e := blk_tensor_unitRow (s := 1 + i) (by exact_mod_cast ht) cc
    rw [e]
    by_cases h1 : (cc : Nat) = 1 + i
    · have eb : b = blk x cc := slice_blk cc hs (by rw [h1]; push_cast; ring)
      rw [if_neg (by omega), if_pos h1, if_pos (by omega), zero_add, eb]
    · rw [if_neg h1, add_zero]
      by_cases h2 : (cc : Nat) ≤ i
      · rw [if_pos h2, if_pos (by omega)]
      · rw [if_neg h2, if_neg (by omega)]

/-- An encoding relation of a tuple whose entries are those of a relation's keys and rows. -/
theorem encRel_of_rel {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B) {x : Tup (ExactMatrix q n 1 m) 6}
    (hA : ∀ a j, Enc.A x a j = A a j) (hc : ∀ j, Enc.c x j = c j) : EncRel sm x t B := by
  have eA : Enc.A x = A := funext fun a ↦ funext fun j ↦ hA a j
  have ec : Enc.c x = c := funext hc
  unfold EncRel
  rw [eA, ec]
  exact h

/-- `Int.fmod i N` of a member index. -/
theorem fmod_nat (i N : Nat) : Int.fmod ((i : Nat) : Int) (N : Int) = ((i % N : Nat) : Int) := by
  have := Gen.fmod_ofNat i N
  simpa using this

/-- `bucket_of_cells` for cell facts indexed by the cell (bucket `≥ 1` cells name their member
`⌊i / W⌋`). -/
theorem bucket_of_cells1 {q m M W K p : Nat} {sm sp : ErrorPoly n} {hT : ExactMatrix q n 1 m}
    (β : Fin K → ErrorPoly n) (μ : Fin M → ErrorPoly n) (t : Fin (M * W) → Int)
    (s : Fin (M * W) → ErrorPoly n) {fams : Tup (Fin (M * W) → ExactMatrix q n 1 m) 6} {Bc : Nat}
    (hcell : ∀ i : Fin (M * W), ∃ k : Fin K, ((k : Nat) : Int) = t i ∧
      ∃ o : Fin M, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) W ∧
      EncRel sm (Tup.ev fams i)
        (fun j ↦ reducePoly q n ((p : Int) * (β k * μ o * s i) *
          unitPair sp j) • hT) Bc)
    (o : Fin M) {out : Tup (ExactMatrix q n 1 m) 6} (hrow : RowSum.Spec fams W o out)
    (sel : Fin W) (hsel : ∀ j, s (cellAt o j) = if j = sel then (1 : ErrorPoly n) else 0)
    (key : Fin K) (hkey : ((key : Nat) : Int) = t (cellAt o sel)) :
    EncRel sm out (fun i ↦ reducePoly q n ((p : Int) * (β key * μ o) * unitPair sp i) • hT)
      (W * Bc) :=
  bucket_of_cells β μ t s (cells_at_member (P := fun i k o ↦ EncRel sm (Tup.ev fams i)
    (fun j ↦ reducePoly q n ((p : Int) * (β k * μ o * s i) *
      unitPair sp j) • hT) Bc) hcell) o hrow sel hsel key hkey

/-- A scaled target `μ • (p ν (s, 1)_j G)` is the target of `p (μ ν)`. -/
theorem rel_scale_target {q m : Nat} {sm sp μ ν : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} {p : Int} {G : ExactMatrix q n 1 m}
    (h : Rel sm A (fun j ↦ reducePoly q n μ • (reducePoly q n ((p : ErrorPoly n) * ν * unitPair sp j) • G)) c B) :
    Rel sm A (fun j ↦ reducePoly q n ((p : ErrorPoly n) * (μ * ν) * unitPair sp j) • G) c B := by
  refine h.target_eq ?_
  funext j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

end RefreshTopProof.GenInst
