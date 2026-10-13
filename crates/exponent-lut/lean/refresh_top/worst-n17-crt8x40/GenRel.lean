import MxxIR
import ExponentLut
import ExponentPublic
import GenPbc

/-!
# The PRF cells and buckets on the encoding relation, for any parameter set

The encoding-level counterpart of `GenPbc`: nothing here names the ring dimension, the label
length `ν`, the cells per member `W`, the members `M`, the buckets `B`, a modulus or a gadget
width.

* `gather_*`: a Gather read of the export (`0 ≤ idx`, `idx < count`,
  `familyGetDynamic f idx out`) is the member `f (tab i)` of the family at the table's entry; the
  bounds are implied by the read itself.
* `rel_monomial_gen`: the placement `X^k` of an encoding.
* `cellBound`, `cell_step_gen`: one PRF cell. It reads label member `k` (the table entry of the
  cell) from the six label families and the payload family, multiplies it publicly into the
  member's state at `q`, multiplies the product privately by the cell's selector bit and moves it
  down by `p` (`q = p q'`).
* `bucket_member_gen`: the `W` cells of member `o` in one bucket sum to an encoding of the label
  entry at the key's coordinate (one-hot selectors consistent with the table).
* `chainBound`, `pbc_chain_gen`, `pbc_chain_Xe`: `B` buckets, each at its own modulus `q b`
  and gadget width `m b`, accumulate `x^(Σ_b a (key_b o))`; the bound is the explicit recursion
  `chainBound`.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## Gather reads -/

/-- A family read at any index: the index is a valid position. -/
theorem gather_toNat {α : Type} {count : Nat} {f : Fin count → α} {idx : Int} {out : α}
    (h : familyGetDynamic f idx out) : ∃ hk : idx.toNat < count, out = f ⟨idx.toNat, hk⟩ := by
  obtain ⟨pos, hpos, hout⟩ := h
  have hv : pos.val = idx.toNat := by omega
  exact ⟨hv ▸ pos.isLt, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- The exported Gather read `0 ≤ idx ∧ idx < count ∧ familyGetDynamic f idx out`. -/
theorem gather_eq {α : Type} {count : Nat} {f : Fin count → α} {idx : Int} {out : α}
    (h0 : 0 ≤ idx) (h1 : idx < count) (h : familyGetDynamic f idx out) :
    out = f ⟨idx.toNat, by omega⟩ := by
  obtain ⟨_, hout⟩ := gather_toNat h
  exact hout

/-- A read at the table entry `tab i`: `a.at(table f)`. -/
theorem gather_table {α : Type} {count F : Nat} {f : Fin count → α} (tab : Fin F → Fin count)
    (i : Fin F) {idx : Int} {out : α} (hidx : idx = ((tab i : Nat) : Int))
    (h : familyGetDynamic f idx out) : out = f (tab i) := by
  obtain ⟨pos, hpos, hout⟩ := h
  rw [hout]
  exact congrArg f (Fin.ext (by omega))

/-- A read whose index is a constant table `selectInt s table` agreeing with `tab i`. -/
theorem gather_selectInt {α : Type} {count F : Nat} {f : Fin count → α} (tab : Fin F → Fin count)
    {table : List Int} (i : Fin F) (htab : MxxIR.selectInt (Int.ofNat i) table = ((tab i : Nat) : Int))
    {out : α} (h : familyGetDynamic f (MxxIR.selectInt (Int.ofNat i) table) out) : out = f (tab i) :=
  gather_table tab i htab h

/-- A read of cell `W o + j` of a family of `M W` cells. -/
theorem gather_cellAt {α : Type} {M W : Nat} {f : Fin (M * W) → α} (o : Fin M) (j : Fin W)
    {out : α} (h : familyGetDynamic f ((W : Int) * Int.ofNat o + Int.ofNat j) out) :
    out = f (cellAt o j) := by
  refine gather_table (fun _ : Fin 1 ↦ cellAt o j) 0 ?_ h
  simp only [cellAt, Int.ofNat_eq_coe]
  push_cast
  ring

/-- A read at the shifted index `c + i` (a loop over the members `c, c + 1, …`). -/
theorem gather_add {α : Type} {count : Nat} {f : Fin count → α} (c i : Nat) {out : α}
    (h : familyGetDynamic f ((c : Int) + Int.ofNat i) out) :
    ∃ hk : c + i < count, out = f ⟨c + i, hk⟩ := by
  obtain ⟨hk, hout⟩ := gather_toNat h
  have hv : ((c : Int) + Int.ofNat i).toNat = c + i := by simp only [Int.ofNat_eq_coe]; omega
  exact ⟨hv ▸ hk, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- **A gather loop of the export** over the six label families (`∀ i : Fin F, ∃ x₀ … x₅,
(0 ≤ t i ∧ t i < K ∧ familyGetDynamic f_k (t i) x_k)_k ∧ child`): every iteration's child runs on
member `k i` of every family, with `k i = t i` the table entry. -/
theorem gather_loop6 {α : Type} {F K : Nat} {f0 f1 f2 f3 f4 f5 : Fin K → α} {t : Fin F → Int}
    (P : Fin F → Int → α → α → α → α → α → α → Prop)
    (h : ∀ i : Fin F, ∃ x0 x1 x2 x3 x4 x5 : α,
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f0 (t i) x0 ∧
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f1 (t i) x1 ∧
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f2 (t i) x2 ∧
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f3 (t i) x3 ∧
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f4 (t i) x4 ∧
      0 ≤ t i ∧ t i < K ∧ familyGetDynamic f5 (t i) x5 ∧
      P i (t i) x0 x1 x2 x3 x4 x5) (i : Fin F) :
    ∃ k : Fin K, ((k : Nat) : Int) = t i ∧ P i (t i) (f0 k) (f1 k) (f2 k) (f3 k) (f4 k) (f5 k) := by
  obtain ⟨x0, x1, x2, x3, x4, x5, -, -, r0, -, -, r1, -, -, r2, -, -, r3, -, -, r4, -, -, r5, hP⟩ :=
    h i
  obtain ⟨k, hk, rfl⟩ := r0
  have e : ∀ {g : Fin K → α} {x : α}, familyGetDynamic g (t i) x → x = g k := by
    intro g x hx
    obtain ⟨k', hk', rfl⟩ := hx
    exact congrArg g (Fin.ext (by omega))
  rw [e r1, e r2, e r3, e r4, e r5] at hP
  exact ⟨k, hk, hP⟩

/-- The table of a gather loop as `Fin K`-valued, given the loop (`gather_loop6`). -/
theorem gather_table_fin {F K : Nat} {t : Fin F → Int} (h : ∀ i, ∃ k : Fin K, ((k : Nat) : Int) = t i) :
    ∃ tab : Fin F → Fin K, ∀ i, ((tab i : Nat) : Int) = t i :=
  ⟨fun i ↦ (h i).choose, fun i ↦ (h i).choose_spec⟩

/-! ## Monomial placement -/

theorem multiplyMonomial_eq_gen {q rows columns : Nat} (x : ExactMatrix q n rows columns) {k : Nat}
    (hk : k < 2 * n) : multiplyMonomial x (k : Int) = reducePoly q n (Xe (n := n) k) • x := by
  funext r c
  simp only [multiplyMonomial, Matrix.smul_apply, smul_eq_mul]
  have hmod : ((k : Int) % (2 * (n : Int))).toNat = k := by
    have hk' : (k : Int) < 2 * (n : Int) := by exact_mod_cast hk
    rw [Int.emod_eq_of_lt (by omega) hk']
    simp
  rw [hmod, reducePoly_Xe, mul_comm]

/-- The placement `X^k` of an encoding multiplies its target by `X^k` and keeps its bound. -/
theorem rel_monomial_gen (hn : 1 < n) {q m : Nat} {sm : ErrorPoly n}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A t c B) {k : Nat} (hk : k < 2 * n) :
    Rel sm (fun a j ↦ multiplyMonomial (A a j) (k : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) k) • t j) (fun j ↦ multiplyMonomial (c j) (k : Int)) B := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨fun j ↦ Xe (n := n) k • e j, fun j ↦ ?_, fun j ↦ ?_⟩
  · simpa using smulGain_root_pow hn k (he j)
  · simp only [multiplyMonomial_eq_gen _ hk, hc j]
    funext r col
    simp only [Matrix.smul_apply, Matrix.add_apply, Matrix.sub_apply, smul_eq_mul, reduceMatrix,
      map_mul]
    ring

/-! ## Member sums of the export -/

/-- A loop whose iteration `i` adds `f i` sums `f` over its iterations (the member sums). -/
theorem iter_sum_gen {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {init final : α} (f : Nat → α)
    (hstep : ∀ i current next, body i current next → next = current + f i)
    (h : MxxIR.IterRuns body count init final) : final = init + ∑ i ∈ Finset.range count, f i :=
  MxxIR.IterRuns.invariant (Invariant := fun k (s : α) ↦ s = init + ∑ i ∈ Finset.range k, f i)
    (by simp) (fun i current next hcur hbody ↦ by
      show next = init + ∑ j ∈ Finset.range (i + 1), f j
      rw [hstep i current next hbody, hcur, Finset.sum_range_succ, add_assoc]) h

/-- The cells `0` and `1` and the loop's cells `2, …, w + 1` of a member sum. -/
theorem sum_cells_gen {α : Type} [AddCommMonoid α] {w : Nat} (f : Fin (w + 2) → α) :
    f 0 + f 1 + ∑ i ∈ Finset.range w, (if h : 2 + i < w + 2 then f ⟨2 + i, h⟩ else 0) =
      ∑ k : Fin (w + 2), f k := by
  rw [Fin.sum_univ_succ, Fin.sum_univ_succ, ← add_assoc]
  congr 1
  rw [← Fin.sum_univ_eq_sum_range]
  apply Finset.sum_congr rfl
  intro i _
  rw [dif_pos (by omega)]
  congr 1
  ext
  simp [Fin.succ]
  omega

/-- **A row sum of the export** (`par[M]` member sums): the member's `w + 2` cells `U k` are
unpacked, the first two added, and a loop of `w` iterations adds cell `2 + i` in iteration `i`;
the result is the sum of all cells. Any state type (the six blocks of an encoding as a tuple). -/
theorem rowsum_loop_gen {S : Type} [AddCommMonoid S] {w : Nat} (U : Fin (w + 2) → S)
    {body : Nat → S → S → Prop} {final : S}
    (hstep : ∀ i cur next, (hi : i < w) → body i cur next → next = cur + U ⟨2 + i, by omega⟩)
    (h : MxxIR.IterRuns body w (U 0 + U 1) final) : final = ∑ k, U k := by
  rw [← sum_cells_gen U]
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : S) ↦ k ≤ w → s = U 0 + U 1 +
      ∑ i ∈ Finset.range k, (if h : 2 + i < w + 2 then U ⟨2 + i, h⟩ else 0)) ?_ ?_ h le_rfl
  · intro _
    simp
  · intro i cur next hcur hbody hi
    rw [hstep i cur next (by omega) hbody, hcur (by omega), Finset.sum_range_succ,
      dif_pos (by omega), add_assoc]

/-- The sum of a family of six-block tuples is the tuple of the sums. -/
theorem blocks6_sum {ι : Type} {A0 A1 A2 A3 A4 A5 : Type} [AddCommMonoid A0] [AddCommMonoid A1]
    [AddCommMonoid A2] [AddCommMonoid A3] [AddCommMonoid A4] [AddCommMonoid A5] (s : Finset ι)
    (a0 : ι → A0) (a1 : ι → A1) (a2 : ι → A2) (a3 : ι → A3) (a4 : ι → A4) (a5 : ι → A5) :
    (∑ k ∈ s, ((a0 k, a1 k, a2 k, a3 k, a4 k, a5 k, ()) : A0 × A1 × A2 × A3 × A4 × A5 × Unit)) =
      (∑ k ∈ s, a0 k, ∑ k ∈ s, a1 k, ∑ k ∈ s, a2 k, ∑ k ∈ s, a3 k, ∑ k ∈ s, a4 k, ∑ k ∈ s, a5 k,
        ()) := by
  classical
  induction s using Finset.induction_on with
  | empty => rfl
  | @insert i s hi ih => simp only [Finset.sum_insert hi, ih]; rfl

/-! ## One PRF cell -/

/-- The bound of a PRF cell whose label payload has smul gain `K` (`SmulGain α K`; `K = 1` for a
monomial `X^k`, `n` for a payload of coefficient bound `1`): the label (bound `Ba`, gadget width `m`) acts publicly on the state
(bound `Bs`, width `m'`) with decomposition bound `bound`; the product is multiplied privately by
the selector (switching error `BB`) and moved down by `p`. -/
abbrev cellBoundK (n m m' bound p Ba Bs BB K : Nat) : Nat :=
  ((m' + m') * n * ((m + m) * n * Ba * bound + K * Bs) * bound + (n + 2) * (p / 2)) / p +
    n * 1 * BB

/-- The bound of a PRF cell (a label payload of coefficient bound `1`, smul gain `n`). -/
abbrev cellBound (n m m' bound p Ba Bs BB : Nat) : Nat :=
  ((m' + m') * n * ((m + m) * n * Ba * bound + n * 1 * Bs) * bound + (n + 2) * (p / 2)) / p +
    n * 1 * BB

set_option maxHeartbeats 4000000 in
/-- **The multiplications of a PRF cell.** A label encoding of `α` (gadget `G` of width `m`, its
payload `α`) acts publicly on the member's state (an encoding of `p μ` at `q`, gadget `g`), and the
product is multiplied privately by the selector `ν` (the switching ciphertext `top`/`bot`,
`hrhs`) and moved down to `q'`: an encoding of `p' (α μ ν)` with gadget `h`. -/
theorem cell_mul_gen (hn : 0 < n) {q q' p p' m m' m'' bound : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1} {Ba : Nat}
    (hα : polyNorm α ≤ 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : xpay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ * ν) * unitPair sp j) • h) c''
      (cellBound n m m' bound p Ba Bs BB) := by
  -- The public multiplication at `q`.
  have hpub := mul_public (m := m) (m' := m') (bound := bound) hn hq1 layout hlayout hexact hG
    hα ha hs hl01 hl hr01 hr hboth hXs hD' hD hpay hY hA0 hA1 (c' := c') (fun j ↦ hc' j)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly q n ((p : Int) * (α * μ) *
      unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, moving down by `p`.
  have hpriv := mul_private_down (m := m') (m' := m'') (bound := bound) hn hq hq1 hq' hp layout'
    hlayout' hexact' hg' hg hsm hpub' hl01' hl' hr01' hr' hboth' hXb hE' hE hrhs heB hαμ
    hX hM0 hM1 hc'' hA''
  refine hpriv.target_eq ?_
  funext j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

set_option maxHeartbeats 4000000 in
/-- **One PRF cell.** The cell reads member `k` (its table entry, `idx = k`) of the six label
families and of the payload family, multiplies the label publicly into the member's state (an
encoding of `p μ` at `q`, gadget `g`), multiplies the product privately by the selector `ν` (the
switching ciphertext `top`/`bot`, `hrhs`) and moves it down to `q'`: an encoding of
`p' (α_k μ ν)` with gadget `h`. -/
theorem cell_step_gen (hn : 0 < n) {q q' p p' m m' m'' bound K : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin K → ExactMatrix q n 1 m}
    {apay : Fin K → ExactMatrix q n 1 1} {α : Fin K → ErrorPoly n} {Ba : Nat}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    (ha : ∀ k, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly q n (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k, apay k = fun _ _ ↦ reducePoly q n (α k))
    (k : Fin K) {idx : Int} (hidx : idx = ((k : Nat) : Int))
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1}
    (r0 : familyGetDynamic ca0 idx x0) (r1 : familyGetDynamic ca1 idx x1)
    (r2 : familyGetDynamic aA00 idx xA00) (r3 : familyGetDynamic aA10 idx xA10)
    (r4 : familyGetDynamic aA01 idx xA01) (r5 : familyGetDynamic aA11 idx xA11)
    (r6 : familyGetDynamic apay idx xpay)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α k * μ * ν) * unitPair sp j) • h) c''
      (cellBound n m m' bound p Ba Bs BB) := by
  -- The Gather reads.
  have e0 := gather_table id k hidx r0
  have e1 := gather_table id k hidx r1
  have e2 := gather_table id k hidx r2
  have e3 := gather_table id k hidx r3
  have e4 := gather_table id k hidx r4
  have e5 := gather_table id k hidx r5
  have e6 := gather_table id k hidx r6
  simp only [id] at e0 e1 e2 e3 e4 e5 e6
  subst e0 e1 e2 e3 e4 e5 e6
  exact cell_mul_gen hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG (hα k)
    (hαμ k) (ha k) (hpay k) hg' hg hs hl01 hl hr01 hr hboth hXs hD' hD hY hA0 hA1 hc' hl01' hl'
    hr01' hr' hboth' hXb hE' hE hrhs heB hX hM0 hM1 hc'' hA''

set_option maxHeartbeats 4000000 in
/-- **One PRF cell with an aligned label** (the buckets after the first). The cell reads member
`k` of the label families at `Q = pA q` (gadget `GA` of width `mA`), aligns it to `q` (the
decomposition `HA` of `pA GR`, `GR` the rebased gadget `G` of `q`, and the move down by `pA`),
reduces the payload to `q`, and runs `cell_mul_gen`. The label bound becomes
`(mA n Ba (baseA / 2) + (n + 2) (pA / 2)) / pA`. -/
theorem cell_step_aligned_gen (hn : 0 < n) {Q q q' pA p p' mA m m' m'' bound K : Nat}
    (hQ : Q = pA * q) (hQ1 : 1 < Q) (hpA : 0 < pA) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {baseA digitsA base digits base' digits' : Int}
    (layoutA : RegularLayout Q) (hlayoutA : backend.regularLayout Q n = some layoutA)
    (hexactA : layoutA.droppedModuli = 0)
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {GA : ExactMatrix Q n 1 mA} (hGA : gadgetMatrixRuns backend baseA digitsA GA)
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin K → ExactMatrix Q n 1 mA}
    {apay : Fin K → ExactMatrix Q n 1 1} {α : Fin K → ErrorPoly n} {Ba : Nat}
    (hα : ∀ k, polyNorm (α k) ≤ 1) (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    (ha : ∀ k, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q n (α k * unitPair sm j) • GA) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k, apay k = fun _ _ ↦ reducePoly Q n (α k))
    (k : Fin K) {idx : Int} (hidx : idx = ((k : Nat) : Int))
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix Q n 1 mA} {xpay : ExactMatrix Q n 1 1}
    (r0 : familyGetDynamic ca0 idx x0) (r1 : familyGetDynamic ca1 idx x1)
    (r2 : familyGetDynamic aA00 idx xA00) (r3 : familyGetDynamic aA10 idx xA10)
    (r4 : familyGetDynamic aA01 idx xA01) (r5 : familyGetDynamic aA11 idx xA11)
    (r6 : familyGetDynamic apay idx xpay)
    -- the alignment to `q`
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    {scaled : ExactMatrix Q n 1 m} (hscaled : scaled = reducePoly Q n ((pA : Int) : ErrorPoly n) • GR)
    {HA : ExactMatrix Q n mA m} (hHA : gadgetDecomposeRuns backend baseA digitsA scaled HA)
    {basisA : List Nat} {y0 y1 yA00 yA10 yA01 yA11 : ExactMatrix q n 1 m}
    (hy0 : blockModSwitchRuns basisA 1 (matrixMul x0 HA) y0)
    (hy1 : blockModSwitchRuns basisA 1 (matrixMul x1 HA) y1)
    (hyA00 : blockModSwitchRuns basisA 1 (matrixMul xA00 HA) yA00)
    (hyA10 : blockModSwitchRuns basisA 1 (matrixMul xA10 HA) yA10)
    (hyA01 : blockModSwitchRuns basisA 1 (matrixMul xA01 HA) yA01)
    (hyA11 : blockModSwitchRuns basisA 1 (matrixMul xA11 HA) yA11)
    {ypay : ExactMatrix q n 1 1} (hypay : modulusReduceRuns xpay ypay)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows y0 yA00 l01) (hl : concatRows l01 yA10 l)
    (hr01 : concatRows y1 yA01 r01) (hr : concatRows r01 yA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) ypay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α k * μ * ν) * unitPair sp j) • h) c''
      (cellBound n m m' bound p ((mA * n * Ba * (baseA.toNat / 2) + (n + 2) * (pA / 2)) / pA)
        Bs BB) := by
  -- The Gather reads.
  have e0 := gather_table id k hidx r0
  have e1 := gather_table id k hidx r1
  have e2 := gather_table id k hidx r2
  have e3 := gather_table id k hidx r3
  have e4 := gather_table id k hidx r4
  have e5 := gather_table id k hidx r5
  have e6 := gather_table id k hidx r6
  simp only [id] at e0 e1 e2 e3 e4 e5 e6
  subst e0 e1 e2 e3 e4 e5 e6
  -- The alignment of the label from `Q` to `q`.
  have hal := align_down (q := Q) (q' := q) (p := pA) (m := mA) (m' := m) hn hQ hQ1 hq1 hpA
    layoutA hlayoutA hexactA hGA hGR hsm (ha k) hscaled hHA
    (Y := fun j ↦ matrixMul (![ca0 k, ca1 k] j) HA)
    (K := fun a j ↦ matrixMul (![![aA00 k, aA01 k], ![aA10 k, aA11 k]] a j) HA)
    (fun j ↦ rfl) (fun a j ↦ rfl) (c' := ![y0, y1]) (A' := ![![yA00, yA01], ![yA10, yA11]])
    (fun j ↦ by fin_cases j; exacts [hy0, hy1])
    (fun a j ↦ by fin_cases a <;> fin_cases j; exacts [hyA00, hyA01, hyA10, hyA11])
  -- The payload at `q`.
  have hpayq := modulusReduce_poly hQ1 hq1 hn (s := α k) (by rw [← hpay k]; exact hypay)
  exact cell_mul_gen hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG (hα k)
    (hαμ k) hal hpayq hg' hg hs hl01 hl hr01 hr hboth hXs hD' hD hY hA0 hA1 hc' hl01' hl'
    hr01' hr' hboth' hXb hE' hE hrhs heB hX hM0 hM1 hc'' hA''

set_option maxHeartbeats 4000000 in
/-- **The multiplications of a PRF cell** with a label payload of smul gain `K` (`K = 1` for a
monomial payload `X^k`, `smulGain_root_pow`; `cell_mul_gen` is `K = n`). A label encoding of `α` (gadget `G` of width `m`, its
payload `α`) acts publicly on the member's state (an encoding of `p μ` at `q`, gadget `g`), and the
product is multiplied privately by the selector `ν` (the switching ciphertext `top`/`bot`,
`hrhs`) and moved down to `q'`: an encoding of `p' (α μ ν)` with gadget `h`. -/
theorem cell_mul_genK (hn : 0 < n) {q q' p p' m m' m'' bound : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1} {Ba : Nat}
    {K : Nat} (hα : SmulGain α K) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : xpay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ * ν) * unitPair sp j) • h) c''
      (cellBoundK n m m' bound p Ba Bs BB K) := by
  -- The public multiplication at `q`.
  have hpub := mul_public_gain (m := m) (m' := m') (bound := bound) hn hq1 layout hlayout hexact hG
    hα ha hs hl01 hl hr01 hr hboth hXs hD' hD hpay hY hA0 hA1 (c' := c') (fun j ↦ hc' j)
  have hpub' := hpub.target_eq (t' := fun j ↦ reducePoly q n ((p : Int) * (α * μ) *
      unitPair sp j) • g) (by
    funext j
    rw [smul_smul, ← map_mul]
    congr 2
    ring)
  -- The private multiplication by the selector, moving down by `p`.
  have hpriv := mul_private_down (m := m') (m' := m'') (bound := bound) hn hq hq1 hq' hp layout'
    hlayout' hexact' hg' hg hsm hpub' hl01' hl' hr01' hr' hboth' hXb hE' hE hrhs heB hαμ
    hX hM0 hM1 hc'' hA''
  refine hpriv.target_eq ?_
  funext j
  rw [smul_smul, ← map_mul]
  congr 2
  ring

set_option maxHeartbeats 4000000 in
/-- **One PRF cell.** The cell reads member `k` (its table entry, `idx = k`) of the six label
families and of the payload family, multiplies the label publicly into the member's state (an
encoding of `p μ` at `q`, gadget `g`), multiplies the product privately by the selector `ν` (the
switching ciphertext `top`/`bot`, `hrhs`) and moves it down to `q'`: an encoding of
`p' (α_k μ ν)` with gadget `h`. -/
theorem cell_step_genK (hn : 0 < n) {q q' p p' m m' m'' bound K : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin K → ExactMatrix q n 1 m}
    {apay : Fin K → ExactMatrix q n 1 1} {α : Fin K → ErrorPoly n} {Ba : Nat}
    {Kα : Nat} (hα : ∀ k, SmulGain (α k) Kα) (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    (ha : ∀ k, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly q n (α k * unitPair sm j) • G) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k, apay k = fun _ _ ↦ reducePoly q n (α k))
    (k : Fin K) {idx : Int} (hidx : idx = ((k : Nat) : Int))
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix q n 1 m} {xpay : ExactMatrix q n 1 1}
    (r0 : familyGetDynamic ca0 idx x0) (r1 : familyGetDynamic ca1 idx x1)
    (r2 : familyGetDynamic aA00 idx xA00) (r3 : familyGetDynamic aA10 idx xA10)
    (r4 : familyGetDynamic aA01 idx xA01) (r5 : familyGetDynamic aA11 idx xA11)
    (r6 : familyGetDynamic apay idx xpay)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows x0 xA00 l01) (hl : concatRows l01 xA10 l)
    (hr01 : concatRows x1 xA01 r01) (hr : concatRows r01 xA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) xpay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α k * μ * ν) * unitPair sp j) • h) c''
      (cellBoundK n m m' bound p Ba Bs BB Kα) := by
  -- The Gather reads.
  have e0 := gather_table id k hidx r0
  have e1 := gather_table id k hidx r1
  have e2 := gather_table id k hidx r2
  have e3 := gather_table id k hidx r3
  have e4 := gather_table id k hidx r4
  have e5 := gather_table id k hidx r5
  have e6 := gather_table id k hidx r6
  simp only [id] at e0 e1 e2 e3 e4 e5 e6
  subst e0 e1 e2 e3 e4 e5 e6
  exact cell_mul_genK hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG (hα k)
    (hαμ k) (ha k) (hpay k) hg' hg hs hl01 hl hr01 hr hboth hXs hD' hD hY hA0 hA1 hc' hl01' hl'
    hr01' hr' hboth' hXb hE' hE hrhs heB hX hM0 hM1 hc'' hA''

set_option maxHeartbeats 4000000 in
/-- **One PRF cell with an aligned label** (the buckets after the first). The cell reads member
`k` of the label families at `Q = pA q` (gadget `GA` of width `mA`), aligns it to `q` (the
decomposition `HA` of `pA GR`, `GR` the rebased gadget `G` of `q`, and the move down by `pA`),
reduces the payload to `q`, and runs `cell_mul_gen`. The label bound becomes
`(mA n Ba (baseA / 2) + (n + 2) (pA / 2)) / pA`. -/
theorem cell_step_aligned_genK (hn : 0 < n) {Q q q' pA p p' mA m m' m'' bound K : Nat}
    (hQ : Q = pA * q) (hQ1 : 1 < Q) (hpA : 0 < pA) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {baseA digitsA base digits base' digits' : Int}
    (layoutA : RegularLayout Q) (hlayoutA : backend.regularLayout Q n = some layoutA)
    (hexactA : layoutA.droppedModuli = 0)
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0)
    {sm sp μ ν : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {GA : ExactMatrix Q n 1 mA} (hGA : gadgetMatrixRuns backend baseA digitsA GA)
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin K → ExactMatrix Q n 1 mA}
    {apay : Fin K → ExactMatrix Q n 1 1} {α : Fin K → ErrorPoly n} {Ba : Nat}
    {Kα : Nat} (hα : ∀ k, SmulGain (α k) Kα) (hαμ : ∀ k, polyNorm (α k * μ) ≤ 1)
    (ha : ∀ k, Rel sm ![![aA00 k, aA01 k], ![aA10 k, aA11 k]]
      (fun j ↦ reducePoly Q n (α k * unitPair sm j) • GA) ![ca0 k, ca1 k] Ba)
    (hpay : ∀ k, apay k = fun _ _ ↦ reducePoly Q n (α k))
    (k : Fin K) {idx : Int} (hidx : idx = ((k : Nat) : Int))
    {x0 x1 xA00 xA10 xA01 xA11 : ExactMatrix Q n 1 mA} {xpay : ExactMatrix Q n 1 1}
    (r0 : familyGetDynamic ca0 idx x0) (r1 : familyGetDynamic ca1 idx x1)
    (r2 : familyGetDynamic aA00 idx xA00) (r3 : familyGetDynamic aA10 idx xA10)
    (r4 : familyGetDynamic aA01 idx xA01) (r5 : familyGetDynamic aA11 idx xA11)
    (r6 : familyGetDynamic apay idx xpay)
    -- the alignment to `q`
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    {scaled : ExactMatrix Q n 1 m} (hscaled : scaled = reducePoly Q n ((pA : Int) : ErrorPoly n) • GR)
    {HA : ExactMatrix Q n mA m} (hHA : gadgetDecomposeRuns backend baseA digitsA scaled HA)
    {basisA : List Nat} {y0 y1 yA00 yA10 yA01 yA11 : ExactMatrix q n 1 m}
    (hy0 : blockModSwitchRuns basisA 1 (matrixMul x0 HA) y0)
    (hy1 : blockModSwitchRuns basisA 1 (matrixMul x1 HA) y1)
    (hyA00 : blockModSwitchRuns basisA 1 (matrixMul xA00 HA) yA00)
    (hyA10 : blockModSwitchRuns basisA 1 (matrixMul xA10 HA) yA10)
    (hyA01 : blockModSwitchRuns basisA 1 (matrixMul xA01 HA) yA01)
    (hyA11 : blockModSwitchRuns basisA 1 (matrixMul xA11 HA) yA11)
    {ypay : ExactMatrix q n 1 1} (hypay : modulusReduceRuns xpay ypay)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g)
    {As : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hs : Rel sm As (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) cs Bs)
    -- the public multiplication
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows y0 yA00 l01) (hl : concatRows l01 yA10 l)
    (hr01 : concatRows y1 yA01 r01) (hr : concatRows r01 yA11 r) (hboth : concatColumns l r both)
    {Xs : Fin 2 → ExactMatrix q n (1 + 1) m'} (hXs : ∀ j, concatRows (As 0 j) (As 1 j) (Xs j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xs j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {Y : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Y j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j))
    {c' : Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, c' j = Y j + matrixMulScalarRight (cs j) ypay)
    -- the private multiplication and the move down
    {l01' r01' : ExactMatrix q n (1 + 1) m'} {l' r' : ExactMatrix q n (1 + 1 + 1) m'}
    {both' : ExactMatrix q n (1 + 1 + 1) (m' + m')}
    (hl01' : concatRows (c' 0) (A' 0 0) l01') (hl' : concatRows l01' (A' 1 0) l')
    (hr01' : concatRows (c' 1) (A' 0 1) r01') (hr' : concatRows r01' (A' 1 1) r')
    (hboth' : concatColumns l' r' both')
    {top bot : Fin 2 → ExactMatrix q' n 1 m''} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m''}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {E' : Fin 2 → ExactMatrix q' n (m' + m') m''}
    (hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (Xb j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m' + m') m''} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • top j + bot j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • h + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB)
    {X : Fin 2 → ExactMatrix q n 1 m''} {Mk : Fin 2 → Fin 2 → ExactMatrix q n 1 m''}
    (hX : ∀ j, sliceMatrix (matrixMul both' (E j)) 0 1 0 m'' (X j))
    (hM0 : ∀ j, sliceMatrix (matrixMul both' (E j)) 1 2 0 m'' (Mk 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both' (E j)) 2 3 0 m'' (Mk 1 j))
    {basis : List Nat} {c'' : Fin 2 → ExactMatrix q' n 1 m''}
    {A'' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m''}
    (hc'' : ∀ j, blockModSwitchRuns basis 1 (X j) (c'' j))
    (hA'' : ∀ a j, blockModSwitchRuns basis 1 (Mk a j) (A'' a j)) :
    Rel sm A'' (fun j ↦ reducePoly q' n ((p' : Int) * (α k * μ * ν) * unitPair sp j) • h) c''
      (cellBoundK n m m' bound p ((mA * n * Ba * (baseA.toNat / 2) + (n + 2) * (pA / 2)) / pA)
        Bs BB Kα) := by
  -- The Gather reads.
  have e0 := gather_table id k hidx r0
  have e1 := gather_table id k hidx r1
  have e2 := gather_table id k hidx r2
  have e3 := gather_table id k hidx r3
  have e4 := gather_table id k hidx r4
  have e5 := gather_table id k hidx r5
  have e6 := gather_table id k hidx r6
  simp only [id] at e0 e1 e2 e3 e4 e5 e6
  subst e0 e1 e2 e3 e4 e5 e6
  -- The alignment of the label from `Q` to `q`.
  have hal := align_down (q := Q) (q' := q) (p := pA) (m := mA) (m' := m) hn hQ hQ1 hq1 hpA
    layoutA hlayoutA hexactA hGA hGR hsm (ha k) hscaled hHA
    (Y := fun j ↦ matrixMul (![ca0 k, ca1 k] j) HA)
    (K := fun a j ↦ matrixMul (![![aA00 k, aA01 k], ![aA10 k, aA11 k]] a j) HA)
    (fun j ↦ rfl) (fun a j ↦ rfl) (c' := ![y0, y1]) (A' := ![![yA00, yA01], ![yA10, yA11]])
    (fun j ↦ by fin_cases j; exacts [hy0, hy1])
    (fun a j ↦ by fin_cases a <;> fin_cases j; exacts [hyA00, hyA01, hyA10, hyA11])
  -- The payload at `q`.
  have hpayq := modulusReduce_poly hQ1 hq1 hn (s := α k) (by rw [← hpay k]; exact hypay)
  exact cell_mul_genK hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG (hα k)
    (hαμ k) hal hpayq hg' hg hs hl01 hl hr01 hr hboth hXs hD' hD hY hA0 hA1 hc' hl01' hl'
    hr01' hr' hboth' hXb hE' hE hrhs heB hX hM0 hM1 hc'' hA''

/-! ## One bucket of one member -/

/-- **One bucket of member `o`.** If cell `W o + j` encodes `p (β(tab f) μ_o s_f)` and the
selectors of the member's row are one-hot at the slot whose table entry is the key's coordinate,
the row sums to an encoding of `p (β(key o) μ_o)`, with bound `W Bc`. -/
theorem bucket_member_gen {q m M W ν p : Nat} {sm sp : ErrorPoly n} {h : ExactMatrix q n 1 m}
    (β : Fin (ν + 1) → ErrorPoly n) (μ : Fin M → ErrorPoly n) (tab : Fin (M * W) → Fin (ν + 1))
    (s : Fin (M * W) → Int) (o : Fin M) (sel : Fin W) (key : Fin (ν + 1))
    (hsel : ∀ j, s (cellAt o j) = if j = sel then 1 else 0) (htab : tab (cellAt o sel) = key)
    {f : Fin (M * W) → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin (M * W) → Fin 2 → ExactMatrix q n 1 m} {Bc : Nat}
    (hcell : ∀ j : Fin W, Rel sm (f (cellAt o j))
      (fun i ↦ reducePoly q n ((p : Int) * (β (tab (cellAt o j)) * μ o *
        ((s (cellAt o j) : Int) : ErrorPoly n)) * unitPair sp i) • h) (c (cellAt o j)) Bc) :
    Rel sm (∑ j : Fin W, f (cellAt o j))
      (fun i ↦ reducePoly q n ((p : Int) * (β key * μ o) * unitPair sp i) • h)
      (∑ j : Fin W, c (cellAt o j)) (W * Bc) := by
  have hsum := Rel.sum Finset.univ (fun j _ ↦ hcell j)
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  rw [Finset.sum_eq_single sel]
  · funext i
    simp [hsel, htab]
  · intro j _ hj
    funext i
    simp [hsel, hj]
  · simp

/-! ## The bucket chain -/

/-- The bound after `b` buckets: `B₀`, then `F b` of the previous bound. -/
def chainBound (F : Nat → Nat → Nat) (B0 : Nat) : Nat → Nat
  | 0 => B0
  | b + 1 => F b (chainBound F B0 b)

theorem chainBound_zero (F : Nat → Nat → Nat) (B0 : Nat) : chainBound F B0 0 = B0 := rfl

theorem chainBound_succ (F : Nat → Nat → Nat) (B0 b : Nat) :
    chainBound F B0 (b + 1) = F b (chainBound F B0 b) := rfl

/-- The exponent accumulated by member `o` over the first `b` of `B` buckets. -/
def pbcExp {M B ν : Nat} (a : Fin (ν + 1) → Nat) (key : Fin B → Fin M → Fin (ν + 1)) (b : Nat)
    (o : Fin M) : Nat :=
  ∑ i ∈ Finset.range b, if hi : i < B then a (key ⟨i, hi⟩ o) else 0

theorem pbcExp_zero {M B ν : Nat} (a : Fin (ν + 1) → Nat) (key : Fin B → Fin M → Fin (ν + 1))
    (o : Fin M) : pbcExp a key 0 o = 0 := by
  simp [pbcExp]

theorem pbcExp_succ {M B ν : Nat} (a : Fin (ν + 1) → Nat) (key : Fin B → Fin M → Fin (ν + 1))
    (b : Fin B) (o : Fin M) : pbcExp a key (b + 1) o = a (key b o) + pbcExp a key b o := by
  rw [pbcExp, Finset.sum_range_succ, dif_pos b.isLt, add_comm]
  rfl

theorem pbcExp_full {M B ν : Nat} (a : Fin (ν + 1) → Nat) (key : Fin B → Fin M → Fin (ν + 1))
    (o : Fin M) : pbcExp a key B o = ∑ b : Fin B, a (key b o) := by
  rw [pbcExp, Finset.sum_range]
  exact Finset.sum_congr rfl fun i _ ↦ by rw [dif_pos i.isLt]

/-- **The bucket chain.** Bucket `b` runs at modulus `q b` with gadget `h b` and scale `P b`.
Starting from encodings of `P 0` (the private input `mono(0) = 1`), if every bucket maps
encodings of `P b x^(e_b o)` (`e_b = pbcExp`, the exponent accumulated so far) with bound `Bs` to
encodings of `P (b + 1) x^(e_(b+1) o)` with bound `F b Bs`, then after `B` buckets member `o`
encodes `P B x^(Σ_b a (key_b o))` with bound `chainBound F B0 B`. -/
theorem pbc_chain_gen {M B ν : Nat} (q m P : Nat → Nat) {sm sp : ErrorPoly n}
    (h : ∀ b, ExactMatrix (q b) n 1 (m b))
    (A : ∀ b, Fin M → Fin 2 → Fin 2 → ExactMatrix (q b) n 1 (m b))
    (c : ∀ b, Fin M → Fin 2 → ExactMatrix (q b) n 1 (m b))
    (x : ErrorPoly n) (a : Fin (ν + 1) → Nat) (key : Fin B → Fin M → Fin (ν + 1))
    (F : Nat → Nat → Nat) (B0 : Nat)
    (h0 : ∀ o, Rel sm (A 0 o) (fun j ↦ reducePoly (q 0) n ((P 0 : Int) * 1 * unitPair sp j) • h 0)
      (c 0 o) B0)
    (hstep : ∀ (b : Fin B) (Bs : Nat),
      (∀ o, Rel sm (A b o) (fun j ↦ reducePoly (q b) n ((P b : Int) * x ^ pbcExp a key b o *
        unitPair sp j) • h b) (c b o) Bs) →
      ∀ o, Rel sm (A (b + 1) o) (fun j ↦ reducePoly (q (b + 1)) n ((P (b + 1) : Int) *
        x ^ pbcExp a key (b + 1) o * unitPair sp j) • h (b + 1)) (c (b + 1) o) (F b Bs))
    (o : Fin M) :
    Rel sm (A B o) (fun j ↦ reducePoly (q B) n ((P B : Int) * x ^ (∑ b : Fin B, a (key b o)) *
      unitPair sp j) • h B) (c B o) (chainBound F B0 B) := by
  have hpre : ∀ b ≤ B, ∀ o, Rel sm (A b o) (fun j ↦ reducePoly (q b) n ((P b : Int) *
      x ^ pbcExp a key b o * unitPair sp j) • h b) (c b o) (chainBound F B0 b) := by
    intro b hb
    induction b with
    | zero => intro o; simpa [chainBound, pbcExp_zero] using h0 o
    | succ b ih => exact hstep ⟨b, by omega⟩ (chainBound F B0 b) (ih (by omega))
  have hB := hpre B le_rfl o
  rw [pbcExp_full] at hB
  exact hB

/-- The step of the chain from one bucket's member relation (`bucket_member_gen` with
`β = x^a`, `μ = x^(e_b)`): `x^(a (key_b o)) x^(e_b o) = x^(e_(b+1) o)`. -/
theorem pbc_step_target {M B ν : Nat} (x : ErrorPoly n) (a : Fin (ν + 1) → Nat)
    (key : Fin B → Fin M → Fin (ν + 1)) (b : Fin B) (o : Fin M) :
    x ^ a (key b o) * x ^ pbcExp a key b o = x ^ pbcExp a key (b + 1) o := by
  rw [pbcExp_succ, pow_add]

/-- The chain in the PRF's exponent layout `x = X^s` with `s Q = 2 n`: the accumulated exponent
only matters modulo `Q`. -/
theorem pbc_chain_Xe {s Q : Nat} (hsQ : s * Q = 2 * n) {B ν : Nat} (a : Fin (ν + 1) → Nat)
    (key : Fin B → Fin (ν + 1)) :
    (Xe (n := n) s) ^ (∑ b : Fin B, a (key b)) =
      Xe (n := n) (s * ((∑ b : Fin B, a (key b)) % Q)) := by
  rw [← Xe_stride_mod hsQ]
  simp only [Xe, pow_mul]

end RefreshTopProof.Gen
