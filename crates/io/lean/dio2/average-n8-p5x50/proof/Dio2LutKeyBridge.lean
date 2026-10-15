import Dio2Comp
import GenKey

/-!
# The bundle bridges of a dio2 LUT key table

The dio2 counterparts of refresh_top's `LutKeyTableIndex.rel`, `LutKeyTableStep.rel`/`.rel_1`
(+ `.hstep`/`.hstep_1`) and `LutKeyTable.rel`/`.rel_1` (`generic/GenBridge_LutKey.lean`), for the
bundles `Dio2Proof.Comp.LutKeyTable{Index,Step,Flat}.Comp(_1)`:

* `LutKeyTableIndex.rel`: entry `j` of a table is `selectInt j vals + off`.
* `LutKeyTableStep.rel` (variant 0, an exponent table: the loop input is `(cur, one, s, tv)`):
  iteration `j` adds `keyTermE σ o tv one s j`; `LutKeyTableStep.rel_1` (variant 1, a coefficient
  table: the loop input is `(cur, tv, one, s)`) adds `keyTermC σ tv one s j`. Note that the variant
  numbering is the reverse of refresh_top's.
* `LutKeyTableFlat.rel` (variant 0: the gadget rebased from `q2`) and `.rel_1` (variant 1: the
  gadget at the working modulus): the switching ciphertexts of one LUT branch. The tops are
  `topPairG` of the branch tags `tableTags σ i`, and since dio2 encrypts these ciphertexts without a
  Gaussian term, `σ(s) top_j + bot_j = (T u_j) G` holds **exactly**, with the branch table
  `T = keyTableG a b (Σ_{k < cnt} F tv k)` summed over the table loop and `u = (s, 1)`.
  `LutKeyTableFlat.rel_index` and `.rel_1_index` combine them with the index and step bridges.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

namespace LutKey

variable {n : Nat}

/-- Term `j` of a coefficient table (`lut.key-table.step`, variant 1): `lift(t_j) σ(one X^(j s))`
(zero past the `W` entries). -/
noncomputable def keyTermC {q W : Nat} (σ : Int) (tv : Fin W → Int) (one : ExactMatrix q n 1 1)
    (s : Int) (j : Nat) : ExactMatrix q n 1 1 :=
  if h : j < W then
    matrixMulScalarLeft (liftInteger (tv ⟨j, h⟩))
      (ringAutomorphism σ.toNat (multiplyMonomial one (Int.ofNat j * s)))
  else 0

/-- Term `j` of an exponent table (`lut.key-table.step`, variant 0): `σ(one X^(j s)) X^(t_j o)`
(zero past the `W` entries). -/
noncomputable def keyTermE {q W : Nat} (σ : Int) (o : Int) (tv : Fin W → Int)
    (one : ExactMatrix q n 1 1) (s : Int) (j : Nat) : ExactMatrix q n 1 1 :=
  if h : j < W then
    multiplyMonomial (ringAutomorphism σ.toNat (multiplyMonomial one (Int.ofNat j * s)))
      (tv ⟨j, h⟩ * o)
  else 0

/-- The table of a LUT branch: `a (b S)` for the summed terms `S`. -/
noncomputable def keyTableG {q : Nat} (a b : List Int) (S : ExactMatrix q n 1 1) :
    ExactMatrix q n 1 1 :=
  matrixMulScalarLeft (matrixPolynomial a) (matrixMulScalarLeft (matrixPolynomial b) S)

/-- A `seq` loop `sum + term j` from zero sums its terms (any additive state). -/
theorem table_sum_gen {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {final : α} (term : Nat → α) (hstep : ∀ j cur nxt, j < count → body j cur nxt → nxt = cur + term j)
    (h : MxxIR.IterRuns body count 0 final) : final = ∑ j ∈ Finset.range count, term j := by
  rw [iter_add_gen term hstep h, zero_add]

/-- `sigma`, the tag component of a branch table. -/
abbrev sigmaTagB : Blob := [115, 105, 103, 109, 97]
/-- `branch`, the tag component of a branch table. -/
abbrev branchTagB : Blob := [98, 114, 97, 110, 99, 104]

/-- The tag components of the tops of branch `i` with automorphism `σ`. -/
abbrev tableTags (σ : Int) (i : Nat) : List HashTagComponent :=
  [.bytes sigmaTagB, .u64Le σ, .bytes branchTagB, .u64Le (Int.ofNat i)]

set_option maxHeartbeats 4000000 in
/-- **An error-free LUT branch switching ciphertext** (one column): `bot = -σ(s) top + (T u) G + 0`,
so `σ(s) top + bot = (T u) G` exactly. -/
theorem branch_rhs_exact {q m : Nat} {σsp u : ErrorPoly n} {sσ T uM : ExactMatrix q n 1 1}
    (hsσ : sσ = fun _ _ ↦ reducePoly q n σsp) (hu : uM = fun _ _ ↦ reducePoly q n u)
    {G top : ExactMatrix q n 1 m} :
    reducePoly q n σsp • top + matrixAdd (matrixAdd (matrixMulScalarLeft (matrixNeg sσ) top)
        (matrixMulScalarLeft (matrixMulScalarLeft T uM) G)) (0 : ExactMatrix q n 1 m) =
      (T 0 0 * reducePoly q n u) • G := by
  subst hsσ hu
  simp only [matrixMulScalarLeft_eq, matrixNeg]
  funext r c
  simp only [matrixAdd, Matrix.add_apply, Matrix.smul_apply, Matrix.neg_apply, Matrix.zero_apply,
    smul_eq_mul]
  ring

/-- Both columns of an error-free branch: `σ(s) top_j + bot_j = (T (s, 1)_j) G`. -/
theorem branch_rhs_exact_pair {q m : Nat} (hq : 1 < q) (hn : 0 < n) {σ : Int} {sp : ErrorPoly n}
    {sM oneM sσ T : ExactMatrix q n 1 1} (hs : sM = fun _ _ ↦ reducePoly q n sp)
    (hone : oneM = fun _ _ ↦ reducePoly q n 1) (hσ : ringAutomorphismRuns σ sM sσ)
    {G top0 top1 : ExactMatrix q n 1 m} :
    ∃ hodd : Odd σ.toNat, ∀ j : Fin 2,
      reducePoly q n (substitution σ.toNat hodd sp) • ![top0, top1] j +
          ![matrixAdd (matrixAdd (matrixMulScalarLeft (matrixNeg sσ) top0)
              (matrixMulScalarLeft (matrixMulScalarLeft T sM) G)) (0 : ExactMatrix q n 1 m),
            matrixAdd (matrixAdd (matrixMulScalarLeft (matrixNeg sσ) top1)
              (matrixMulScalarLeft (matrixMulScalarLeft T oneM) G)) (0 : ExactMatrix q n 1 m)] j =
        (T 0 0 * reducePoly q n (unitPair sp j)) • G := by
  subst hs
  obtain ⟨hodd, hsσ⟩ := ringAutomorphism_reduceG hq hn hσ
  refine ⟨hodd, Fin.forall_fin_two.2 ⟨?_, ?_⟩⟩
  · exact branch_rhs_exact (u := sp) hsσ rfl
  · exact branch_rhs_exact (u := 1) hsσ hone

end LutKey

open LutKey

/-! ## Table entries and steps -/

/-- **A table index** (`lut.key-table.index`): entry `j` is `selectInt j vals + off`. -/
theorem LutKeyTableIndex.rel {vals : List Int} {off : Int} {backend : BackendContext}
    {hashModel : HashModel} {j : Nat} {v : Int}
    (h : Dio2Proof.Comp.LutKeyTableIndex.Comp vals off backend hashModel j () v) :
    v = MxxIR.selectInt (Int.ofNat j) vals + off := h

/-- **One term of an exponent table** (`lut.key-table.step`, variant 0, loop input
`(cur, one, s, tv)`): iteration `j` adds `keyTermE σ o tv one s j`. -/
theorem LutKeyTableStep.rel {n q W : Nat} {σ lo hi o : Int}
    {backend : BackendContext} {hashModel : HashModel} {j : Nat}
    {cur nxt one : ExactMatrix q n 1 1} {tv : Fin W → Int} {s : Int}
    (h : Dio2Proof.Comp.LutKeyTableStep.Comp q n W σ lo hi o backend hashModel j
      (cur, one, s, tv, ()) nxt) :
    nxt = cur + keyTermE σ o tv one s j := by
  obtain ⟨v0, v1, a0, -, -, a3, rfl⟩ := h
  obtain ⟨hj, rfl⟩ := familyGet_ofNatG j a3
  obtain rfl := a0.2.2.2
  rw [keyTermE, dif_pos hj]
  rfl

/-- **One term of a coefficient table** (`lut.key-table.step`, variant 1, loop input
`(cur, tv, one, s)`): iteration `j` adds `keyTermC σ tv one s j`. -/
theorem LutKeyTableStep.rel_1 {n q W : Nat} {σ lo hi : Int}
    {backend : BackendContext} {hashModel : HashModel} {j : Nat}
    {cur nxt one : ExactMatrix q n 1 1} {tv : Fin W → Int} {s : Int}
    (h : Dio2Proof.Comp.LutKeyTableStep.Comp_1 q n W lo hi σ backend hashModel j
      (cur, tv, one, s, ()) nxt) :
    nxt = cur + keyTermC σ tv one s j := by
  obtain ⟨v0, v1, -, -, a2, a3, rfl⟩ := h
  obtain ⟨hj, rfl⟩ := familyGet_ofNatG j a2
  obtain rfl := a3.2.2.2
  rw [keyTermC, dif_pos hj]
  rfl

/-- The steps of an exponent-table loop whose body bundles to `LutKeyTableStep.Comp`
(`hcomp`: the step adapter), for every table family `tv`. -/
theorem LutKeyTableStep.hstep {n q W : Nat} {σ lo hi o : Int}
    {backend : BackendContext} {hashModel : HashModel} {one : ExactMatrix q n 1 1} {s : Int}
    {child : Nat → ExactMatrix q n 1 1 × ExactMatrix q n 1 1 × Int × (Fin W → Int) × Unit →
      ExactMatrix q n 1 1 → Prop}
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1), child j (cur, one, s, tv, ()) nxt →
      Dio2Proof.Comp.LutKeyTableStep.Comp q n W σ lo hi o backend hashModel j
        (cur, one, s, tv, ()) nxt) :
    ∀ (tv : Fin W → Int) j cur nxt, child j (cur, one, s, tv, ()) nxt →
      nxt = cur + keyTermE σ o tv one s j :=
  fun tv j cur nxt hc ↦ LutKeyTableStep.rel (hcomp tv j cur nxt hc)

/-- The steps of a coefficient-table loop whose body bundles to `LutKeyTableStep.Comp_1`. -/
theorem LutKeyTableStep.hstep_1 {n q W : Nat} {σ lo hi : Int}
    {backend : BackendContext} {hashModel : HashModel} {one : ExactMatrix q n 1 1} {s : Int}
    {child : Nat → ExactMatrix q n 1 1 × (Fin W → Int) × ExactMatrix q n 1 1 × Int × Unit →
      ExactMatrix q n 1 1 → Prop}
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1), child j (cur, tv, one, s, ()) nxt →
      Dio2Proof.Comp.LutKeyTableStep.Comp_1 q n W lo hi σ backend hashModel j
        (cur, tv, one, s, ()) nxt) :
    ∀ (tv : Fin W → Int) j cur nxt, child j (cur, tv, one, s, ()) nxt →
      nxt = cur + keyTermC σ tv one s j :=
  fun tv j cur nxt hc ↦ LutKeyTableStep.rel_1 (hcomp tv j cur nxt hc)

/-! ## The switching ciphertexts of a LUT branch -/

set_option maxHeartbeats 64000000 in
/-- **The switching ciphertexts of LUT branch `i`** (`lut.key-table.flat`, variant 0: the table
loop input `(cur, one, -stride, tv)`, the gadget rebased from `q2`): the tops are hash samples
`topPairG … (tableTags σ i)` and, exactly (no error term),
`σ(s) top_j + bot_j = (T u_j) G` with `T = keyTableG a b (Σ_{k < cnt} F tv k)`, the table family
`tv` (`htable`), the loop's terms `F` (`hchild`) and `u = (s, 1)`. -/
theorem LutKeyTableFlat.rel {n q q2 m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {σ : Int} {tag : Blob} {oneL : List Int}
    {stride cnt : Int} {a b : List Int} {i : Nat} {key : ByteArray}
    {child : Nat → ExactMatrix q n 1 1 × ExactMatrix q n 1 1 × Int × (Fin W → Int) × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM oneM : ExactMatrix q n 1 1} {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : Dio2Proof.Comp.LutKeyTableFlat.Comp W q n q n m' q2 σ tag σ tag σ oneL stride cnt base
      digits a b a b backend hashModel i child table (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (tv : Fin W → Int) (htable : ∀ (j : Fin W) v, table j v → v = tv j)
    (F : (Fin W → Int) → Nat → ExactMatrix q n 1 1)
    (hchild : ∀ (tv' : Fin W → Int) j cur nxt,
      child j (cur, matrixPolynomial oneL, -stride, tv', ()) nxt → nxt = cur + F tv' j)
    (hq : 1 < q) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q n sp) (hone : oneM = fun _ _ ↦ reducePoly q n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags σ i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags σ i) 1 ∧
      ∃ (G0 : ExactMatrix q2 n 1 m') (G : ExactMatrix q n 1 m'),
        gadgetMatrixRuns backend base digits G0 ∧ centeredRebaseRuns G0 G ∧
      ∃ hodd : Odd σ.toNat, ∀ j : Fin 2,
        reducePoly q n (substitution σ.toNat hodd sp) • ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F tv k) : ExactMatrix q n 1 1) 0 0 *
            reducePoly q n (unitPair sp j)) • G := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, a0, a1, a2, a3, a4, a5, a6, hout⟩ := h
  simp only [Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl, rfl, rfl, -⟩ := hout
  obtain rfl : v3 = tv := funext fun j ↦ htable j _ (a3 j)
  obtain rfl := table_sum_gen (F v3) (fun j cur nxt _ hb ↦ hchild v3 j cur nxt hb) a4
  obtain ⟨hodd, hpay⟩ := branch_rhs_exact_pair (m := m') (G := v6) (top0 := top0) (top1 := top1)
    (T := keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F v3 k)) hq hn hs hone a2
  exact ⟨topPairG_of_hash (pre := tableTags σ i) 0 (by norm_num) a0,
    topPairG_of_hash (pre := tableTags σ i) 1 (by norm_num) a1, v5, v6, a5, a6, hodd, hpay⟩

set_option maxHeartbeats 64000000 in
/-- **The switching ciphertexts of LUT branch `i`** (variant 1: the table loop input
`(cur, tv, one, -stride)`, the gadget at the working modulus); as `LutKeyTableFlat.rel`. -/
theorem LutKeyTableFlat.rel_1 {n q m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {σ : Int} {tag : Blob} {oneL : List Int}
    {stride cnt : Int} {a b : List Int} {i : Nat} {key : ByteArray}
    {child : Nat → ExactMatrix q n 1 1 × (Fin W → Int) × ExactMatrix q n 1 1 × Int × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM oneM : ExactMatrix q n 1 1} {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : Dio2Proof.Comp.LutKeyTableFlat.Comp_1 W q n q n m' σ tag σ tag σ oneL stride cnt base
      digits a b a b backend hashModel i child table (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (tv : Fin W → Int) (htable : ∀ (j : Fin W) v, table j v → v = tv j)
    (F : (Fin W → Int) → Nat → ExactMatrix q n 1 1)
    (hchild : ∀ (tv' : Fin W → Int) j cur nxt,
      child j (cur, tv', matrixPolynomial oneL, -stride, ()) nxt → nxt = cur + F tv' j)
    (hq : 1 < q) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q n sp) (hone : oneM = fun _ _ ↦ reducePoly q n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags σ i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags σ i) 1 ∧
      ∃ G : ExactMatrix q n 1 m', gadgetMatrixRuns backend base digits G ∧
      ∃ hodd : Odd σ.toNat, ∀ j : Fin 2,
        reducePoly q n (substitution σ.toNat hodd sp) • ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F tv k) : ExactMatrix q n 1 1) 0 0 *
            reducePoly q n (unitPair sp j)) • G := by
  obtain ⟨v0, v1, v2, v3, v4, v5, a0, a1, a2, a3, a4, a5, hout⟩ := h
  simp only [Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl, rfl, rfl, -⟩ := hout
  obtain rfl : v3 = tv := funext fun j ↦ htable j _ (a3 j)
  obtain rfl := table_sum_gen (F v3) (fun j cur nxt _ hb ↦ hchild v3 j cur nxt hb) a4
  obtain ⟨hodd, hpay⟩ := branch_rhs_exact_pair (m := m') (G := v5) (top0 := top0) (top1 := top1)
    (T := keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F v3 k)) hq hn hs hone a2
  exact ⟨topPairG_of_hash (pre := tableTags σ i) 0 (by norm_num) a0,
    topPairG_of_hash (pre := tableTags σ i) 1 (by norm_num) a1, v5, a5, hodd, hpay⟩

/-- **`LutKeyTableFlat.rel` with the table and the loop resolved**: the table bundles to
`LutKeyTableIndex.Comp vals off` and the loop body to `LutKeyTableStep.Comp` (an exponent table),
so `T = keyTableG a b (Σ_{k < cnt} keyTermE σ o tv (oneL) (-stride) k)` with
`tv j = selectInt j vals + off`. -/
theorem LutKeyTableFlat.rel_index {n q q2 m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {σ : Int} {tag : Blob} {oneL : List Int}
    {stride cnt : Int} {a b : List Int} {i : Nat} {key : ByteArray} {vals : List Int}
    {off lo hi o : Int}
    {child : Nat → ExactMatrix q n 1 1 × ExactMatrix q n 1 1 × Int × (Fin W → Int) × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM oneM : ExactMatrix q n 1 1} {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : Dio2Proof.Comp.LutKeyTableFlat.Comp W q n q n m' q2 σ tag σ tag σ oneL stride cnt base
      digits a b a b backend hashModel i child table (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (htable : ∀ j v, table j v → Dio2Proof.Comp.LutKeyTableIndex.Comp vals off backend hashModel j () v)
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1),
      child j (cur, matrixPolynomial oneL, -stride, tv, ()) nxt →
      Dio2Proof.Comp.LutKeyTableStep.Comp q n W σ lo hi o backend hashModel j
        (cur, matrixPolynomial oneL, -stride, tv, ()) nxt)
    (hq : 1 < q) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q n sp) (hone : oneM = fun _ _ ↦ reducePoly q n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags σ i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags σ i) 1 ∧
      ∃ (G0 : ExactMatrix q2 n 1 m') (G : ExactMatrix q n 1 m'),
        gadgetMatrixRuns backend base digits G0 ∧ centeredRebaseRuns G0 G ∧
      ∃ hodd : Odd σ.toNat, ∀ j : Fin 2,
        reducePoly q n (substitution σ.toNat hodd sp) • ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat,
              keyTermE σ o (fun j : Fin W ↦ MxxIR.selectInt (Int.ofNat j) vals + off)
                (matrixPolynomial oneL) (-stride) k) : ExactMatrix q n 1 1) 0 0 *
            reducePoly q n (unitPair sp j)) • G :=
  LutKeyTableFlat.rel h (fun j ↦ MxxIR.selectInt (Int.ofNat j) vals + off)
    (fun j v hv ↦ LutKeyTableIndex.rel (htable j v hv))
    (fun tv k ↦ keyTermE σ o tv (matrixPolynomial oneL) (-stride) k)
    (LutKeyTableStep.hstep hcomp) hq hn hs hone

/-- **`LutKeyTableFlat.rel_1` with the table and the loop resolved** (a coefficient table):
`T = keyTableG a b (Σ_{k < cnt} keyTermC σ tv (oneL) (-stride) k)` with
`tv j = selectInt j vals + off`. -/
theorem LutKeyTableFlat.rel_1_index {n q m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {σ : Int} {tag : Blob} {oneL : List Int}
    {stride cnt : Int} {a b : List Int} {i : Nat} {key : ByteArray} {vals : List Int}
    {off lo hi : Int}
    {child : Nat → ExactMatrix q n 1 1 × (Fin W → Int) × ExactMatrix q n 1 1 × Int × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM oneM : ExactMatrix q n 1 1} {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : Dio2Proof.Comp.LutKeyTableFlat.Comp_1 W q n q n m' σ tag σ tag σ oneL stride cnt base
      digits a b a b backend hashModel i child table (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (htable : ∀ j v, table j v → Dio2Proof.Comp.LutKeyTableIndex.Comp vals off backend hashModel j () v)
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1),
      child j (cur, tv, matrixPolynomial oneL, -stride, ()) nxt →
      Dio2Proof.Comp.LutKeyTableStep.Comp_1 q n W lo hi σ backend hashModel j
        (cur, tv, matrixPolynomial oneL, -stride, ()) nxt)
    (hq : 1 < q) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q n sp) (hone : oneM = fun _ _ ↦ reducePoly q n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags σ i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags σ i) 1 ∧
      ∃ G : ExactMatrix q n 1 m', gadgetMatrixRuns backend base digits G ∧
      ∃ hodd : Odd σ.toNat, ∀ j : Fin 2,
        reducePoly q n (substitution σ.toNat hodd sp) • ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat,
              keyTermC σ (fun j : Fin W ↦ MxxIR.selectInt (Int.ofNat j) vals + off)
                (matrixPolynomial oneL) (-stride) k) : ExactMatrix q n 1 1) 0 0 *
            reducePoly q n (unitPair sp j)) • G :=
  LutKeyTableFlat.rel_1 h (fun j ↦ MxxIR.selectInt (Int.ofNat j) vals + off)
    (fun j v hv ↦ LutKeyTableIndex.rel (htable j v hv))
    (fun tv k ↦ keyTermC σ tv (matrixPolynomial oneL) (-stride) k)
    (LutKeyTableStep.hstep_1 hcomp) hq hn hs hone

/-! ## The keys of a LUT (key side, no block switch) -/

namespace LutKey

variable {n : Nat}

/-- The input keys `A` of a LUT key loop given as `(A00, A10, A01, A11)`. -/
def keyA {α : Type} (a00 a10 a01 a11 : α) : Fin 2 → Fin 2 → α := ![![a00, a01], ![a10, a11]]

/-- Four switching families as `fam k j`. -/
def fam4 {α : Type} {W : Nat} (f00 f01 f10 f11 : Fin W → α) : Fin 2 → Fin 2 → Fin W → α :=
  ![![f00, f01], ![f10, f11]]

/-- The keys of a dio2 LUT branch with automorphism `σ`: the input keys `A` (at `q`) routed by `σ`
and switched by the branch's ciphertexts `sw` (at `q'`, rebased to `q`), then multiplied by the
helper keys `Ad` (at `q''`, the switched keys rebased to `q''`). Unlike refresh_top's `branchKeyG`
there is no block modulus switch between the two multiplications. -/
noncomputable def branchKeyD {q' q q'' m m' : Nat} (backend : BackendContext) (base digits : Int)
    (σ : Nat) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (Ad : Fin 2 → Fin 2 → ExactMatrix q'' n 1 m)
    (sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m') : Fin 2 → Fin 2 → ExactMatrix q'' n 1 m' :=
  fun k j ↦
    let Asw : Fin 2 → ExactMatrix q n 1 m' := fun k' ↦
      ringAutomorphism σ (A k' 0) * rdecG (inner := m) backend base digits (sw 0 j) +
        ringAutomorphism σ (A k' 1) * rdecG (inner := m) backend base digits (sw 1 j)
    Ad k 0 * rdecG (inner := m) backend base digits (Asw 0) +
      Ad k 1 * rdecG (inner := m) backend base digits (Asw 1)

set_option maxHeartbeats 4000000 in
/-- The keys of a dio2 LUT branch as the preprocessing computes them (two-row stacks, one-row
decompositions, no block switch). -/
theorem branchKeyD_of_pre {q' q q'' m m' bound : Nat} {backend : BackendContext}
    {base digits : Int} {index : Int}
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q'' n 1 m}
    {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    (hσ00 : ringAutomorphismRuns index (A 0 0) a00) (hσ10 : ringAutomorphismRuns index (A 1 0) a10)
    (hσ01 : ringAutomorphismRuns index (A 0 1) a01) (hσ11 : ringAutomorphismRuns index (A 1 1) a11)
    {S0 S1 : ExactMatrix q n (1 + 1) m} {T0 T1 : ExactMatrix q'' n (1 + 1) m}
    (hS0 : concatRows a00 a10 S0) (hS1 : concatRows a01 a11 S1)
    (hT0 : concatRows (Ad 0 0) (Ad 1 0) T0) (hT1 : concatRows (Ad 0 1) (Ad 1 1) T1)
    {sw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {dA dB : Fin 2 → ExactMatrix q' n m m'} {rA rB : Fin 2 → ExactMatrix q n m m'}
    (hdA : ∀ j, gadgetDecomposeRuns backend base digits (sw 0 j) (dA j))
    (hrA : ∀ j, compactCenteredRebaseRuns bound (dA j) (rA j))
    (hdB : ∀ j, gadgetDecomposeRuns backend base digits (sw 1 j) (dB j))
    (hrB : ∀ j, compactCenteredRebaseRuns bound (dB j) (rB j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 0 1 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixAdd (matrixMul S0 (rA j)) (matrixMul S1 (rB j))) 1 2 0 m' (M 1 j))
    {dC dE : Fin 2 → ExactMatrix q n m m'} {rC rE : Fin 2 → ExactMatrix q'' n m m'}
    (hdC : ∀ j, gadgetDecomposeRuns backend base digits (M 0 j) (dC j))
    (hrC : ∀ j, compactCenteredRebaseRuns bound (dC j) (rC j))
    (hdE : ∀ j, gadgetDecomposeRuns backend base digits (M 1 j) (dE j))
    (hrE : ∀ j, compactCenteredRebaseRuns bound (dE j) (rE j))
    {K : Fin 2 → Fin 2 → ExactMatrix q'' n 1 m'}
    (hK0 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 0 1 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixAdd (matrixMul T0 (rC j)) (matrixMul T1 (rE j))) 1 2 0 m' (K 1 j)) :
    K = branchKeyD backend base digits index.toNat A Ad sw := by
  have hσ : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = ringAutomorphism index.toNat x := fun hxy ↦ hxy.2.2.2
  have hM : ∀ k j, M k j =
      ringAutomorphism index.toNat (A k 0) * rdecG (inner := m) backend base digits (sw 0 j) +
      ringAutomorphism index.toNat (A k 1) * rdecG (inner := m) backend base digits (sw 1 j) := by
    intro k j
    obtain ⟨s0a, s0b⟩ := stack2_mul_rowsG hS0 (rA j)
    obtain ⟨s1a, s1b⟩ := stack2_mul_rowsG hS1 (rB j)
    fin_cases k
    · show M 0 j = ringAutomorphism index.toNat (A 0 0) * _ + ringAutomorphism index.toNat (A 0 1) * _
      rw [slice_rowOfG (by norm_num) (hM0 j)]
      show rowOf (S0 * rA j + S1 * rB j) 0 = _
      rw [rowOf_addG, s0a, s1a, rdecG_eq (hdA j) (hrA j), rdecG_eq (hdB j) (hrB j), hσ hσ00,
        hσ hσ01]
    · show M 1 j = ringAutomorphism index.toNat (A 1 0) * _ + ringAutomorphism index.toNat (A 1 1) * _
      rw [slice_rowOfG (by norm_num) (hM1 j)]
      show rowOf (S0 * rA j + S1 * rB j) 1 = _
      rw [rowOf_addG, s0b, s1b, rdecG_eq (hdA j) (hrA j), rdecG_eq (hdB j) (hrB j), hσ hσ10,
        hσ hσ11]
  funext k j
  obtain ⟨t0a, t0b⟩ := stack2_mul_rowsG hT0 (rC j)
  obtain ⟨t1a, t1b⟩ := stack2_mul_rowsG hT1 (rE j)
  simp only [branchKeyD]
  fin_cases k
  · show K 0 j = Ad 0 0 * _ + Ad 0 1 * _
    rw [slice_rowOfG (by norm_num) (hK0 j)]
    show rowOf (T0 * rC j + T1 * rE j) 0 = _
    rw [rowOf_addG, t0a, t1a, rdecG_eq (hdC j) (hrC j), rdecG_eq (hdE j) (hrE j), hM, hM]
  · show K 1 j = Ad 1 0 * _ + Ad 1 1 * _
    rw [slice_rowOfG (by norm_num) (hK1 j)]
    show rowOf (T0 * rC j + T1 * rE j) 1 = _
    rw [rowOf_addG, t0b, t1b, rdecG_eq (hdC j) (hrC j), rdecG_eq (hdE j) (hrE j), hM, hM]

/-- The keys of branch `i` of a width-`W` dio2 LUT with automorphisms `σf`, root helper keys
`keyPairG … tag [σ_i]` and switching families `fam` (zero past the branches). -/
noncomputable def lutKeysD {q m m' W : Nat} (backend : BackendContext) (base digits : Int)
    (model : HashModel) (key : ByteArray) (tag : Blob) (σf : Nat → Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m')
    (i : Nat) : KeysG n q m' :=
  if h : i < W then
    keyTupleG (branchKeyD backend base digits (σf i).toNat A
      (keyPairG model key q m tag [.integer (σf i)]) (fun k j ↦ fam k j ⟨i, h⟩))
  else 0

/-- The automorphisms of a LUT table by a natural index. -/
def sigN (tab : List Int) : Nat → Int := fun i ↦ MxxIR.selectInt (Int.ofNat i) tab

/-- The input tuple of a dio2 LUT key-loop body: the four keys, the hash key, the input keys
`(A00, A10)`, the family `sw00`, the input keys `(A01, A11)` and the families `sw10, sw01, sw11`. -/
abbrev KeyStepIn (n q m m' W : Nat) :=
  ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
    ByteArray × ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q n 1 m') ×
    ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q n 1 m') ×
    (Fin W → ExactMatrix q n 1 m') × (Fin W → ExactMatrix q n 1 m') × Unit

end LutKey

/-! ## One iteration of a LUT key loop -/

set_option maxHeartbeats 64000000 in
/-- **One iteration of a dio2 LUT key loop** (`lut.key-round.step`, variant 0: the automorphism `σ`
a constant of the scope): the bundle on the keys `cur`, the hash key, the input keys
`keyA a00 a10 a01 a11` and the switching families adds the branch keys `lutKeysD … i` with the
root helper keys `keyPairG … tag [σ]`. -/
theorem LutKeyRoundStep.rel {n q m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {σ : Int} {tag : Blob} {key : ByteArray}
    {lo hi : Int} {bound : Nat} {mw : Int} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q n 1 m'} {i : Nat} {cur nxt : KeysG n q m'}
    (h : Dio2Proof.Comp.LutKeyRoundStep.Comp q n m' q m' m' m' m' m' m' m' m W q m W q W q W q m m
      tag σ tag σ σ σ lo hi base digits bound σ σ lo hi base digits bound mw base digits bound
      tag σ tag σ mw base digits bound mw lo hi base digits bound lo hi base digits bound
      mw base digits bound mw base digits bound mw mw mw backend hashModel i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ()) nxt)
    (hmw : mw = (m' : Int)) :
    nxt = cur + lutKeysD backend base digits hashModel key tag (fun _ ↦ σ)
      (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11) i := by
  subst hmw
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, c0, c1, c2, c3, c4, c5, -, -, c8, c9, c10, c11, c12, c13, -, -, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, -, -, c31, c32, c33, -, -, c36, c37, c38, c39,
    c40, c41, c42, c43, c44, c45, c46, c47, rfl⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_ofNatG i c8
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c16
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c31
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c36
  have hAd : (![![v0, v18], ![v1, v19]] : Fin 2 → Fin 2 → ExactMatrix q n 1 m) =
      keyPairG hashModel key q m tag [.integer σ] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer σ]) 0 0 (by norm_num) c0
    · exact keyPairG_of_hash (pre := [.integer σ]) 0 1 (by norm_num) c22
    · exact keyPairG_of_hash (pre := [.integer σ]) 1 0 (by norm_num) c1
    · exact keyPairG_of_hash (pre := [.integer σ]) 1 1 (by norm_num) c23
  have hK := branchKeyD_of_pre (A := keyA a00 a10 a01 a11) (Ad := ![![v0, v18], ![v1, v19]])
    (sw := fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i, hi⟩) c3 c4 c11 c12 c5 c13 c2 c24
    (dA := ![v7, v26]) (Fin.forall_fin_two.2 ⟨c9, c32⟩)
    (rA := ![v8, v27]) (Fin.forall_fin_two.2 ⟨c10, c33⟩)
    (dB := ![v13, v29]) (Fin.forall_fin_two.2 ⟨c17, c37⟩)
    (rB := ![v14, v30]) (Fin.forall_fin_two.2 ⟨c18, c38⟩)
    (M := ![![v15, v31], ![v21, v34]]) (Fin.forall_fin_two.2 ⟨c19, c39⟩)
    (Fin.forall_fin_two.2 ⟨c25, c42⟩)
    (dC := ![v16, v32]) (Fin.forall_fin_two.2 ⟨c20, c40⟩)
    (rC := ![v17, v33]) (Fin.forall_fin_two.2 ⟨c21, c41⟩)
    (dE := ![v22, v35]) (Fin.forall_fin_two.2 ⟨c26, c43⟩)
    (rE := ![v23, v36]) (Fin.forall_fin_two.2 ⟨c27, c44⟩)
    (K := ![![v24, v37], ![v38, v39]]) (Fin.forall_fin_two.2 ⟨c28, c45⟩)
    (Fin.forall_fin_two.2 ⟨c46, c47⟩)
  rw [hAd] at hK
  rw [lutKeysD, dif_pos hi, ← hK]
  rfl

set_option maxHeartbeats 64000000 in
/-- **One iteration of a dio2 LUT key loop** (variant 1: the automorphism of iteration `i` read
from the table, `σ_i = selectInt i tab`): adds `lutKeysD … (sigN tab) … i`. -/
theorem LutKeyRoundStep.rel_1 {n q m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {tag : Blob} {key : ByteArray}
    {lo hi : Int} {bound : Nat} {mw : Int} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q n 1 m'} {i : Nat} {cur nxt : KeysG n q m'}
    (h : Dio2Proof.Comp.LutKeyRoundStep.Comp_1 q n m' q m' m' m' m' m' m' m' m W q m W q W q W q m m
      tag tab tag tab tab tab lo hi base digits bound tab tab lo hi base digits bound mw base digits
      bound tag tab tag tab mw base digits bound mw lo hi base digits bound lo hi base digits bound
      mw base digits bound mw base digits bound mw mw mw backend hashModel i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ()) nxt)
    (hmw : mw = (m' : Int)) :
    nxt = cur + lutKeysD backend base digits hashModel key tag (sigN tab)
      (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11) i := by
  subst hmw
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, c0, c1, c2, c3, c4, c5, -, -, c8, c9, c10, c11, c12, c13, -, -, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, -, -, c31, c32, c33, -, -, c36, c37, c38, c39,
    c40, c41, c42, c43, c44, c45, c46, c47, rfl⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_ofNatG i c8
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c16
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c31
  obtain ⟨_, rfl⟩ := familyGet_ofNatG i c36
  have hAd : (![![v0, v18], ![v1, v19]] : Fin 2 → Fin 2 → ExactMatrix q n 1 m) =
      keyPairG hashModel key q m tag [.integer (sigN tab i)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (sigN tab i)]) 0 0 (by norm_num) c0
    · exact keyPairG_of_hash (pre := [.integer (sigN tab i)]) 0 1 (by norm_num) c22
    · exact keyPairG_of_hash (pre := [.integer (sigN tab i)]) 1 0 (by norm_num) c1
    · exact keyPairG_of_hash (pre := [.integer (sigN tab i)]) 1 1 (by norm_num) c23
  have hK := branchKeyD_of_pre (index := sigN tab i) (A := keyA a00 a10 a01 a11) (Ad := ![![v0, v18], ![v1, v19]])
    (sw := fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i, hi⟩) c3 c4 c11 c12 c5 c13 c2 c24
    (dA := ![v7, v26]) (Fin.forall_fin_two.2 ⟨c9, c32⟩)
    (rA := ![v8, v27]) (Fin.forall_fin_two.2 ⟨c10, c33⟩)
    (dB := ![v13, v29]) (Fin.forall_fin_two.2 ⟨c17, c37⟩)
    (rB := ![v14, v30]) (Fin.forall_fin_two.2 ⟨c18, c38⟩)
    (M := ![![v15, v31], ![v21, v34]]) (Fin.forall_fin_two.2 ⟨c19, c39⟩)
    (Fin.forall_fin_two.2 ⟨c25, c42⟩)
    (dC := ![v16, v32]) (Fin.forall_fin_two.2 ⟨c20, c40⟩)
    (rC := ![v17, v33]) (Fin.forall_fin_two.2 ⟨c21, c41⟩)
    (dE := ![v22, v35]) (Fin.forall_fin_two.2 ⟨c26, c43⟩)
    (rE := ![v23, v36]) (Fin.forall_fin_two.2 ⟨c27, c44⟩)
    (K := ![![v24, v37], ![v38, v39]]) (Fin.forall_fin_two.2 ⟨c28, c45⟩)
    (Fin.forall_fin_two.2 ⟨c46, c47⟩)
  rw [hAd] at hK
  rw [lutKeysD, dif_pos hi, ← hK]
  rfl

/-! ## LUT key members -/

/-- **The summed branch keys of a LUT** (`lut.key-member.flat`): from zero keys, a loop of
`cnt` iterations whose body adds `F k` (`hchild`) ends on `Σ_{k < cnt} F k` (no move down). -/
theorem LutKeyMemberFlat.rel {n q m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q n 1 m'}
    {child : Nat → KeyStepIn n q m m' W → KeysG n q m' → Prop} {cnt : Int}
    {outputs : KeysG n q m'}
    (h : Dio2Proof.Comp.LutKeyMemberFlat.Comp q n m' q n m q n m W q n m' q n m q n m W q n m' W
      q n m' W q n m' cnt backend hashModel child (key, a00, a10, f00, a01, a11, f10, f01, f11, ())
      outputs)
    {F : Nat → KeysG n q m'}
    (hchild : ∀ i (cur nxt : KeysG n q m'),
      child i (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ())
        nxt → nxt = cur + F i) :
    outputs = ∑ k ∈ Finset.range cnt.toNat, F k := by
  obtain ⟨v0, v1, v2, v3, a0, rfl⟩ := h
  have := iter_add_gen F (fun i cur nxt _ hb ↦ hchild i cur nxt hb) a0
  rw [this]
  exact zero_add _

/-- **The summed branch keys of a LUT member whose loop body bundles to `LutKeyRoundStep.Comp_1`**:
`Σ_{k < cnt} lutKeysD … (sigN tab) (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11) k`. -/
theorem LutKeyMemberFlat.rel_steps {n q m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {tag : Blob} {key : ByteArray}
    {lo hi : Int} {bound : Nat} {mw : Int} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q n 1 m'}
    {child : Nat → KeyStepIn n q m m' W → KeysG n q m' → Prop} {cnt : Int}
    {outputs : KeysG n q m'}
    (h : Dio2Proof.Comp.LutKeyMemberFlat.Comp q n m' q n m q n m W q n m' q n m q n m W q n m' W
      q n m' W q n m' cnt backend hashModel child (key, a00, a10, f00, a01, a11, f10, f01, f11, ())
      outputs)
    (hcomp : ∀ i (cur nxt : KeysG n q m'),
      child i (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ())
        nxt →
      Dio2Proof.Comp.LutKeyRoundStep.Comp_1 q n m' q m' m' m' m' m' m' m' m W q m W q W q W q m m
        tag tab tag tab tab tab lo hi base digits bound tab tab lo hi base digits bound mw base
        digits bound tag tab tag tab mw base digits bound mw lo hi base digits bound lo hi base
        digits bound mw base digits bound mw base digits bound mw mw mw backend hashModel i
        (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ()) nxt)
    (hmw : mw = (m' : Int)) :
    outputs = ∑ k ∈ Finset.range cnt.toNat, lutKeysD backend base digits hashModel key tag
      (sigN tab) (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11) k :=
  LutKeyMemberFlat.rel h fun i cur nxt hc ↦ LutKeyRoundStep.rel_1 (hcomp i cur nxt hc) hmw

end Dio2Proof
