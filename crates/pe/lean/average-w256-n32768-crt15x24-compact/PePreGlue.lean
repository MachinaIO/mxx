import PeStepData
import PePreBase

/-!
# Shared glue for the preprocessing root

Single-iteration loops, the selection of the attribute position, the product key rows of one
position from its top step and key loop, and the assembly of the per-lane results of a LUT depth
(the first-depth key rows of every branch, the parent keys of the second depth, and the output key
from the member sums).
-/

set_option maxRecDepth 200000

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem iter_one {σ : Type} {body : Nat → σ → σ → Prop} {init final : σ}
    (h : MxxIR.IterRuns body (Int.toNat 1) init final) : body 0 init final := by
  obtain ⟨c, hc, hb⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h
  cases hc
  exact hb

theorem select_two {α : Type} {a b o : α} {idx : Int} {p : Fin 2}
    (h : select idx [a, b] o) (hp : idx = ((p : Nat) : Int)) : o = ![a, b] p := by
  obtain ⟨q, hq, ho⟩ := h
  have hqp : (q : Nat) = p := by omega
  rw [ho]
  obtain ⟨q, hq'⟩ := q
  obtain ⟨p, hp'⟩ := p
  simp only at hqp
  subst hqp
  interval_cases q <;> rfl

theorem familyStatic_four {f : Fin 2 → Int} {o : Int} {k : Nat} (hk : k < 2)
    (h : familyGetStatic f (k : Int) o) : o = f ⟨k, hk⟩ := by
  obtain ⟨q, hq, ho⟩ := h
  have e : q = ⟨k, hk⟩ := Fin.ext (by simp only; omega)
  rw [ho, e]

theorem familyStatic_eq {α : Type} {count : Nat} {f : Fin count → α} {o : α} {k : Nat} (hk : k < count)
    (h : familyGetStatic f (k : Int) o) : o = f ⟨k, hk⟩ := by
  obtain ⟨q, hq, ho⟩ := h
  have e : q = ⟨k, hk⟩ := Fin.ext (by simp only; omega)
  rw [ho, e]

/-- The product key rows of one position: the top step from zero rows, then the key loop. -/
theorem pos_rows {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 30} {A0 A1 : ExactMatrix Q N 1 60}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {X : ExactMatrix Q N 1 60} {comp : Fin 2 → Fin 60 → Fin 30 → ExactPoly Q N}
    {t0 t1 f0 f1 : ExactMatrix Q N 1 60}
    (htop : ∀ (r : Fin 2) (w : Fin 60), ![t0, t1] r 0 w = ![(0 : ExactMatrix Q N 1 60), 0] r 0 w +
      ∑ a : Fin 30, ![A0, A1] r 0 ⟨a, by omega⟩ * rowDigits X a w)
    (hkey : ∀ (k : Fin 2) (jj : Fin 60), ![f0, f1] k 0 jj = ![t0, t1] k 0 jj + ∑ a, comp k jj a)
    (r : Fin 2) (jj : Fin 60) :
    ![f0, f1] r 0 jj = ∑ a, comp r jj a + ∑ a : Fin 30, A r 0 0 a * rowDigits X a jj := by
  have hz : ![(0 : ExactMatrix Q N 1 60), 0] r 0 jj = 0 := by fin_cases r <;> rfl
  have h1 := hkey r jj
  have h2 := htop r jj
  rw [hz, zero_add] at h2
  rw [h1, h2, add_comm]
  exact congrArg (fun x ↦ ∑ a, comp r jj a + x)
    (Finset.sum_congr rfl fun a _ ↦ by rw [key_row_entry hA0 hA1])

/-- The key rows of first-depth branch `b` for the incoming key `M`: the LUT key companion sums
plus the routed public top term. -/
noncomputable def lutTRow (model : HashModel) (key : ByteArray) (b : Fin 30)
    (M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30) (r : Fin 2) (jj : Fin 60) : ExactPoly Q N :=
  ∑ a, lutKeyCompanionSum model key b M r jj a +
    ∑ a : Fin 30, substitution (kSw b) (kSw_odd b) (M r 0 0 a) * rowDigits (switchTopRow model key b) a jj

/-- The key rows of every first-depth branch from the key rows of the internal-branch lanes and
the leaf lanes. -/
theorem branch_rows {model : HashModel} {key : ByteArray} {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    (d1 : BranchData) {TI0 TI1 : Fin 15 → ExactMatrix Q N 1 60} {TL0 TL1 : Fin 15 → ExactMatrix Q N 1 60}
    (hI : ∀ (b : Fin 15) (r : Fin 2) (jj : Fin 60), ![TI0 b, TI1 b] r 0 jj = lutTRow model key ⟨b, by omega⟩ M r jj)
    (hL : ∀ (m : Fin 15) (r : Fin 2) (jj : Fin 60), ![TL0 m, TL1 m] r 0 jj =
      lutTRow model key ⟨15 + m, by omega⟩ M r jj)
    (h0 : d1.TI0 = TI0) (h1 : d1.TI1 = TI1) (h2 : d1.TL0 = TL0) (h3 : d1.TL1 = TL1)
    (b : Fin 30) (r : Fin 2) (jj : Fin 60) :
    ![d1.Tb0 b b.isLt, d1.Tb1 b b.isLt] r 0 jj = lutTRow model key b M r jj := by
  obtain ⟨bv, hb⟩ := b
  by_cases hlt : bv < 15
  · have e := hI ⟨bv, hlt⟩ r jj
    simp only [BranchData.Tb0, BranchData.Tb1, dif_pos hlt, h0, h1]
    exact e
  · have e := hL ⟨bv - 15, by omega⟩ r jj
    simp only [BranchData.Tb0, BranchData.Tb1, dif_neg hlt, h2, h3]
    rw [e]
    exact congrArg (fun c : Fin 30 ↦ lutTRow model key c M r jj) (Fin.ext (by simp only; omega))

/-- The key of internal branch `b`, from the internal key rows. -/
theorem key1_int {model : HashModel} {key : ByteArray} (d1 : BranchData)
    {TI0 TI1 : Fin 15 → ExactMatrix Q N 1 60} (hk : d1.key = key) (h0 : d1.TI0 = TI0) (h1 : d1.TI1 = TI1)
    (b : Fin 15) (r j : Fin 2) :
    d1.key1 model ⟨b, by omega⟩ r j = pairMul (helperKey model key (kSw b) r) (actionDigits (TI0 b) (TI1 b) j) := by
  simp only [BranchData.key1, BranchData.Tb0, BranchData.Tb1, dif_pos b.isLt, hk, h0, h1]

/-- The key of first-depth leaf `15 + m`, from the leaf key rows. -/
theorem key1_leaf {model : HashModel} {key : ByteArray} (d1 : BranchData)
    {TL0 TL1 : Fin 15 → ExactMatrix Q N 1 60} (hk : d1.key = key) (h2 : d1.TL0 = TL0) (h3 : d1.TL1 = TL1)
    (m : Fin 15) (r j : Fin 2) :
    d1.key1 model ⟨15 + m, by omega⟩ r j =
      pairMul (helperKey model key (kSw (15 + m)) r) (actionDigits (TL0 m) (TL1 m) j) := by
  have hn : ¬ (15 + (m : Nat) < 15) := by omega
  have hm : (⟨15 + (m : Nat) - 15, by omega⟩ : Fin 15) = m := Fin.ext (by simp)
  simp only [BranchData.key1, BranchData.Tb0, BranchData.Tb1, dif_neg hn, hk, h2, h3, hm]

/-- The key of second-depth leaf `l`, from the second-depth key rows. -/
theorem key2_eq {model : HashModel} {key : ByteArray} (d2 : Leaf2Data)
    {T0 T1 : Fin 225 → ExactMatrix Q N 1 60} (hk : d2.key = key) (h0 : d2.T0 = T0) (h1 : d2.T1 = T1)
    (l : Fin 225) (r j : Fin 2) :
    d2.key2 model l r j =
      pairMul (helperKey model key (kSw (15 + l / 15)) r) (actionDigits (T0 l) (T1 l) j) := by
  simp only [Leaf2Data.key2, Leaf2Data.Tb0, Leaf2Data.Tb1, hk, h0, h1]

/-- The switched keys of the internal lanes are the parent keys `key1` of the second depth. -/
theorem parent_keys {model : HashModel} {key : ByteArray} (d1 : BranchData)
    {TI0 TI1 : Fin 15 → ExactMatrix Q N 1 60} (hk : d1.key = key) (h0 : d1.TI0 = TI0) (h1 : d1.TI1 = TI1)
    {K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 30}
    (hK : ∀ (e : Fin 15) (r j : Fin 2), ![![K00 e, K01 e], ![K10 e, K11 e]] r j =
      pairMul (helperKey model key (kSw e) r) (actionDigits (TI0 e) (TI1 e) j))
    (e : Fin 15) : parentKeys K00 K01 K10 K11 e = d1.key1 model ⟨e, by omega⟩ := by
  funext r j
  rw [key1_int d1 hk h0 h1 e r j, ← hK e r j]
  rfl

/-- The initial state of a member sum: members `0` and `1` added. -/
theorem member_init {n : Nat} {f00 f01 f10 f11 : Fin (n + 2) → ExactMatrix Q N 1 30}
    {a0 a1 b0 b1 c0 c1 d0 d1 : ExactMatrix Q N 1 30}
    (ha0 : familyGetStatic f00 0 a0) (ha1 : familyGetStatic f00 1 a1)
    (hb0 : familyGetStatic f01 0 b0) (hb1 : familyGetStatic f01 1 b1)
    (hc0 : familyGetStatic f10 0 c0) (hc1 : familyGetStatic f10 1 c1)
    (hd0 : familyGetStatic f11 0 d0) (hd1 : familyGetStatic f11 1 d1) (r j : Fin 2) :
    keyOf (matrixAdd a0 a1, matrixAdd b0 b1, matrixAdd c0 c1, matrixAdd d0 d1, ()) r j =
      ![![f00 0, f01 0], ![f10 0, f11 0]] r j + ![![f00 1, f01 1], ![f10 1, f11 1]] r j := by
  rw [familyStatic_eq (k := 0) (by omega) ha0, familyStatic_eq (k := 1) (by omega) ha1,
    familyStatic_eq (k := 0) (by omega) hb0, familyStatic_eq (k := 1) (by omega) hb1,
    familyStatic_eq (k := 0) (by omega) hc0, familyStatic_eq (k := 1) (by omega) hc1,
    familyStatic_eq (k := 0) (by omega) hd0, familyStatic_eq (k := 1) (by omega) hd1]
  fin_cases r <;> fin_cases j <;> rfl

/-- The output key of a step from the product key term and the three member sums: the explicit
form of `stepKey`. -/
theorem pre_sum {model : HashModel} {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 30}
    (d1 : BranchData) (d2 : Leaf2Data) (coef : Nat → ExactMatrix Q N 1 1)
    {XI00 XI01 XI10 XI11 : Fin 15 → ExactMatrix Q N 1 30} {XL00 XL01 XL10 XL11 : Fin 15 → ExactMatrix Q N 1 30}
    {XM00 XM01 XM10 XM11 : Fin 225 → ExactMatrix Q N 1 30}
    (hI : ∀ (b : Fin 15) (r j : Fin 2), ![![XI00 b, XI01 b], ![XI10 b, XI11 b]] r j =
      d1.key1 model ⟨b, by omega⟩ r j * scalarV (coef (kSw b)))
    (hL : ∀ (m : Fin 15) (r j : Fin 2), ![![XL00 m, XL01 m], ![XL10 m, XL11 m]] r j =
      d1.key1 model ⟨15 + m, by omega⟩ r j * scalarV (coef (kSw (15 + m))))
    (hM : ∀ (l : Fin 225) (r j : Fin 2), ![![XM00 l, XM01 l], ![XM10 l, XM11 l]] r j =
      d2.key2 model l r j * scalarV (coef (kLeaf2 l)))
    {s : KeyState}
    (hs : ∀ r j, keyOf s r j = M r j * scalarV (coef 1) +
      ∑ b, ![![XI00 b, XI01 b], ![XI10 b, XI11 b]] r j + ∑ m, ![![XL00 m, XL01 m], ![XL10 m, XL11 m]] r j +
      ∑ l, ![![XM00 l, XM01 l], ![XM10 l, XM11 l]] r j) (r j : Fin 2) :
    keyOf s r j = M r j * scalarV (coef 1) +
      ∑ b : Fin 15, d1.key1 model ⟨b, by omega⟩ r j * scalarV (coef (kSw b)) +
      ∑ i : Fin 15, d1.key1 model ⟨15 + i, by omega⟩ r j * scalarV (coef (kSw (15 + i))) +
      ∑ l : Fin 225, d2.key2 model l r j * scalarV (coef (kLeaf2 l)) := by
  rw [hs]
  simp only [hI, hL, hM]

/-- Grouping leaves by member preserves the sum: slot zero is the first-depth leaf,
and slot `1 + parent` is second-depth leaf `parent + 15 * member`. -/
theorem grouped_leaf_sum {α : Type} [AddCommMonoid α]
    (first : Fin 15 → α) (second : Fin 225 → α) :
    (∑ member : Fin 15, (first member +
      ∑ parent : Fin 15, second ⟨parent + 15 * member, by omega⟩)) =
      (∑ member : Fin 15, first member) + ∑ leaf : Fin 225, second leaf := by
  rw [Finset.sum_add_distrib]
  congr 1
  rw [← (finProdFinEquiv : Fin 15 × Fin 15 ≃ Fin 225).sum_comp,
    Fintype.sum_prod_type]
  rfl

end PeProof
