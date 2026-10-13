import PeOnlineGlue
import PePreBase

/-!
# Shared glue for the preprocessing root

Single-iteration loops, the selection of the attribute position, the halves of a concatenation,
and the product key rows of one position from its top step and key loop.
-/

namespace PeProof

open Mxx.Primitives MxxRuntime MxxExponentLut MxxPe

theorem iter_one {σ : Type} {body : Nat → σ → σ → Prop} {init final : σ}
    (h : MxxIR.IterRuns body (Int.toNat 1) init final) : body 0 init final := by
  obtain ⟨c, hc, hb⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h
  cases hc
  exact hb

theorem select_four {α : Type} {a b c d o : α} {idx : Int} {p : Fin 4}
    (h : select idx [a, b, c, d] o) (hp : idx = ((p : Nat) : Int)) : o = ![a, b, c, d] p := by
  obtain ⟨q, hq, ho⟩ := h
  have hqp : (q : Nat) = p := by omega
  rw [ho]
  obtain ⟨q, hq'⟩ := q
  obtain ⟨p, hp'⟩ := p
  simp only at hqp
  subst hqp
  interval_cases q <;> rfl

theorem familyStatic_four {f : Fin 4 → Int} {o : Int} {k : Nat} (hk : k < 4)
    (h : familyGetStatic f (k : Int) o) : o = f ⟨k, hk⟩ := by
  obtain ⟨q, hq, ho⟩ := h
  have e : q = ⟨k, hk⟩ := Fin.ext (by simp only; omega)
  rw [ho, e]

theorem concat_left {a b : Nat} {x : ExactMatrix Q N 1 a} {y : ExactMatrix Q N 1 b}
    {T : ExactMatrix Q N 1 (a + b)} (h : concatColumns x y T) (c : Nat) (hc : c < a) :
    T 0 ⟨c, by omega⟩ = x 0 ⟨c, hc⟩ := by
  rw [h 0 ⟨c, by omega⟩, dif_pos hc]

theorem concat_right {a b : Nat} {x : ExactMatrix Q N 1 a} {y : ExactMatrix Q N 1 b}
    {T : ExactMatrix Q N 1 (a + b)} (h : concatColumns x y T) (c : Nat) (hc : c < b) :
    T 0 ⟨a + c, by omega⟩ = y 0 ⟨c, hc⟩ := by
  rw [h 0 ⟨a + c, by omega⟩, dif_neg (by simp)]
  exact congrArg (y 0) (Fin.ext (by simp))

theorem rowBlock_concat_left {x : ExactMatrix Q N 1 1936} {y : ExactMatrix Q N 1 9240}
    {T : ExactMatrix Q N 1 11176} (h : concatColumns x y T) (b : Nat) (hb : b < 22) :
    rowBlock T b (by omega) = rowBlock x b (by omega) := by
  funext r l
  rw [Subsingleton.elim r 0]
  exact concat_left h _ (by omega)

theorem rowBlock_concat_right {x : ExactMatrix Q N 1 1936} {y : ExactMatrix Q N 1 9240}
    {T : ExactMatrix Q N 1 11176} (h : concatColumns x y T) (l : Nat) (hl : l < 105) :
    rowBlock T (22 + l) (by omega) = rowBlock y l (by omega) := by
  funext r c
  rw [Subsingleton.elim r 0]
  have := concat_right h (88 * l + c) (by omega)
  show T 0 ⟨88 * (22 + l) + c, by omega⟩ = y 0 ⟨88 * l + c, by omega⟩
  rw [← this]
  exact congrArg (T 0) (Fin.ext (by simp only; omega))

/-- The entry `a < 44` of the joined key row `r` is the key entry. -/
theorem key_row_entry {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {A0 A1 : ExactMatrix Q N 1 88}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    (r : Fin 2) (a : Fin 44) : ![A0, A1] r 0 ⟨a, by omega⟩ = A r 0 0 a := by
  fin_cases r
  · exact concat_left hA0 a a.isLt
  · exact concat_left hA1 a a.isLt

/-- The product key rows of one position: the top step from zero rows, then the key loop. -/
theorem pos_rows {A : Fin 2 → Fin 2 → ExactMatrix Q N 1 44} {A0 A1 : ExactMatrix Q N 1 88}
    (hA0 : concatColumns (A 0 0) (A 0 1) A0) (hA1 : concatColumns (A 1 0) (A 1 1) A1)
    {X : ExactMatrix Q N 1 88} {comp : Fin 2 → Fin 88 → Fin 44 → ExactPoly Q N}
    {t0 t1 f0 f1 : ExactMatrix Q N 1 88}
    (htop : ∀ (r : Fin 2) (w : Fin 88), ![t0, t1] r 0 w = ![(0 : ExactMatrix Q N 1 88), 0] r 0 w +
      ∑ a : Fin 44, ![A0, A1] r 0 ⟨a, by omega⟩ * rowDigits X a w)
    (hkey : ∀ (k : Fin 2) (jj : Fin 88), ![f0, f1] k 0 jj = ![t0, t1] k 0 jj + ∑ a, comp k jj a)
    (r : Fin 2) (jj : Fin 88) :
    ![f0, f1] r 0 jj = ∑ a, comp r jj a + ∑ a : Fin 44, A r 0 0 a * rowDigits X a jj := by
  have hz : ![(0 : ExactMatrix Q N 1 88), 0] r 0 jj = 0 := by fin_cases r <;> rfl
  have h1 := hkey r jj
  have h2 := htop r jj
  rw [hz, zero_add] at h2
  rw [h1, h2, add_comm]
  exact congrArg (fun x ↦ ∑ a, comp r jj a + x)
    (Finset.sum_congr rfl fun a _ ↦ by rw [key_row_entry hA0 hA1])

/-- The first-depth LUT key rows at block `b` and column `jj`. -/
theorem lut_T1 {model : HashModel} {key : ByteArray} {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    {A0 A1 : ExactMatrix Q N 1 88}
    (hA0 : concatColumns (M 0 0) (M 0 1) A0) (hA1 : concatColumns (M 1 0) (M 1 1) A1)
    {t0 t1 f0 f1 z : ExactMatrix Q N 1 1936} (hz : z = 0)
    (htop : ∀ (r : Fin 2) (p : Fin 1936), ![t0, t1] r 0 p = ![z, z] r 0 p +
      ∑ a : Fin 44, substitution (kSw ((p : Nat) / 88)) (kSw_odd ⟨(p : Nat) / 88, by omega⟩)
        (![A0, A1] r 0 ⟨a, by omega⟩) *
        rowDigits (switchTopRow model key ⟨(p : Nat) / 88, by omega⟩) a ⟨(p : Nat) % 88, by omega⟩)
    (hkey : ∀ (r : Fin 2) (p : Fin 1936), ![f0, f1] r 0 p = ![t0, t1] r 0 p +
      ∑ a, lutKeyCompanionSum model key ⟨(p : Nat) / 88, by omega⟩ M r ⟨(p : Nat) % 88, by omega⟩ a)
    (b : Fin 22) (r : Fin 2) (jj : Fin 88) :
    ![f0, f1] r 0 ⟨88 * (b : Nat) + jj, by omega⟩ =
      ∑ a, lutKeyCompanionSum model key b M r jj a +
      ∑ a : Fin 44, substitution (kSw b) (kSw_odd b) (M r 0 0 a) * rowDigits (switchTopRow model key b) a jj := by
  have h0 : ![z, z] r 0 ⟨88 * (b : Nat) + jj, by omega⟩ = 0 := by subst hz; fin_cases r <;> rfl
  rw [hkey, htop, h0, zero_add, add_comm]
  obtain ⟨bv, hb⟩ := b
  obtain ⟨jv, hj⟩ := jj
  have hdiv : (88 * bv + jv) / 88 = bv := by omega
  have hmod : (88 * bv + jv) % 88 = jv := by omega
  simp only [hdiv, hmod, key_row_entry hA0 hA1]

/-- The second-depth LUT key rows at leaf `l` and column `jj`, for parent keys `Kf`. -/
theorem lut_T2 {model : HashModel} {key : ByteArray} {K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44}
    {Kf : Fin 15 → Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    (hK : ∀ e : Fin 15, parentKeys K00 K01 K10 K11 e = Kf e)
    {t0 t1 f0 f1 z : ExactMatrix Q N 1 9240} (hz : z = 0)
    (htop : ∀ (r : Fin 2) (p : Fin 9240), ![t0, t1] r 0 p = ![z, z] r 0 p +
      ∑ a : Fin 44, substitution (kSw (15 + (p : Nat) / 88 / 15)) (kSw_odd (member2 ((p : Nat) / 88) (by omega)))
        (![K00, K10] r ⟨(p : Nat) / 88 % 15, by omega⟩ 0 a) *
        rowDigits (switchTopRow model key (member2 ((p : Nat) / 88) (by omega))) a ⟨(p : Nat) % 88, by omega⟩)
    (hkey : ∀ (r : Fin 2) (p : Fin 9240), ![f0, f1] r 0 p = ![t0, t1] r 0 p +
      ∑ a, lutKeyCompanionSum model key (member2 ((p : Nat) / 88) (by omega))
        (parentKeys K00 K01 K10 K11 ⟨(p : Nat) / 88 % 15, by omega⟩) r ⟨(p : Nat) % 88, by omega⟩ a)
    (l : Fin 105) (r : Fin 2) (jj : Fin 88) :
    ![f0, f1] r 0 ⟨88 * (l : Nat) + jj, by omega⟩ =
      ∑ a, lutKeyCompanionSum model key (member2 l l.isLt) (Kf ⟨l % 15, by omega⟩) r jj a +
      ∑ a : Fin 44, substitution (kSw (15 + l / 15)) (kSw_odd (member2 l l.isLt))
        (Kf ⟨l % 15, by omega⟩ r 0 0 a) * rowDigits (switchTopRow model key (member2 l l.isLt)) a jj := by
  have h0 : ![z, z] r 0 ⟨88 * (l : Nat) + jj, by omega⟩ = 0 := by subst hz; fin_cases r <;> rfl
  have hKr : ∀ e : Fin 15, ![K00, K10] r e = Kf e r 0 := by
    intro e
    rw [← hK e]
    fin_cases r <;> rfl
  rw [hkey, htop, h0, zero_add, add_comm]
  simp only [hKr, ← hK]
  obtain ⟨lv, hl⟩ := l
  obtain ⟨jv, hj⟩ := jj
  have hdiv : (88 * lv + jv) / 88 = lv := by omega
  have hmod : (88 * lv + jv) % 88 = jv := by omega
  simp only [hdiv, hmod]

/-- The output key of a step from its preprocessing sums. -/
theorem pre_sum {model : HashModel} {key : ByteArray} {M : Fin 2 → Fin 2 → ExactMatrix Q N 1 44}
    (d1 : BranchData) (d2 : Leaf2Data) (coef : Nat → ExactMatrix Q N 1 1)
    {x0 x1 : ExactMatrix Q N 1 1936} {y0 y1 : ExactMatrix Q N 1 9240}
    (hd1k : d1.key = key) (hd10 : concatColumns x0 y0 d1.T0) (hd11 : concatColumns x1 y1 d1.T1)
    (hd2k : d2.key = key) (hd20 : concatColumns x0 y0 d2.T0) (hd21 : concatColumns x1 y1 d2.T1)
    {K00 K01 K10 K11 : Fin 15 → ExactMatrix Q N 1 44}
    (hK : ∀ (b : Fin 15) (r j : Fin 2), ![![K00 b, K01 b], ![K10 b, K11 b]] r j =
      pairMul (helperKey model key (kSw b) r) (actionDigits (rowBlock x0 b (by omega)) (rowBlock x1 b (by omega)) j))
    {s0 s1 s2 s3 : KeyState}
    (h0 : ∀ r j, keyOf s0 r j = M r j * scalarV (coef 1))
    (h1 : ∀ r j, keyOf s1 r j = keyOf s0 r j +
      ∑ b : Fin 15, keyOf (K00 b, K01 b, K10 b, K11 b, ()) r j * scalarV (coef (kSw b)))
    (h2 : ∀ r j, keyOf s2 r j = keyOf s1 r j + ∑ i : Fin 7,
      pairMul (helperKey model key (kSw (15 + i)) r)
        (actionDigits (rowBlock x0 (15 + i) (by omega)) (rowBlock x1 (15 + i) (by omega)) j) *
        scalarV (coef (kSw (15 + i))))
    (h3 : ∀ r j, keyOf s3 r j = keyOf s2 r j + ∑ i : Fin 105,
      pairMul (helperKey model key (kSw (15 + i / 15)) r)
        (actionDigits (rowBlock y0 i (by omega)) (rowBlock y1 i (by omega)) j) * scalarV (coef (kLeaf2 i))) :
    keyOf s3 = stepKey model M d1 d2 coef := by
  funext r j
  unfold stepKey
  rw [h3, h2, h1, h0]
  have e1 : ∀ b : Fin 15, keyOf (K00 b, K01 b, K10 b, K11 b, ()) r j = d1.key1 model ⟨b, by omega⟩ r j := by
    intro b
    rw [show keyOf (K00 b, K01 b, K10 b, K11 b, ()) r j = ![![K00 b, K01 b], ![K10 b, K11 b]] r j from rfl, hK]
    unfold BranchData.key1 BranchData.Tb0 BranchData.Tb1
    rw [hd1k, rowBlock_concat_left hd10 b (by omega), rowBlock_concat_left hd11 b (by omega)]
  have e2 : ∀ i : Fin 7, pairMul (helperKey model key (kSw (15 + i)) r)
      (actionDigits (rowBlock x0 (15 + i) (by omega)) (rowBlock x1 (15 + i) (by omega)) j) =
      d1.key1 model ⟨15 + i, by omega⟩ r j := by
    intro i
    unfold BranchData.key1 BranchData.Tb0 BranchData.Tb1
    rw [hd1k, rowBlock_concat_left hd10 (15 + i) (by omega), rowBlock_concat_left hd11 (15 + i) (by omega)]
  have e3 : ∀ i : Fin 105, pairMul (helperKey model key (kSw (15 + i / 15)) r)
      (actionDigits (rowBlock y0 i (by omega)) (rowBlock y1 i (by omega)) j) = d2.key2 model i r j := by
    intro i
    unfold Leaf2Data.key2 Leaf2Data.Tb0 Leaf2Data.Tb1
    rw [hd2k, rowBlock_concat_right hd20 i (by omega), rowBlock_concat_right hd21 i (by omega)]
  simp only [e1, e2, e3]

end PeProof
