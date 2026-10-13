import ExponentRuntime

/-!
# The label payloads for any parameter set

The online stage bakes label member `i` of the PRF's public input family `a ∈ Z_Q'^(ν+1)` in as
the constant `c_i X^(e_i)` (`c_i = ±1`, `0 ≤ e_i < 2 n`), which is the monomial `X^(s a_i)` in the
exponent layout of stride `s` (`s Q' = 2 n`). Entry `ν` is the zero member, `c = 1`, `e = 0`,
`a = 0`. The per-entry arithmetic is the closed predicate `LabelEntry`, decidable entry by entry,
and `label_value_gen` turns it into the payload equation for every member of a table of any
length.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-- The closed per-entry predicate of a label table: `c X^e = X^(s a)` because `c = ±1`,
`0 ≤ e < 2 n` and `2 e + (1 - c) n = 2 s a`. -/
abbrev LabelEntry (n s : Nat) (c e : Int) (a : Nat) : Prop :=
  (c = 1 ∨ c = -1) ∧ 0 ≤ e ∧ e < 2 * (n : Int) ∧ 2 * e + (1 - c) * (n : Int) = 2 * ((s * a : Nat) : Int)

/-- The zero member: `1 · X^0 = X^(s · 0)`. -/
theorem labelEntry_zero (n s : Nat) (hn : 0 < n) : LabelEntry n s 1 0 0 := by
  refine ⟨Or.inl rfl, le_rfl, by omega, by simp⟩

/-- A label entry's plaintext is its monomial. -/
theorem label_message_gen {s : Nat} {c e : Int} {a : Nat} (h : LabelEntry n s c e a) :
    message (n := n) c e = Xe (n := n) (s * a) := by
  obtain ⟨hc, he0, he1, hr⟩ := h
  unfold message Xe
  have hmod : (e % (2 * n : Nat) : Int).toNat = e.toNat := by
    rw [Int.emod_eq_of_lt he0 (by push_cast; omega)]
  rw [hmod]
  rcases hc with rfl | rfl
  · have : s * a = e.toNat := by omega
    rw [this]; simp
  · have : s * a = e.toNat + n := by omega
    rw [this, pow_add, Negacyclic.root_pow_n]; simp

/-- **The label table.** If every entry of the table `(c, e)` satisfies `LabelEntry` against the
label exponents `a` (a closed predicate, decidable entry by entry), every member's payload is
`X^(s a_i)`. -/
theorem label_value_gen {q s ν : Nat} (c e : Fin (ν + 1) → Int) (a : Fin (ν + 1) → Nat)
    (h : ∀ i, LabelEntry n s (c i) (e i) (a i)) (i : Fin (ν + 1)) :
    multiplyMonomial (liftInteger (c i) : ExactMatrix q n 1 1) (e i) =
      fun _ _ ↦ reducePoly q n (Xe (n := n) (s * a i)) := by
  rw [multiplyMonomial_liftInteger, label_message_gen (h i)]

end RefreshTopProof.Gen
