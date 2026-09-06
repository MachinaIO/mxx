import ExponentRefreshCrt

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Adding the public half offset to the antiperiodic native trace produces
an integer digit. The second half of the domain selects one minus the table
entry, rather than indexing a fictitious second half of the table. -/
def refreshNativeDigit {width : Nat} (table : Fin width → Int)
    (exponent : Fin (2 * width)) : Int :=
  if h : exponent.val < width then table ⟨exponent.val, h⟩
  else 1 - table ⟨exponent.val - width, by omega⟩

theorem refresh_native_digit_lift {q n width : Nat}
    (table : Fin width → Int) (exponent : Fin (2 * width))
    (half : ExactPoly q n) (halfCorrect : half * 2 = 1) :
    (if h : exponent.val < width then (table ⟨exponent.val, h⟩ : ExactPoly q n) - half
      else half - (table ⟨exponent.val - width, by omega⟩ : ExactPoly q n)) + half =
        (refreshNativeDigit table exponent : ExactPoly q n) := by
  unfold refreshNativeDigit
  split_ifs
  · abel
  · push_cast
    linear_combination halfCorrect

/-- The actual native table need only satisfy the signed digit interval.
No Boolean-valued premise is used. This interval is invariant under d ↦ 1-d. -/
theorem refresh_native_digit_range {width : Nat} (base : Nat)
    (table : Fin width → Int) (exponent : Fin (2 * width))
    (tableRange : ∀ i, 1 - (base / 2 : Nat) ≤ table i ∧ table i ≤ (base / 2 : Nat)) :
    1 - (base / 2 : Nat) ≤ refreshNativeDigit table exponent ∧
      refreshNativeDigit table exponent ≤ (base / 2 : Nat) := by
  unfold refreshNativeDigit
  split_ifs with h
  · exact tableRange _
  · have := tableRange ⟨exponent.val - width, by omega⟩
    omega

/-- Base-p positional weights telescope to the simulator's p^D-1 support. -/
theorem refresh_radix_budget (base count : Nat) (hb : 1 ≤ base) :
    (∑ i : Fin count, base ^ i.val * (base - 1)) = base ^ count - 1 := by
  induction count with
  | zero => simp
  | succ count ih =>
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    rw [ih, pow_succ]
    have hp : 1 ≤ base ^ count := Nat.one_le_pow count base hb
    have hm := Nat.sub_add_cancel hb
    have hp' := Nat.sub_add_cancel hp
    have product : base ^ count * (base - 1) + base ^ count = base ^ count * base := by nlinarith
    omega

/-- Every nonnegative radix digit tuple lies in the full positional interval. -/
theorem refresh_unsigned_radix_range {count : Nat} (base : Nat) (hb : 1 ≤ base)
    (digit : Fin count → Int)
    (support : ∀ i, 0 ≤ digit i ∧ digit i ≤ (base - 1 : Nat)) :
    0 ≤ ∑ i : Fin count, (base : Int) ^ i.val * digit i ∧
      (∑ i : Fin count, (base : Int) ^ i.val * digit i) ≤ (base ^ count - 1 : Nat) := by
  constructor
  · exact Finset.sum_nonneg fun i _ ↦ mul_nonneg (by positivity) (support i).1
  · have h := Finset.sum_le_sum (s := Finset.univ) fun i _ ↦
      mul_le_mul_of_nonneg_left (support i).2 (show 0 ≤ (base : Int) ^ i.val by positivity)
    have budget := refresh_radix_budget base count hb
    exact h.trans_eq (by exact_mod_cast budget)

/-- The public mask offset subtracts floor(p^D/2) after shifting every native
signed digit by its minimum. This yields exactly the simulator's mask bound. -/
theorem refresh_mask_radix_bound {count : Nat} (base : Nat) (hb : 2 ≤ base)
    (even : base % 2 = 0) (digit : Fin count → Int)
    (support : ∀ i, 1 - (base / 2 : Nat) ≤ digit i ∧ digit i ≤ (base / 2 : Nat)) :
    ((∑ i : Fin count, (base : Int) ^ i.val * (digit i - (1 - (base / 2 : Nat)))) -
      ((base ^ count / 2 : Nat) : Int)).natAbs ≤ base ^ count / 2 := by
  have halves := Nat.mod_add_div base 2
  have unsigned : ∀ i, 0 ≤ digit i - (1 - (base / 2 : Nat)) ∧
      digit i - (1 - (base / 2 : Nat)) ≤ (base - 1 : Nat) := by
    intro i
    have := support i
    omega
  have range := refresh_unsigned_radix_range base (by omega)
    (fun i ↦ digit i - (1 - (base / 2 : Nat))) unsigned
  dsimp only at range
  have split := Nat.mod_add_div (base ^ count) 2
  have rem := Nat.mod_lt (base ^ count) (by decide : 0 < 2)
  have positive : 1 ≤ base ^ count := Nat.one_le_pow count base (by omega)
  rw [← Nat.cast_le (α := Int), Int.natCast_natAbs, abs_le]
  omega

/-- Fresh digits retain their signs. Their positional absolute bound is the
existing conservative p^D-1 bound, with no extra CRT or dimension factor. -/
theorem refresh_fresh_radix_bound {count : Nat} (base : Nat) (hb : 2 ≤ base)
    (digit : Fin count → Int)
    (support : ∀ i, 1 - (base / 2 : Nat) ≤ digit i ∧ digit i ≤ (base / 2 : Nat)) :
    (∑ i : Fin count, (base : Int) ^ i.val * digit i).natAbs ≤ base ^ count - 1 := by
  have bounded : ∀ i, (digit i).natAbs ≤ base - 1 := by
    intro i
    rw [← Nat.cast_le (α := Int), Int.natCast_natAbs, abs_le]
    have := support i
    have := Nat.div_mul_le_self base 2
    omega
  apply refresh_readout_support digit id (fun i ↦ (base : Int) ^ i.val) bounded
  simpa using (refresh_radix_budget base count (by omega)).le

/-- Assemble the actual selected native digits coefficientwise using the same
integer lift as the CRT adapter. Exponents remain the supplied readout choices. -/
theorem refresh_native_fresh_lift_bound {n columns width count : Nat}
    (hn : 0 < n) (base : Nat) (hb : 2 ≤ base) (table : Fin width → Int)
    (exponent : Fin 1 → Fin columns → Fin n → Fin count → Fin (2 * width))
    (tableRange : ∀ i, 1 - (base / 2 : Nat) ≤ table i ∧ table i ≤ (base / 2 : Nat)) :
    CoeffBound (refreshFreshLift fun row column coefficient ↦
      ∑ i : Fin count, (base : Int) ^ i.val * refreshNativeDigit table (exponent row column coefficient i))
      (base ^ count - 1) := by
  apply refreshFreshLift_bounded hn
  intro row column coefficient
  exact refresh_fresh_radix_bound base hb _ fun i ↦
    refresh_native_digit_range base table (exponent row column coefficient i) tableRange

/-- Coefficientwise centered masks use the same selected exponents as the
readout proof; the public offset accounts for both the minimum digit and center. -/
theorem refresh_native_mask_lift_bound {n columns width count : Nat}
    (hn : 0 < n) (base : Nat) (hb : 2 ≤ base) (even : base % 2 = 0)
    (table : Fin width → Int)
    (exponent : Fin 1 → Fin columns → Fin n → Fin count → Fin (2 * width))
    (tableRange : ∀ i, 1 - (base / 2 : Nat) ≤ table i ∧ table i ≤ (base / 2 : Nat)) :
    CoeffBound (refreshFreshLift fun row column coefficient ↦
      (∑ i : Fin count, (base : Int) ^ i.val *
        (refreshNativeDigit table (exponent row column coefficient i) - (1 - (base / 2 : Nat)))) -
          ((base ^ count / 2 : Nat) : Int)) (base ^ count / 2) := by
  apply refreshFreshLift_bounded hn
  intro row column coefficient
  exact refresh_mask_radix_bound base hb even _ fun i ↦
    refresh_native_digit_range base table (exponent row column coefficient i) tableRange

/-- The scalar integer digit is also its canonical reduction as an integer
polynomial, providing the lift expected by the routing equations. -/
theorem refresh_native_digit_reduction {q n width : Nat}
    (table : Fin width → Int) (exponent : Fin (2 * width))
    (half : ExactPoly q n) (halfCorrect : half * 2 = 1) :
    reducePoly q n (refreshNativeDigit table exponent : ErrorPoly n) =
      (if h : exponent.val < width then (table ⟨exponent.val, h⟩ : ExactPoly q n) - half
        else half - (table ⟨exponent.val - width, by omega⟩ : ExactPoly q n)) + half := by
  rw [refresh_native_digit_lift table exponent half halfCorrect]
  simp

end MxxExponentLut
