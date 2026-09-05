import ExponentBounds

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Native signed monomial input contract, with the exponent including the
negacyclic sign bit. This is an equality in the actual residue ring. -/
def SignedMonomial {q n : Nat} (value : ExactPoly q n) : Prop :=
  ∃ exponent : Fin (2 * n), value =
    AdjoinRoot.root (negacyclicModulus n (ZMod q)) ^ exponent.val

private theorem root_power_remainder {n : Nat} {R : Type} [CommRing R] (exponent : Nat) :
    AdjoinRoot.root (negacyclicModulus n R) ^ exponent =
      algebraMap Int (Negacyclic n R) ((-1 : Int) ^ (exponent / n)) *
        AdjoinRoot.root (negacyclicModulus n R) ^ (exponent % n) := by
  conv_lhs => rw [← Nat.div_add_mod exponent n]
  rw [pow_add, pow_mul, Negacyclic.root_pow_n]
  simp

/-- Every allowed input has an explicit integer signed basis witness with
unit polynomial L1. The norm conclusion is proved from the ring expression. -/
theorem signedMonomial_integer_lift {q n : Nat} (hn : 0 < n)
    {value : ExactPoly q n} (hvalue : SignedMonomial value) :
    ∃ lift : ErrorPoly n, reducePoly q n lift = value ∧ polynomialL1 lift ≤ 1 := by
  obtain ⟨exponent, heq⟩ := hvalue
  let index : Fin n := ⟨exponent.val % n, Nat.mod_lt _ hn⟩
  let sign : Int := (-1) ^ (exponent.val / n)
  refine ⟨scaledBasis sign index, ?_, ?_⟩
  · rw [heq, root_power_remainder]
    simp [scaledBasis, reducePoly, sign, index]
  · rw [polynomialL1_scaledBasis hn]
    simp [sign, Int.natAbs_pow]

private theorem root_power_period {n : Nat} {R : Type} [CommRing R] :
    AdjoinRoot.root (negacyclicModulus n R) ^ (2 * n) = 1 := by
  rw [Nat.mul_comm, pow_mul, Negacyclic.root_pow_n]
  ring

private theorem root_power_mod {n : Nat} {R : Type} [CommRing R] (exponent : Nat) :
    AdjoinRoot.root (negacyclicModulus n R) ^ exponent =
      AdjoinRoot.root (negacyclicModulus n R) ^ (exponent % (2 * n)) := by
  conv_lhs => rw [← Nat.div_add_mod exponent (2 * n)]
  rw [pow_add, pow_mul, root_power_period]
  simp

/-- Native multiplication adds exponents, including negacyclic wraparound. -/
theorem SignedMonomial.mul {q n : Nat} (hn : 0 < n) {left right : ExactPoly q n}
    (hl : SignedMonomial left) (hr : SignedMonomial right) : SignedMonomial (left * right) := by
  obtain ⟨a, ha⟩ := hl
  obtain ⟨b, hb⟩ := hr
  refine ⟨⟨(a.val + b.val) % (2 * n), Nat.mod_lt _ (by omega)⟩, ?_⟩
  rw [ha, hb, ← pow_add, root_power_mod]

theorem signedMonomial_one {q n : Nat} (hn : 0 < n) :
    SignedMonomial (1 : ExactPoly q n) := by
  exact ⟨⟨0, by omega⟩, by simp⟩

/-- Multiplication by an actual Boolean selector either vanishes or retains
the input signed monomial; zero is not asserted to be a signed monomial. -/
theorem selector_mul_monomial {q n : Nat} {selector value : ExactPoly q n}
    (hselector : selector = 0 ∨ selector = 1) (hvalue : SignedMonomial value) :
    selector * value = 0 ∨ SignedMonomial (selector * value) := by
  rcases hselector with hzero | hone
  · exact Or.inl (by rw [hzero, zero_mul])
  · exact Or.inr (by simpa [hone] using hvalue)

/-- Inactive selector branches also satisfy the gate's unit-L1 contract via
the explicit zero integer polynomial. -/
theorem selector_mul_integer_lift {q n : Nat} (hn : 0 < n)
    {selector value : ExactPoly q n}
    (hselector : selector = 0 ∨ selector = 1) (hvalue : SignedMonomial value) :
    ∃ lift : ErrorPoly n,
      reducePoly q n lift = selector * value ∧ polynomialL1 lift ≤ 1 := by
  rcases selector_mul_monomial hselector hvalue with hzero | hmonomial
  · refine ⟨0, ?_, ?_⟩
    · rw [hzero, map_zero]
    · simp [polynomialL1]
  · exact signedMonomial_integer_lift hn hmonomial

/-- Actual one-hot coefficients select exactly one member of the finite
family, so the evaluator's sum retains a signed monomial. -/
theorem one_hot_sum_monomial {q n count : Nat}
    (selector values : Fin count → ExactPoly q n) (selected : Fin count)
    (hselected : selector selected = 1)
    (hothers : ∀ i, i ≠ selected → selector i = 0)
    (hvalue : SignedMonomial (values selected)) :
    SignedMonomial (∑ i, selector i * values i) := by
  rw [Finset.sum_eq_single selected]
  · simpa [hselected] using hvalue
  · intro i hi hne
    rw [hothers i hne, zero_mul]
  · simp

/-- The selected sum supplies the exact lift needed by private and public
gate transfers, with no assumed monomial or error conclusion for the sum. -/
theorem one_hot_sum_integer_lift {q n count : Nat} (hn : 0 < n)
    (selector values : Fin count → ExactPoly q n) (selected : Fin count)
    (hselected : selector selected = 1)
    (hothers : ∀ i, i ≠ selected → selector i = 0)
    (hvalue : SignedMonomial (values selected)) :
    ∃ lift : ErrorPoly n,
      reducePoly q n lift = ∑ i, selector i * values i ∧ polynomialL1 lift ≤ 1 :=
  signedMonomial_integer_lift hn
    (one_hot_sum_monomial selector values selected hselected hothers hvalue)

end MxxExponentLut
