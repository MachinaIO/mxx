import GenSmudge
import GenFinalFresh
import RuntimeAllCoefficients

/-! Parameter-independent smudge polynomial lemmas, extracted from the original toy proof. -/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenFinal
open scoped BigOperators

/-- One digit specializes `GenSmudge` to the integer half and its explicit bound. -/
theorem smudge_one_digit_bound {P : Nat} (hP : P % 2 = 0) (digit : Int)
    (hodd : digit % 2 = 1) (hbound : |digit| ≤ (P : Int) - 1) :
    ((digit - 1) / 2).natAbs ≤ P / 2 := by
  simpa [smudgeInt] using
    smudge_natAbs_le (d := 0) hP (fun _ ↦ digit) (fun _ ↦ hodd) (fun _ ↦ hbound)

/-- The one-digit target is twice the integer smudge plus one, with weight `P^0 = 1`. -/
theorem smudge_one_digit_two_mul {P : Nat} (hP : P % 2 = 0) (digit : Int)
    (hodd : digit % 2 = 1) : digit = 2 * ((digit - 1) / 2) + 1 := by
  simpa [smudgeInt] using smudge_two_mul (d := 0) hP (fun _ ↦ digit) hodd

/-- Subtracting a modular half from an odd scalar readout gives its integer half. -/
theorem smudge_one_digit_value {q n : Nat} {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Int)
    (hodd : digit % 2 = 1) :
    reducePoly q n (digit : ErrorPoly n) * (half : ExactPoly q n) - half =
      reducePoly q n (((digit - 1) / 2 : Int) : ErrorPoly n) := by
  calc
    _ = reducePoly q n ((half * digit - half : Int) : ErrorPoly n) := by
      simp only [reducePoly_intCast]
      push_cast
      ring
    _ = _ := half_digit_gen hhalf digit hodd

/-- The coefficientwise centered one-digit smudge polynomial. -/
noncomputable def centeredSmudge {n : Nat} (digit : Fin n → Int) : ErrorPoly n :=
  intPoly (fun coefficient ↦ (digit coefficient - 1) / 2)

/-- The centered polynomial has coefficient bound `P / 2`, independently of ring dimension. -/
theorem centeredSmudge_bound {n P : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    (digit : Fin n → Int) (hodd : ∀ coefficient, digit coefficient % 2 = 1)
    (hbound : ∀ coefficient, |digit coefficient| ≤ (P : Int) - 1) :
    ∀ coefficient, ((centeredSmudge digit).coeff coefficient).natAbs ≤ P / 2 := by
  intro coefficient
  rw [centeredSmudge, intPoly_coeff hn]
  exact smudge_one_digit_bound hP _ (hodd _) (hbound _)

/-- Coefficientwise half subtraction reduces to the small integer smudge polynomial. -/
theorem smudge_coefficientwise_offset {q n : Nat} {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Fin n → Int)
    (hodd : ∀ coefficient, digit coefficient % 2 = 1) :
    reducePoly q n (intPoly digit) * (half : ExactPoly q n) -
        reducePoly q n (intPoly (fun _ ↦ half)) =
      reducePoly q n (centeredSmudge digit) := by
  simp only [intPoly, centeredSmudge, map_sum, map_mul]
  rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro coefficient _
  have hone := smudge_one_digit_value (n := n) hhalf (digit coefficient) (hodd coefficient)
  simp only [reducePoly_intCast] at hone
  calc
    _ = ((digit coefficient : ExactPoly q n) * half - half) *
        reducePoly q n (Xe coefficient.val) := by
      simp only [reducePoly_intCast]
      ring
    _ = _ := by rw [hone, reducePoly_intCast]

/-- Coefficients of a constant polynomial, including the zero coefficient. -/
theorem smudge_constant_coeff {n : Nat} (hn : 0 < n) (value : Int) (coefficient : Fin n) :
    ((value : ErrorPoly n)).coeff coefficient = if coefficient.val = 0 then value else 0 := by
  have hone := coeff_root_pow_lt (n := n) hn (w := 0) hn coefficient
  rw [pow_zero] at hone
  have hvalue := intCast_mul_coeff value (1 : ErrorPoly n) coefficient
  rw [mul_one, hone] at hvalue
  rw [hvalue]
  split_ifs <;> simp

/-- The residual left by subtracting the scalar half instead of coefficientwise halves. -/
noncomputable def scalarHalfResidual (n : Nat) (half : Int) : ErrorPoly n :=
  intPoly (fun coefficient ↦ if coefficient.val = 0 then 0 else half)

/-- The residual is the difference between coefficientwise and scalar offsets. -/
theorem scalarHalfResidual_eq {n : Nat} (hn : 0 < n) (half : Int) :
    scalarHalfResidual n half = intPoly (fun _ ↦ half) - (half : ErrorPoly n) := by
  apply negacyclic_ext hn
  intro coefficient
  rw [scalarHalfResidual, intPoly_coeff hn, Negacyclic.coeff_sub, intPoly_coeff hn,
    smudge_constant_coeff hn]
  split_ifs <;> simp

/-- The actual scalar-offset graph has the centered smudge plus a nonconstant half residual. -/
theorem smudge_scalar_offset {q n : Nat} (hn : 0 < n) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1) (digit : Fin n → Int)
    (hodd : ∀ coefficient, digit coefficient % 2 = 1) :
    reducePoly q n (intPoly digit) * (half : ExactPoly q n) - half =
      reducePoly q n (centeredSmudge digit + scalarHalfResidual n half) := by
  rw [map_add, scalarHalfResidual_eq hn, map_sub, reducePoly_intCast,
    ← smudge_coefficientwise_offset hhalf digit hodd]
  ring

/-- The scalar-offset witness is bounded, but includes the half-modulus residual. -/
theorem smudge_scalar_offset_bound {n P : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    (half : Int) (digit : Fin n → Int) (hodd : ∀ coefficient, digit coefficient % 2 = 1)
    (hbound : ∀ coefficient, |digit coefficient| ≤ (P : Int) - 1) :
    ∀ coefficient,
      ((centeredSmudge digit + scalarHalfResidual n half).coeff coefficient).natAbs ≤
        P / 2 + half.natAbs := by
  intro coefficient
  rw [Negacyclic.coeff_add]
  apply (Int.natAbs_add_le _ _).trans
  apply Nat.add_le_add (centeredSmudge_bound hn hP digit hodd hbound coefficient)
  rw [scalarHalfResidual, intPoly_coeff hn]
  split_ifs <;> simp

/-- The all-coefficients runtime constant is the reduction of the corresponding integer sum. -/
theorem allCoefficients_reduce {q n rows columns : Nat} (half : Int) :
    (allCoefficients half : ExactMatrix q n rows columns) =
      fun _ _ ↦ reducePoly q n (intPoly (fun _ ↦ half)) := by
  funext row column
  exact polynomialOfCoefficients_eq_reduce _

end Dio2Proof
