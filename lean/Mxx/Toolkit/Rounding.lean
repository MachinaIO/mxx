import Mathlib.Algebra.Order.Round
import Mathlib.Data.Rat.Floor

namespace Mxx.Toolkit

def roundRat (value : Rat) : Int := round value

theorem integer_rounding_exact (value : Int) : roundRat value = value := by
  simp [roundRat]

theorem rounding_decomposition (value : Rat) :
    (roundRat value : Rat) = value + ((roundRat value : Rat) - value) := by ring

theorem rounding_error_bound (value : Rat) :
  |(roundRat value : Rat) - value| ≤ (1 : Rat) / 2 := by
  simpa [roundRat, abs_sub_comm] using (abs_sub_round value)

theorem scaled_rounding_error_bound (numerator denominator value : Rat) :
    |(roundRat ((numerator / denominator) * value) : Rat) -
        (numerator / denominator) * value| ≤ (1 : Rat) / 2 :=
  rounding_error_bound ((numerator / denominator) * value)

end Mxx.Toolkit
