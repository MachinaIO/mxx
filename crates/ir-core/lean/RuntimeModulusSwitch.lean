import RuntimeMatrixOps

namespace MxxRuntime

open Mxx.Primitives

/-- Before destination reduction, exact canonical-residue rounding introduces at
most half the odd switching factor in scaled integer error. -/
theorem roundedCanonical_scaled_error (a factor destination : Int)
    (hf : 0 < factor) (hp : 0 < destination)
    (hodd : factor % 2 = 1) (hpodd : destination % 2 = 1) :
    (let rounded := (a * destination + factor * destination / 2) / (factor * destination);
      -(factor / 2) ≤ factor * rounded - a ∧
        factor * rounded - a ≤ factor / 2) := by
  dsimp
  let rounded := (a * destination + factor * destination / 2) / (factor * destination)
  let remainder := (a * destination + factor * destination / 2) % (factor * destination)
  have hfactor : factor = 2 * (factor / 2) + 1 := by omega
  have hdest : destination = 2 * (destination / 2) + 1 := by omega
  have hprod : 0 < factor * destination := mul_pos hf hp
  have hrem0 : 0 ≤ remainder := Int.emod_nonneg _ (ne_of_gt hprod)
  have hremLt : remainder < factor * destination := Int.emod_lt_of_pos _ hprod
  have hdivision : factor * destination * rounded + remainder =
      a * destination + factor * destination / 2 := by
    exact Int.mul_ediv_add_emod _ _
  have hhalf : factor * destination / 2 = (factor / 2) * destination + destination / 2 := by
    have heq : factor * destination =
        2 * ((factor / 2) * destination + destination / 2) + 1 := by nlinarith
    omega
  have hdesthalf : 0 ≤ destination / 2 ∧ destination / 2 < destination := by omega
  change -(factor / 2) ≤ factor * rounded - a ∧ factor * rounded - a ≤ factor / 2
  constructor
  · by_contra h
    have hbad : factor * rounded - a ≤ -(factor / 2) - 1 := by omega
    have := mul_le_mul_of_nonneg_right hbad (le_of_lt hp)
    nlinarith
  · by_contra h
    have hbad : factor / 2 + 1 ≤ factor * rounded - a := by omega
    have := mul_le_mul_of_nonneg_right hbad (le_of_lt hp)
    nlinarith

/-- The exported runtime relation determines the actual rounded coefficient. -/
theorem modulusSwitchRuns_coeff {q p n rows columns : Nat}
    (hn : 0 < n) (input : ExactMatrix q n rows columns)
    (output : ExactMatrix p n rows columns) (runs : modulusSwitchRuns input output)
    (row : Fin rows) (column : Fin columns) (i : Fin n) :
    (output row column).coeff i =
      ((((Int.ofNat ((input row column).coeff i).val * (p : Int) + Int.ofNat (q / 2)) /
        Int.ofNat q) % (p : Int) : Int) : ZMod p) := by
  rcases runs with ⟨hp, _, _, _, rfl⟩
  exact polynomialOfCoefficients_coeff hp hn _ i

/-- A successful exported switch has a concrete integer lift of each output
coefficient whose scaled error is bounded by the odd factor's half-width. -/
theorem modulusSwitchRuns_rounding_error {q p n rows columns factor : Nat}
    (hn : 0 < n) (hf : 0 < factor) (hq : q = factor * p)
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns)
    (runs : modulusSwitchRuns input output)
    (row : Fin rows) (column : Fin columns) (i : Fin n) :
    ∃ rounded : Int, (output row column).coeff i = (rounded : ZMod p) ∧
      -((factor : Int) / 2) ≤ (factor : Int) * rounded -
        Int.ofNat ((input row column).coeff i).val ∧
      (factor : Int) * rounded - Int.ofNat ((input row column).coeff i).val ≤
        (factor : Int) / 2 := by
  have hp : 0 < p := by have := runs.1; omega
  have hfactorodd : factor % 2 = 1 := by
    have ho := runs.2.2.1
    rw [hq, Nat.mul_mod, runs.2.2.2.1, Nat.mul_one] at ho
    omega
  have bounds := roundedCanonical_scaled_error
    (Int.ofNat ((input row column).coeff i).val) ((factor : Int)) ((p : Int))
    (by exact_mod_cast hf) (by exact_mod_cast hp)
    (by exact_mod_cast hfactorodd) (by exact_mod_cast runs.2.2.2.1)
  refine ⟨_, ?_, bounds⟩
  have coefficient := modulusSwitchRuns_coeff hn input output runs row column i
  rw [coefficient]
  simp [hq]

end MxxRuntime
