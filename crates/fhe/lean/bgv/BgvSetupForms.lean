import Reductions
import BgvEncryptForms

set_option backward.isDefEq.respectTransparency false

/-! Closed forms of the generated simulator setups. -/

namespace BgvSecurity
open MxxSecurity Mxx.Primitives MxxRuntime

/-- Two rows stacked. -/
def rowsOf {q c : Nat} (x y : ExactMatrix q n 1 c) : ExactMatrix q n (1 + 1) c :=
  fun r j ↦ if r.val = 0 then x 0 j else y 0 j

theorem concatRows_rowsOf {q c : Nat} (x y : ExactMatrix q n 1 c) : concatRows x y (rowsOf x y) := by
  intro row column
  fin_cases row <;> rfl

theorem concatRows_eq_rowsOf {q c : Nat} {x y : ExactMatrix q n 1 c} {o : ExactMatrix q n (1 + 1) c}
    (h : concatRows x y o) : o = rowsOf x y := by
  funext row column
  rw [h row column]
  fin_cases row <;> rfl

theorem sim0_setup_eq (ka kb : ExactMatrix QP n 1 3) (pa pb : ExactMatrix Q n 1 1) :
    Sim_0_setup.generatedRoot.comp GeneratedScheme.Sim_0_setup_params (ka, kb, pa, pb, ()) =
      pure (rowsOf ka kb, rowsOf pa pb, ()) := by
  unfold Sim_0_setup.generatedRoot.comp
  congr 1
  refine determined_eq ⟨⟨rowsOf ka kb, rowsOf pa pb⟩, concatRows_rowsOf _ _, concatRows_rowsOf _ _,
    rfl⟩ ?_
  rintro outputs ⟨w, h0, h1, rfl⟩
  rw [concatRows_eq_rowsOf h0, concatRows_eq_rowsOf h1]

theorem sim1_setup_eq (a u : ExactMatrix Q n 1 1) :
    Sim_1_setup.generatedRoot.comp GeneratedScheme.Sim_1_setup_params (a, u, ()) =
      (do let ek ← (sampleResidue : Comp (ExactMatrix QP n 2 3))
          pure (ek, rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u), ())) := by
  unfold Sim_1_setup.generatedRoot.comp
  refine bind_congr fun ek ↦ ?_
  congr 1
  refine determined_eq ⟨⟨ek, rowsOf (matrixScale 1032193 a) (matrixScale 1032193 u)⟩, rfl, rfl,
    concatRows_rowsOf _ _, rfl⟩ ?_
  rintro outputs ⟨w, hw, -, h, rfl⟩
  rw [hw, concatRows_eq_rowsOf h]
end BgvSecurity
