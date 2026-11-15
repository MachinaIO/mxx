import MxxCore.Algorithm

/-!
# A minimal protocol specification

One algorithm, `run`, that should return its input. It exists to exercise the claim generator
(`mxx_core::claims`) end to end; its manifest is `Identity.toml`.
-/

namespace MxxCore.Examples.Identity

open MathcalA MxxCore

structure Impl where
  run : Algorithm

/-- The length in bytes of the input. -/
abbrev Params := Nat

/-- Every run on an input of `p` bytes returns that input, except with probability at most `δ`. -/
def Correct (I : Impl) (p : Params) (δ : ℝ) : Prop :=
  ∀ x : Bytes, x.length = p → 1 - δ ≤ (Pr{let r ← I.run.run [x]}[r = some [x]]).toReal

end MxxCore.Examples.Identity
