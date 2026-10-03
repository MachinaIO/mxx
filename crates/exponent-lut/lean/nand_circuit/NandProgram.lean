import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Linarith

/-!
# The depth-one NAND program

`NandCircuit::compile` turns one NAND gate into four binary LUT rounds over the packed register
state `R₀ + 2 R₁ + 4 R₂`, starting from `(0, 1, 0)`, i.e. exponent `2`. Round `r` multiplies the
state monomial by the RHS monomial `X^(8 b)` of circuit input `[0, 1, 0, 1][r]` and applies its
table to the exponent; the last table keeps one bit. The tables are those of the exported
preprocessing graph, where they appear as the outputs `X^(T k)` of the LUT coefficients.
-/

namespace MxxExponentLutNand

/-- The four LUT tables, in round order. -/
def table : Fin 4 → Fin 16 → Nat :=
  ![![0, 1, 3, 2, 4, 5, 7, 6, 0, 1, 7, 6, 4, 5, 3, 2],
    ![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 5, 4, 7, 6],
    ![0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 6, 7, 4, 5, 2, 3],
    ![0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0]]

/-- The circuit input read by the RHS of each round. -/
def roundInput : Fin 4 → Fin 2 := ![0, 1, 0, 1]

/-- The initial packed state `(0, 1, 0)`. -/
def initialExponent : Nat := 2

/-- A table on natural-number exponents, zero outside its domain. -/
def lookup (r : Fin 4) (k : Nat) : Nat := if h : k < 16 then table r ⟨k, h⟩ else 0

/-- The state exponent entering round `r` (after its RHS multiplication) for input bits `b`. -/
def roundArgument (b : Fin 2 → Nat) : Fin 4 → Nat
  | 0 => initialExponent + 8 * b (roundInput 0)
  | 1 => lookup 0 (roundArgument b 0) + 8 * b (roundInput 1)
  | 2 => lookup 1 (roundArgument b 1) + 8 * b (roundInput 2)
  | 3 => lookup 2 (roundArgument b 2) + 8 * b (roundInput 3)

/-- The exponent of the output monomial. -/
def outputExponent (b : Fin 2 → Nat) : Nat := lookup 3 (roundArgument b 3)

/-- For Boolean inputs every round argument lies in the LUT domain, and the output exponent is the
NAND of the inputs. -/
theorem rounds_nand (b : Fin 2 → Nat) (hb : ∀ i, b i ≤ 1) :
    (∀ r, roundArgument b r < 16) ∧ outputExponent b = 1 - b 0 * b 1 := by
  have h0 : b 0 = 0 ∨ b 0 = 1 := by have := hb 0; omega
  have h1 : b 1 = 0 ∨ b 1 = 1 := by have := hb 1; omega
  rcases h0 with h0 | h0 <;> rcases h1 with h1 | h1 <;>
  · refine ⟨fun r ↦ ?_, ?_⟩
    · match r with
      | 0 | 1 | 2 | 3 =>
        simp (config := { decide := true }) [roundArgument, roundInput, lookup, table,
          initialExponent, h0, h1]
    · simp (config := { decide := true }) [outputExponent, roundArgument, roundInput, lookup,
        table, initialExponent, h0, h1]

end MxxExponentLutNand
