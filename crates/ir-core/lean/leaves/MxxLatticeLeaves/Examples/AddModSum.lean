import MxxCore.Algorithm

/-!
# A specification with one lattice operation

One algorithm, `run`, that should return the coordinatewise sum modulo `q` of two vectors of
`n` residues, each encoded as little-endian 64-bit words. It exercises the composition of glue
with a leaf (`lattice.add_mod`) in a proof image; its manifest is `AddModSum.toml`.
-/

namespace MxxLatticeLeaves.Examples.AddModSum

open MathcalA MxxCore

structure Impl where
  run : Algorithm

structure Params where
  n : Nat
  q : Nat

/-- The sum `(A k + B k) mod q` of two vectors. -/
def sum (q : Nat) (A B : List Nat) : List Nat := List.zipWith (fun a b => (a + b) % q) A B

/-- Every run on two vectors of `n` residues below `q` returns their sum, except with probability
at most `δ`. -/
def Correct (I : Impl) (p : Params) (δ : ℝ) : Prop :=
  ∀ A B : List Nat, A.length = p.n → B.length = p.n → (∀ a ∈ A, a < p.q) → (∀ b ∈ B, b < p.q) →
    1 - δ ≤
      (Pr{let r ← I.run.run [Bytes.u64s A, Bytes.u64s B]}[r = some [Bytes.u64s (sum p.q A B)]]).toReal

end MxxLatticeLeaves.Examples.AddModSum
