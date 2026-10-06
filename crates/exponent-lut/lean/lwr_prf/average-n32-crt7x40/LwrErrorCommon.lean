import LwrPreprocessE
import ExponentErrorForm

/-!
# Explicit wires of the online evaluation

The digits of a stacked pair and the gadgets as functions of their inputs (`theOutput` of the
runtime relations), so that the keys and digits of every cell are functions of the hash model and
the preprocessing outputs.
-/

namespace LwrProof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {tape : SampleTape} {path cellPath : List Nat}

/-- The digits of the stacked pair `(x, y)`: its gadget decomposition, rebased to `q`. -/
noncomputable def stackDigits (q : Nat) {q' cols : Nat} (inner : Nat) (bound : Nat) (base digits : Int)
    (x y : ExactMatrix q' N 1 cols) : ExactMatrix q N (inner + inner) cols :=
  theOutput (compactCenteredRebaseRuns bound)
    (theOutput (gadgetDecomposeRuns Backend.backend base digits)
      (theOutput (fun p : ExactMatrix q' N 1 cols × ExactMatrix q' N 1 cols ↦ concatRows p.1 p.2) (x, y) :
        ExactMatrix q' N (1 + 1) cols) : ExactMatrix q' N (inner + inner) cols)

theorem stackDigits_eq {q q' cols inner bound : Nat} {base digits : Int}
    {x y : ExactMatrix q' N 1 cols} {X : ExactMatrix q' N (1 + 1) cols}
    {D' : ExactMatrix q' N (inner + inner) cols} {D : ExactMatrix q N (inner + inner) cols}
    (hX : concatRows x y X) (hD' : gadgetDecomposeRuns Backend.backend base digits X D')
    (hD : compactCenteredRebaseRuns bound D' D) :
    stackDigits q inner bound base digits x y = D := by
  unfold stackDigits
  rw [theOutput_eq (R := fun p : ExactMatrix q' N 1 cols × ExactMatrix q' N 1 cols ↦ concatRows p.1 p.2)
    (fun _ h ↦ concatRows_unique h hX) hX]
  rw [theOutput_eq (fun _ h ↦ gadgetDecomposeRuns_unique h hD') hD']
  exact theOutput_eq (fun _ h ↦ compactCenteredRebaseRuns_unique h hD) hD

/-- The gadget of `m` digits of base `2^20` at the modulus `q`. -/
noncomputable def gadgetOf (q m : Nat) : ExactMatrix q N 1 m :=
  theOutput (fun (_ : Unit) (G : ExactMatrix q N 1 m) ↦ gadgetMatrixRuns Backend.backend 1048576 m G) ()

end LwrProof
