import TopSetup

/-! The gadgets of every modulus and their centered re-encodings exist. -/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

theorem gadget_exists {q m : Nat} {L : RegularLayout q}
    (hl : Backend.backend.regularLayout q N = some L) (hb : (32768 : Int) = L.base)
    (hd : (m : Int) = L.digitCount) (hw : m = 1 * L.digitCount) :
    ∃ g : ExactMatrix q N 1 m, gadgetMatrixRuns Backend.backend 32768 m g :=
  ⟨_, L, hl, hb, hd, hw, rfl⟩

theorem rebase_exists {q p r c : Nat} (hp : 1 < p) (g : ExactMatrix q N r c) :
    ∃ g' : ExactMatrix p N r c, centeredRebaseRuns g g' :=
  ⟨centeredRebase g, hp, rfl⟩

theorem gadgetQ7 : ∃ g : ExactMatrix Q7 N 1 14, gadgetMatrixRuns Backend.backend 32768 14 g :=
  gadget_exists layoutQ7 (by decide) (by decide) (by decide)
theorem gadgetQ6 : ∃ g : ExactMatrix Q6 N 1 12, gadgetMatrixRuns Backend.backend 32768 12 g :=
  gadget_exists layoutQ6 (by decide) (by decide) (by decide)
theorem gadgetQ5 : ∃ g : ExactMatrix Q5 N 1 10, gadgetMatrixRuns Backend.backend 32768 10 g :=
  gadget_exists layoutQ5 (by decide) (by decide) (by decide)
theorem gadgetQ0 : ∃ g : ExactMatrix Q0 N 1 8, gadgetMatrixRuns Backend.backend 32768 8 g :=
  gadget_exists layoutQ0 (by decide) (by decide) (by decide)
theorem gadgetQ1 : ∃ g : ExactMatrix Q1 N 1 6, gadgetMatrixRuns Backend.backend 32768 6 g :=
  gadget_exists layoutQ1 (by decide) (by decide) (by decide)
theorem gadgetM01 : ∃ g : ExactMatrix M01 N 1 4, gadgetMatrixRuns Backend.backend 32768 4 g :=
  gadget_exists layoutM01 (by decide) (by decide) (by decide)

end RefreshTopProof
