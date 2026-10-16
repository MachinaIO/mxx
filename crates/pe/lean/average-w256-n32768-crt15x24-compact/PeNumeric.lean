import PeDecryptBits
import PeOnlineGlue

namespace PeProof

/-- The exact inherited error bound after one online step. -/
def Bstep (B : Nat) : Nat :=
  stepBound (productBound B companionBound) (bound1 (productBound B companionBound))
    (bound2 (productBound B companionBound))

/-- The exact output bound after the two date lookups. -/
def Bfinal : Nat := Bstep (Bstep 31)

set_option maxRecDepth 100000 in
theorem margin : 4 * (31 + decErr Bfinal + 1) + 4 < Q := by
  have hc : companionBound = 1117673 := by unfold companionBound; norm_num
  have hr : roundBound = 16384 := by unfold roundBound; norm_num
  unfold Bfinal Bstep stepBound productBound bound1 bound2 switchBound scalarBound decErr keyCutoff
  rw [hc, hr]
  decide +kernel

end PeProof
