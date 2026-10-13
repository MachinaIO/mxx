import PeDecryptBits
import PeOnlineGlue

/-!
# The numeric margin

Four online steps from the state error `31`, and the decryption error, stay below `Q / 4`.
-/

namespace PeProof

/-- The error bound after one online step from the state bound `B`. -/
def Bstep (B : Nat) : Nat :=
  stepBound (productBound B companionBound) (bound1 (productBound B companionBound))
    (bound2 (productBound B companionBound))

/-- The output bound after the four steps. -/
def Bfinal : Nat := Bstep (Bstep (Bstep (Bstep 31)))

set_option maxRecDepth 100000 in
theorem margin : 4 * (31 + decErr Bfinal + 1) + 4 < Q := by
  have hc : companionBound = 32786 := by unfold companionBound; norm_num
  have hr : roundBound = 32768 := by unfold roundBound; norm_num
  unfold Bfinal Bstep stepBound productBound bound1 bound2 switchBound scalarBound decErr keyCutoff
  rw [hc, hr]
  decide

end PeProof
