import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard569

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound83673
def owner : Owner := ⟨.program ⟨214⟩, ⟨16263⟩⟩
def transferEvent : Nat := 83673
def frameStart : Nat := 83634
def rule : BoundRule := .identity (.predecessor 0 83672 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83672 .coefficient)
      LeftAuthority83670.bound (LeftAuthority83670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83670.derived selector witness)

def rawBound : CoeffClass := LeftAuthority83670.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority83670.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83673

namespace LeftBound83690
def owner : Owner := ⟨.program ⟨214⟩, ⟨16337⟩⟩
def transferEvent : Nat := 83690
def frameStart : Nat := 83634
def rule : BoundRule := .sum [.predecessor 0 83688 .coefficient, .predecessor 1 83689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83688 .coefficient)
      LeftBound83673.bound (LeftBound83673.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83689 .coefficient)
      LeftAuthority83686.bound (LeftAuthority83686.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority83686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83673.bound, LeftAuthority83686.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83673.bound, LeftAuthority83686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83673.actual selector witness, LeftAuthority83686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83690

namespace LeftBound83693
def owner : Owner := ⟨.program ⟨214⟩, ⟨16338⟩⟩
def transferEvent : Nat := 83693
def frameStart : Nat := 83634
def rule : BoundRule := .identity (.predecessor 0 83692 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83692 .coefficient)
      LeftBound83690.bound (LeftBound83690.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound83690.derived selector witness)

def rawBound : CoeffClass := LeftBound83690.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound83690.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound83693

namespace LeftBound83699
def owner : Owner := ⟨.program ⟨214⟩, ⟨16339⟩⟩
def transferEvent : Nat := 83699
def frameStart : Nat := 83634
def rule : BoundRule := .product (.predecessor 0 83697 .coefficient) (.predecessor 1 83698 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83697 .coefficient)
      LeftAuthority83695.bound (LeftAuthority83695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83698 .coefficient)
      LeftBound83693.bound (LeftBound83693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83693.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority83695.bound LeftBound83693.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83695.bound, LeftBound83693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority83695.actual selector witness) * (LeftBound83693.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83699

namespace LeftBound83707
def owner : Owner := ⟨.program ⟨214⟩, ⟨16340⟩⟩
def transferEvent : Nat := 83707
def frameStart : Nat := 83634
def rule : BoundRule := .sum [.predecessor 0 83705 .coefficient, .predecessor 1 83706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83705 .coefficient)
      LeftAuthority83703.bound (LeftAuthority83703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83706 .coefficient)
      LeftBound83699.bound (LeftBound83699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83703.bound, LeftBound83699.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83703.bound, LeftBound83699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83703.actual selector witness, LeftBound83699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83707

namespace LeftBound83711
def owner : Owner := ⟨.program ⟨214⟩, ⟨28518⟩⟩
def transferEvent : Nat := 83711
def frameStart : Nat := 83634
def rule : BoundRule := .product (.predecessor 0 83709 .coefficient) (.predecessor 1 83710 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83709 .coefficient)
      LeftBound83707.bound (LeftBound83707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83710 .coefficient)
      LeftAuthority83684.bound (LeftAuthority83684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83684.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83707.bound LeftAuthority83684.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83707.bound, LeftAuthority83684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83707.actual selector witness) * (LeftAuthority83684.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83711

namespace LeftBound83722
def owner : Owner := ⟨.program ⟨214⟩, ⟨16309⟩⟩
def transferEvent : Nat := 83722
def frameStart : Nat := 83634
def rule : BoundRule := .product (.predecessor 0 83720 .coefficient) (.predecessor 1 83721 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83720 .coefficient)
      LeftAuthority83695.bound (LeftAuthority83695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83721 .coefficient)
      LeftAuthority83718.bound (LeftAuthority83718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority83695.bound LeftAuthority83718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83695.bound, LeftAuthority83718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority83695.actual selector witness) * (LeftAuthority83718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83722

namespace LeftBound83730
def owner : Owner := ⟨.program ⟨214⟩, ⟨16310⟩⟩
def transferEvent : Nat := 83730
def frameStart : Nat := 83634
def rule : BoundRule := .sum [.predecessor 0 83728 .coefficient, .predecessor 1 83729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83728 .coefficient)
      LeftAuthority83726.bound (LeftAuthority83726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority83726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority83726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83729 .coefficient)
      LeftBound83722.bound (LeftBound83722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority83726.bound, LeftBound83722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority83726.bound, LeftBound83722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority83726.actual selector witness, LeftBound83722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83730

namespace LeftBound83734
def owner : Owner := ⟨.program ⟨214⟩, ⟨28522⟩⟩
def transferEvent : Nat := 83734
def frameStart : Nat := 83634
def rule : BoundRule := .sum [.predecessor 0 83732 .coefficient, .predecessor 1 83733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83732 .coefficient)
      LeftBound83730.bound (LeftBound83730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83733 .coefficient)
      LeftBound83711.bound (LeftBound83711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83730.bound, LeftBound83711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83730.bound, LeftBound83711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83730.actual selector witness, LeftBound83711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83734

namespace LeftBound83747
def owner : Owner := ⟨.program ⟨214⟩, ⟨28520⟩⟩
def transferEvent : Nat := 83747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83745 .coefficient, .predecessor 1 83746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83745 .coefficient)
      LeftBound83576.bound (LeftBound83576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83746 .coefficient)
      LeftBound83559.bound (LeftBound83559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83576.bound, LeftBound83559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83576.bound, LeftBound83559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83576.actual selector witness, LeftBound83559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83747

namespace LeftBound83750
def owner : Owner := ⟨.program ⟨214⟩, ⟨28520⟩⟩
def transferEvent : Nat := 83750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 83744 .summary, .result 83566 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83744 .summary)
      LeftBound83578.bound (LeftBound83578.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21835⟩⟩) (rawTerms := some (Proof.Events327.exact83744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83566 .summary)
      LeftBound83561.bound (LeftBound83561.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28519⟩⟩) (rawTerms := some (Proof.Events326.exact83566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83578.bound, LeftBound83561.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83578.bound, LeftBound83561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83578.actual selector witness, LeftBound83561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83750

namespace LeftBound83774
def owner : Owner := ⟨.program ⟨214⟩, ⟨11638⟩⟩
def transferEvent : Nat := 83774
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 83772 .coefficient) (.predecessor 1 83773 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83772 .coefficient)
      LeftAuthority4011.bound (LeftAuthority4011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83773 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4011.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4011.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4011.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound83774

namespace LeftBound83779
def owner : Owner := ⟨.program ⟨214⟩, ⟨7237⟩⟩
def transferEvent : Nat := 83779
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 83777 .coefficient) (.predecessor 1 83778 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83777 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83778 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound10479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound10479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound10479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound83779

namespace LeftBound83784
def owner : Owner := ⟨.program ⟨214⟩, ⟨11639⟩⟩
def transferEvent : Nat := 83784
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83782 .coefficient, .predecessor 1 83783 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83782 .coefficient)
      LeftBound83779.bound (LeftBound83779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83783 .coefficient)
      LeftBound83774.bound (LeftBound83774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83779.bound, LeftBound83774.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83779.bound, LeftBound83774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83779.actual selector witness, LeftBound83774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83784

namespace LeftBound83788
def owner : Owner := ⟨.program ⟨214⟩, ⟨11640⟩⟩
def transferEvent : Nat := 83788
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 83786 .coefficient, .predecessor 1 83787 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 83786 .coefficient)
      LeftBound83784.bound (LeftBound83784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events327.exact83785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83784.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 83787 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound83784.bound, LeftBound10471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83784.bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound83784.actual selector witness, LeftBound10471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound83788

namespace LeftBound83789
def owner : Owner := ⟨.program ⟨214⟩, ⟨11640⟩⟩
def transferEvent : Nat := 83789
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩ [⟨.result 10472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10472 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨95⟩⟩) (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10471.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound83789

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
