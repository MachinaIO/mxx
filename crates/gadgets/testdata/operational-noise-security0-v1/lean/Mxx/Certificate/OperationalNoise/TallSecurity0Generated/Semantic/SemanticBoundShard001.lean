import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound679
def owner : Owner := ⟨.program ⟨214⟩, ⟨17847⟩⟩
def transferEvent : Nat := 679
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 677 .coefficient) (.predecessor 1 678 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 677 .coefficient)
      LeftAuthority675.bound (LeftAuthority675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 678 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority675.bound LeftAuthority672.bound
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority675.bound, LeftAuthority672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority675.actual selector witness) * (LeftAuthority672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound679

namespace LeftBound689
def owner : Owner := ⟨.program ⟨214⟩, ⟨15537⟩⟩
def transferEvent : Nat := 689
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 687 .coefficient) (.predecessor 1 688 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 687 .coefficient)
      LeftAuthority685.bound (LeftAuthority685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 688 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority685.bound LeftAuthority682.bound
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority685.bound, LeftAuthority682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority685.actual selector witness) * (LeftAuthority682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound689

namespace LeftBound699
def owner : Owner := ⟨.program ⟨214⟩, ⟨15229⟩⟩
def transferEvent : Nat := 699
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 697 .coefficient) (.predecessor 1 698 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 697 .coefficient)
      LeftAuthority695.bound (LeftAuthority695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 698 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority695.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority695.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority695.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound699

namespace LeftBound709
def owner : Owner := ⟨.program ⟨214⟩, ⟨15068⟩⟩
def transferEvent : Nat := 709
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 707 .coefficient) (.predecessor 1 708 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 707 .coefficient)
      LeftAuthority705.bound (LeftAuthority705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority705.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 708 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority705.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority705.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority705.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound709

namespace LeftBound719
def owner : Owner := ⟨.program ⟨214⟩, ⟨14907⟩⟩
def transferEvent : Nat := 719
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 717 .coefficient) (.predecessor 1 718 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 717 .coefficient)
      LeftAuthority715.bound (LeftAuthority715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority715.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 718 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority715.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority715.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority715.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound719

namespace LeftBound726
def owner : Owner := ⟨.program ⟨214⟩, ⟨6379⟩⟩
def transferEvent : Nat := 726
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 724 .coefficient, .predecessor 1 725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 724 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 725 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority722.bound, LeftAuthority722.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftAuthority722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority722.actual selector witness, LeftAuthority722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound726

namespace LeftBound731
def owner : Owner := ⟨.program ⟨214⟩, ⟨14908⟩⟩
def transferEvent : Nat := 731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 729 .coefficient, .predecessor 1 730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 729 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 730 .coefficient)
      LeftBound719.bound (LeftBound719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound719.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound726.actual selector witness, LeftBound719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound731

namespace LeftBound735
def owner : Owner := ⟨.program ⟨214⟩, ⟨15069⟩⟩
def transferEvent : Nat := 735
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 733 .coefficient, .predecessor 1 734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 733 .coefficient)
      LeftBound731.bound (LeftBound731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 734 .coefficient)
      LeftBound709.bound (LeftBound709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound731.bound, LeftBound709.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound731.bound, LeftBound709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound731.actual selector witness, LeftBound709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound735

namespace LeftBound739
def owner : Owner := ⟨.program ⟨214⟩, ⟨15230⟩⟩
def transferEvent : Nat := 739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 737 .coefficient, .predecessor 1 738 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 737 .coefficient)
      LeftBound735.bound (LeftBound735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 738 .coefficient)
      LeftBound699.bound (LeftBound699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound699.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound735.bound, LeftBound699.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound735.bound, LeftBound699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound735.actual selector witness, LeftBound699.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound739

namespace LeftBound743
def owner : Owner := ⟨.program ⟨214⟩, ⟨15538⟩⟩
def transferEvent : Nat := 743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 741 .coefficient, .predecessor 1 742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 741 .coefficient)
      LeftBound739.bound (LeftBound739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 742 .coefficient)
      LeftBound689.bound (LeftBound689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound689.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound739.bound, LeftBound689.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound739.bound, LeftBound689.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound739.actual selector witness, LeftBound689.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound743

namespace LeftBound747
def owner : Owner := ⟨.program ⟨214⟩, ⟨17848⟩⟩
def transferEvent : Nat := 747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 745 .coefficient, .predecessor 1 746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 745 .coefficient)
      LeftBound743.bound (LeftBound743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 746 .coefficient)
      LeftBound679.bound (LeftBound679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound743.bound, LeftBound679.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound743.bound, LeftBound679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound743.actual selector witness, LeftBound679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound747

namespace LeftBound751
def owner : Owner := ⟨.program ⟨214⟩, ⟨17849⟩⟩
def transferEvent : Nat := 751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 749 .coefficient, .predecessor 1 750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 749 .coefficient)
      LeftBound747.bound (LeftBound747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 750 .coefficient)
      LeftBound669.bound (LeftBound669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound747.bound, LeftBound669.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound747.bound, LeftBound669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound747.actual selector witness, LeftBound669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound751

namespace LeftBound755
def owner : Owner := ⟨.program ⟨214⟩, ⟨17850⟩⟩
def transferEvent : Nat := 755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 753 .coefficient, .predecessor 1 754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 753 .coefficient)
      LeftBound751.bound (LeftBound751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 754 .coefficient)
      LeftBound659.bound (LeftBound659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound751.bound, LeftBound659.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound751.bound, LeftBound659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound751.actual selector witness, LeftBound659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound755

namespace LeftBound759
def owner : Owner := ⟨.program ⟨214⟩, ⟨17851⟩⟩
def transferEvent : Nat := 759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 757 .coefficient, .predecessor 1 758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 757 .coefficient)
      LeftBound755.bound (LeftBound755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 758 .coefficient)
      LeftBound649.bound (LeftBound649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound755.bound, LeftBound649.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound755.bound, LeftBound649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound755.actual selector witness, LeftBound649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound759

namespace LeftBound763
def owner : Owner := ⟨.program ⟨214⟩, ⟨18065⟩⟩
def transferEvent : Nat := 763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 761 .coefficient, .predecessor 1 762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 761 .coefficient)
      LeftBound759.bound (LeftBound759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 762 .coefficient)
      LeftBound639.bound (LeftBound639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound759.bound, LeftBound639.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound759.bound, LeftBound639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound759.actual selector witness, LeftBound639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound763

namespace LeftBound767
def owner : Owner := ⟨.program ⟨214⟩, ⟨18066⟩⟩
def transferEvent : Nat := 767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 765 .coefficient, .predecessor 1 766 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 765 .coefficient)
      LeftBound763.bound (LeftBound763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 766 .coefficient)
      LeftBound629.bound (LeftBound629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound763.bound, LeftBound629.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound763.bound, LeftBound629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound763.actual selector witness, LeftBound629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound767

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
