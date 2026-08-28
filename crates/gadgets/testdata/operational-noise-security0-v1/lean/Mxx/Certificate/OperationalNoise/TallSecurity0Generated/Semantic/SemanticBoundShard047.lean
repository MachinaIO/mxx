import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard046

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8711
def owner : Owner := ⟨.program ⟨214⟩, ⟨25473⟩⟩
def transferEvent : Nat := 8711
def frameStart : Nat := 8620
def rule : BoundRule := .product (.predecessor 0 8709 .coefficient) (.predecessor 1 8710 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8709 .coefficient)
      LeftBound8707.bound (LeftBound8707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8710 .coefficient)
      LeftAuthority8664.bound (LeftAuthority8664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8707.bound LeftAuthority8664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8707.bound, LeftAuthority8664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8707.actual selector witness) * (LeftAuthority8664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8711

namespace LeftBound8722
def owner : Owner := ⟨.program ⟨214⟩, ⟨16567⟩⟩
def transferEvent : Nat := 8722
def frameStart : Nat := 8620
def rule : BoundRule := .product (.predecessor 0 8720 .coefficient) (.predecessor 1 8721 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8720 .coefficient)
      LeftAuthority8675.bound (LeftAuthority8675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8721 .coefficient)
      LeftAuthority8718.bound (LeftAuthority8718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8675.bound LeftAuthority8718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8675.bound, LeftAuthority8718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority8675.actual selector witness) * (LeftAuthority8718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8722

namespace LeftBound8730
def owner : Owner := ⟨.program ⟨214⟩, ⟨16568⟩⟩
def transferEvent : Nat := 8730
def frameStart : Nat := 8620
def rule : BoundRule := .sum [.predecessor 0 8728 .coefficient, .predecessor 1 8729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8728 .coefficient)
      LeftAuthority8726.bound (LeftAuthority8726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8729 .coefficient)
      LeftBound8722.bound (LeftBound8722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority8726.bound, LeftBound8722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8726.bound, LeftBound8722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority8726.actual selector witness, LeftBound8722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8730

namespace LeftBound8734
def owner : Owner := ⟨.program ⟨214⟩, ⟨25474⟩⟩
def transferEvent : Nat := 8734
def frameStart : Nat := 8620
def rule : BoundRule := .sum [.predecessor 0 8732 .coefficient, .predecessor 1 8733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8732 .coefficient)
      LeftBound8730.bound (LeftBound8730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8733 .coefficient)
      LeftBound8711.bound (LeftBound8711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8730.bound, LeftBound8711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8730.bound, LeftBound8711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8730.actual selector witness, LeftBound8711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8734

namespace LeftBound8747
def owner : Owner := ⟨.program ⟨214⟩, ⟨25472⟩⟩
def transferEvent : Nat := 8747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8745 .coefficient, .predecessor 1 8746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8745 .coefficient)
      LeftBound8568.bound (LeftBound8568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8746 .coefficient)
      LeftBound8551.bound (LeftBound8551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8568.bound, LeftBound8551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8568.bound, LeftBound8551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8568.actual selector witness, LeftBound8551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8747

namespace LeftBound8750
def owner : Owner := ⟨.program ⟨214⟩, ⟨25472⟩⟩
def transferEvent : Nat := 8750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 8744 .summary, .result 8558 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8744 .summary)
      LeftBound8570.bound (LeftBound8570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19979⟩⟩) (rawTerms := some (Proof.Events034.exact8744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8558 .summary)
      LeftBound8553.bound (LeftBound8553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25471⟩⟩) (rawTerms := some (Proof.Events033.exact8558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8570.bound, LeftBound8553.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8570.bound, LeftBound8553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8570.actual selector witness, LeftBound8553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8750

namespace LeftBound8754
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def transferEvent : Nat := 8754
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8752 .coefficient) (.predecessor 1 8753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8752 .coefficient)
      LeftBound8747.bound (LeftBound8747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8753 .coefficient)
      LeftAuthority8454.bound (LeftAuthority8454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8454.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8747.bound LeftAuthority8454.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8747.bound, LeftAuthority8454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8747.actual selector witness) * (LeftAuthority8454.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8754

namespace LeftBound8755
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def transferEvent : Nat := 8755
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29220⟩⟩]⟩ [⟨.result 8455 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8455 .coefficient)
      LeftAuthority8454.bound (LeftAuthority8454.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29220⟩⟩) (rawTerms := some (Proof.Events033.exact8455RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8454.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8454.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8454.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8755

namespace LeftBound8756
def owner : Owner := ⟨.program ⟨214⟩, ⟨29222⟩⟩
def transferEvent : Nat := 8756
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8751 .summary) (.transfer 8755) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8751 .summary)
      LeftBound8750.bound (LeftBound8750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25472⟩⟩) (rawTerms := some (Proof.Events034.exact8751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8755)
      LeftBound8755.bound (LeftBound8755.actual selector witness) := by
  exact .transfer (LeftBound8755.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8750.bound LeftBound8755.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8750.bound, LeftBound8755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8750.actual selector witness) * (LeftBound8755.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8756

namespace LeftBound8767
def owner : Owner := ⟨.program ⟨214⟩, ⟨22282⟩⟩
def transferEvent : Nat := 8767
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 8765 .coefficient) (.value (.predecessor 1 8766 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8765 .coefficient)
      LeftAuthority8763.bound (LeftAuthority8763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8763.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8766 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority8763.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8763.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8763.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound8767

namespace LeftBound8771
def owner : Owner := ⟨.program ⟨214⟩, ⟨22283⟩⟩
def transferEvent : Nat := 8771
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8769 .coefficient) (.predecessor 1 8770 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8769 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8770 .coefficient)
      LeftBound8767.bound (LeftBound8767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8767.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound8767.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound8767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound8767.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8771

namespace LeftBound8772
def owner : Owner := ⟨.program ⟨214⟩, ⟨22283⟩⟩
def transferEvent : Nat := 8772
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22280⟩⟩]⟩ [⟨.result 8764 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8764 .coefficient)
      LeftAuthority8763.bound (LeftAuthority8763.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22280⟩⟩) (rawTerms := some (Proof.Events034.exact8764RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8763.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8763.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8763.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8763.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8772

namespace LeftBound8773
def owner : Owner := ⟨.program ⟨214⟩, ⟨22283⟩⟩
def transferEvent : Nat := 8773
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 8772) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8772)
      LeftBound8772.bound (LeftBound8772.actual selector witness) := by
  exact .transfer (LeftBound8772.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound8772.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound8772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound8772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8773

namespace LeftBound8868
def owner : Owner := ⟨.program ⟨214⟩, ⟨16566⟩⟩
def transferEvent : Nat := 8868
def frameStart : Nat := 8829
def rule : BoundRule := .identity (.predecessor 0 8867 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8867 .coefficient)
      LeftAuthority8865.bound (LeftAuthority8865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8865.derived selector witness)

def rawBound : CoeffClass := LeftAuthority8865.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8865.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority8865.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8868

namespace LeftBound8885
def owner : Owner := ⟨.program ⟨214⟩, ⟨16605⟩⟩
def transferEvent : Nat := 8885
def frameStart : Nat := 8829
def rule : BoundRule := .sum [.predecessor 0 8883 .coefficient, .predecessor 1 8884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8883 .coefficient)
      LeftBound8868.bound (LeftBound8868.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound8868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8884 .coefficient)
      LeftAuthority8881.bound (LeftAuthority8881.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority8881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8868.bound, LeftAuthority8881.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8868.bound, LeftAuthority8881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8868.actual selector witness, LeftAuthority8881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8885

namespace LeftBound8888
def owner : Owner := ⟨.program ⟨214⟩, ⟨16606⟩⟩
def transferEvent : Nat := 8888
def frameStart : Nat := 8829
def rule : BoundRule := .identity (.predecessor 0 8887 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8887 .coefficient)
      LeftBound8885.bound (LeftBound8885.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound8885.derived selector witness)

def rawBound : CoeffClass := LeftBound8885.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound8885.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8888

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
