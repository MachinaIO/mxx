import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard497
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard529

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78704
def owner : Owner := ⟨.program ⟨214⟩, ⟨15459⟩⟩
def transferEvent : Nat := 78704
def frameStart : Nat := 78645
def rule : BoundRule := .identity (.predecessor 0 78703 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78703 .coefficient)
      LeftBound78701.bound (LeftBound78701.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78701.derived selector witness)

def rawBound : CoeffClass := LeftBound78701.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound78701.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78704

namespace LeftBound78710
def owner : Owner := ⟨.program ⟨214⟩, ⟨15460⟩⟩
def transferEvent : Nat := 78710
def frameStart : Nat := 78645
def rule : BoundRule := .product (.predecessor 0 78708 .coefficient) (.predecessor 1 78709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78708 .coefficient)
      LeftAuthority78706.bound (LeftAuthority78706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78709 .coefficient)
      LeftBound78704.bound (LeftBound78704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78704.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority78706.bound LeftBound78704.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78706.bound, LeftBound78704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority78706.actual selector witness) * (LeftBound78704.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78710

namespace LeftBound78718
def owner : Owner := ⟨.program ⟨214⟩, ⟨15461⟩⟩
def transferEvent : Nat := 78718
def frameStart : Nat := 78645
def rule : BoundRule := .sum [.predecessor 0 78716 .coefficient, .predecessor 1 78717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78716 .coefficient)
      LeftAuthority78714.bound (LeftAuthority78714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78717 .coefficient)
      LeftBound78710.bound (LeftBound78710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78714.bound, LeftBound78710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78714.bound, LeftBound78710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78714.actual selector witness, LeftBound78710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78718

namespace LeftBound78722
def owner : Owner := ⟨.program ⟨214⟩, ⟨26979⟩⟩
def transferEvent : Nat := 78722
def frameStart : Nat := 78645
def rule : BoundRule := .product (.predecessor 0 78720 .coefficient) (.predecessor 1 78721 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78720 .coefficient)
      LeftBound78718.bound (LeftBound78718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78721 .coefficient)
      LeftAuthority78695.bound (LeftAuthority78695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78695.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78718.bound LeftAuthority78695.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78718.bound, LeftAuthority78695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78718.actual selector witness) * (LeftAuthority78695.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78722

namespace LeftBound78733
def owner : Owner := ⟨.program ⟨214⟩, ⟨15514⟩⟩
def transferEvent : Nat := 78733
def frameStart : Nat := 78645
def rule : BoundRule := .product (.predecessor 0 78731 .coefficient) (.predecessor 1 78732 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78731 .coefficient)
      LeftAuthority78706.bound (LeftAuthority78706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78732 .coefficient)
      LeftAuthority78729.bound (LeftAuthority78729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78706.bound LeftAuthority78729.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78706.bound, LeftAuthority78729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority78706.actual selector witness) * (LeftAuthority78729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78733

namespace LeftBound78741
def owner : Owner := ⟨.program ⟨214⟩, ⟨15515⟩⟩
def transferEvent : Nat := 78741
def frameStart : Nat := 78645
def rule : BoundRule := .sum [.predecessor 0 78739 .coefficient, .predecessor 1 78740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78739 .coefficient)
      LeftAuthority78737.bound (LeftAuthority78737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78740 .coefficient)
      LeftBound78733.bound (LeftBound78733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78737.bound, LeftBound78733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78737.bound, LeftBound78733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority78737.actual selector witness, LeftBound78733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78741

namespace LeftBound78745
def owner : Owner := ⟨.program ⟨214⟩, ⟨26984⟩⟩
def transferEvent : Nat := 78745
def frameStart : Nat := 78645
def rule : BoundRule := .sum [.predecessor 0 78743 .coefficient, .predecessor 1 78744 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78743 .coefficient)
      LeftBound78741.bound (LeftBound78741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78744 .coefficient)
      LeftBound78722.bound (LeftBound78722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78741.bound, LeftBound78722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78741.bound, LeftBound78722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78741.actual selector witness, LeftBound78722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78745

namespace LeftBound78758
def owner : Owner := ⟨.program ⟨214⟩, ⟨26981⟩⟩
def transferEvent : Nat := 78758
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78756 .coefficient, .predecessor 1 78757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78756 .coefficient)
      LeftBound78587.bound (LeftBound78587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78757 .coefficient)
      LeftBound78570.bound (LeftBound78570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78587.bound, LeftBound78570.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78587.bound, LeftBound78570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78587.actual selector witness, LeftBound78570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78758

namespace LeftBound78761
def owner : Owner := ⟨.program ⟨214⟩, ⟨26981⟩⟩
def transferEvent : Nat := 78761
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 78755 .summary, .result 78577 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78755 .summary)
      LeftBound78589.bound (LeftBound78589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20751⟩⟩) (rawTerms := some (Proof.Events307.exact78755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78577 .summary)
      LeftBound78572.bound (LeftBound78572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26980⟩⟩) (rawTerms := some (Proof.Events306.exact78577RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78589.bound, LeftBound78572.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78589.bound, LeftBound78572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound78589.actual selector witness, LeftBound78572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78761

namespace LeftBound78765
def owner : Owner := ⟨.program ⟨214⟩, ⟨26982⟩⟩
def transferEvent : Nat := 78765
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78763 .coefficient) (.predecessor 1 78764 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78763 .coefficient)
      LeftBound78758.bound (LeftBound78758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78758.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78764 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78758.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78758.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78758.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78765

namespace LeftBound78766
def owner : Owner := ⟨.program ⟨214⟩, ⟨26982⟩⟩
def transferEvent : Nat := 78766
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78766

namespace LeftBound78767
def owner : Owner := ⟨.program ⟨214⟩, ⟨26982⟩⟩
def transferEvent : Nat := 78767
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 78762 .summary) (.transfer 78766) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78762 .summary)
      LeftBound78761.bound (LeftBound78761.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26981⟩⟩) (rawTerms := some (Proof.Events307.exact78762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78761.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78766)
      LeftBound78766.bound (LeftBound78766.actual selector witness) := by
  exact .transfer (LeftBound78766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78761.bound LeftBound78766.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78761.bound, LeftBound78766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78761.actual selector witness) * (LeftBound78766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78767

namespace LeftBound78782
def owner : Owner := ⟨.program ⟨214⟩, ⟨26763⟩⟩
def transferEvent : Nat := 78782
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78780 .coefficient) (.predecessor 1 78781 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78780 .coefficient)
      LeftBound72799.bound (LeftBound72799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78781 .coefficient)
      LeftAuthority78778.bound (LeftAuthority78778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78778.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72799.bound LeftAuthority78778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72799.bound, LeftAuthority78778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72799.actual selector witness) * (LeftAuthority78778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78782

namespace LeftBound78783
def owner : Owner := ⟨.program ⟨214⟩, ⟨26763⟩⟩
def transferEvent : Nat := 78783
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26761⟩⟩]⟩ [⟨.result 78779 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78779 .coefficient)
      LeftAuthority78778.bound (LeftAuthority78778.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26761⟩⟩) (rawTerms := some (Proof.Events307.exact78779RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78778.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78778.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78778.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78778.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78783

namespace LeftBound78784
def owner : Owner := ⟨.program ⟨214⟩, ⟨26763⟩⟩
def transferEvent : Nat := 78784
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72803 .summary) (.transfer 78783) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72803 .summary)
      LeftBound72802.bound (LeftBound72802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25062⟩⟩) (rawTerms := some (Proof.Events284.exact72803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78783)
      LeftBound78783.bound (LeftBound78783.actual selector witness) := by
  exact .transfer (LeftBound78783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72802.bound LeftBound78783.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72802.bound, LeftBound78783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72802.actual selector witness) * (LeftBound78783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78784

namespace LeftBound78795
def owner : Owner := ⟨.program ⟨214⟩, ⟨20606⟩⟩
def transferEvent : Nat := 78795
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 78793 .coefficient) (.value (.predecessor 1 78794 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78793 .coefficient)
      LeftAuthority78791.bound (LeftAuthority78791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events307.exact78792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78794 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority78791.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78791.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78791.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound78795

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
