import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard569
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard620

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91603
def owner : Owner := ⟨.program ⟨214⟩, ⟨16423⟩⟩
def transferEvent : Nat := 91603
def frameStart : Nat := 91538
def rule : BoundRule := .product (.predecessor 0 91601 .coefficient) (.predecessor 1 91602 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91601 .coefficient)
      LeftAuthority91599.bound (LeftAuthority91599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91602 .coefficient)
      LeftBound91597.bound (LeftBound91597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority91599.bound LeftBound91597.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91599.bound, LeftBound91597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority91599.actual selector witness) * (LeftBound91597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91603

namespace LeftBound91611
def owner : Owner := ⟨.program ⟨214⟩, ⟨16424⟩⟩
def transferEvent : Nat := 91611
def frameStart : Nat := 91538
def rule : BoundRule := .sum [.predecessor 0 91609 .coefficient, .predecessor 1 91610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91609 .coefficient)
      LeftAuthority91607.bound (LeftAuthority91607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91610 .coefficient)
      LeftBound91603.bound (LeftBound91603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91607.bound, LeftBound91603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91607.bound, LeftBound91603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91607.actual selector witness, LeftBound91603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91611

namespace LeftBound91615
def owner : Owner := ⟨.program ⟨214⟩, ⟨28728⟩⟩
def transferEvent : Nat := 91615
def frameStart : Nat := 91538
def rule : BoundRule := .product (.predecessor 0 91613 .coefficient) (.predecessor 1 91614 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91613 .coefficient)
      LeftBound91611.bound (LeftBound91611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91614 .coefficient)
      LeftAuthority91588.bound (LeftAuthority91588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91611.bound LeftAuthority91588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91611.bound, LeftAuthority91588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91611.actual selector witness) * (LeftAuthority91588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91615

namespace LeftBound91626
def owner : Owner := ⟨.program ⟨214⟩, ⟨18841⟩⟩
def transferEvent : Nat := 91626
def frameStart : Nat := 91538
def rule : BoundRule := .product (.predecessor 0 91624 .coefficient) (.predecessor 1 91625 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91624 .coefficient)
      LeftAuthority91599.bound (LeftAuthority91599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91625 .coefficient)
      LeftAuthority91622.bound (LeftAuthority91622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority91599.bound LeftAuthority91622.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91599.bound, LeftAuthority91622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority91599.actual selector witness) * (LeftAuthority91622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91626

namespace LeftBound91634
def owner : Owner := ⟨.program ⟨214⟩, ⟨18847⟩⟩
def transferEvent : Nat := 91634
def frameStart : Nat := 91538
def rule : BoundRule := .sum [.predecessor 0 91632 .coefficient, .predecessor 1 91633 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91632 .coefficient)
      LeftAuthority91630.bound (LeftAuthority91630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91633 .coefficient)
      LeftBound91626.bound (LeftBound91626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91626.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91630.bound, LeftBound91626.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91630.bound, LeftBound91626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91630.actual selector witness, LeftBound91626.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91634

namespace LeftBound91638
def owner : Owner := ⟨.program ⟨214⟩, ⟨28733⟩⟩
def transferEvent : Nat := 91638
def frameStart : Nat := 91538
def rule : BoundRule := .sum [.predecessor 0 91636 .coefficient, .predecessor 1 91637 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91636 .coefficient)
      LeftBound91634.bound (LeftBound91634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91637 .coefficient)
      LeftBound91615.bound (LeftBound91615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91615.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91634.bound, LeftBound91615.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91634.bound, LeftBound91615.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91634.actual selector witness, LeftBound91615.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91638

namespace LeftBound91651
def owner : Owner := ⟨.program ⟨214⟩, ⟨28730⟩⟩
def transferEvent : Nat := 91651
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91649 .coefficient, .predecessor 1 91650 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91649 .coefficient)
      LeftBound91480.bound (LeftBound91480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91650 .coefficient)
      LeftBound91463.bound (LeftBound91463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91480.bound, LeftBound91463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91480.bound, LeftBound91463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91480.actual selector witness, LeftBound91463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91651

namespace LeftBound91654
def owner : Owner := ⟨.program ⟨214⟩, ⟨28730⟩⟩
def transferEvent : Nat := 91654
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91648 .summary, .result 91470 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91648 .summary)
      LeftBound91482.bound (LeftBound91482.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21907⟩⟩) (rawTerms := some (Proof.Events358.exact91648RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91470 .summary)
      LeftBound91465.bound (LeftBound91465.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28729⟩⟩) (rawTerms := some (Proof.Events357.exact91470RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91482.bound, LeftBound91465.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91482.bound, LeftBound91465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91482.actual selector witness, LeftBound91465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91654

namespace LeftBound91658
def owner : Owner := ⟨.program ⟨214⟩, ⟨28731⟩⟩
def transferEvent : Nat := 91658
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91656 .coefficient) (.predecessor 1 91657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91656 .coefficient)
      LeftBound91651.bound (LeftBound91651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91657 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91651.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91651.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91651.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91658

namespace LeftBound91659
def owner : Owner := ⟨.program ⟨214⟩, ⟨28731⟩⟩
def transferEvent : Nat := 91659
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩ [⟨.result 5635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5635 .coefficient)
      LeftAuthority5634.bound (LeftAuthority5634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6673⟩⟩) (rawTerms := some (Proof.Events022.exact5635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91659

namespace LeftBound91660
def owner : Owner := ⟨.program ⟨214⟩, ⟨28731⟩⟩
def transferEvent : Nat := 91660
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91655 .summary) (.transfer 91659) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91655 .summary)
      LeftBound91654.bound (LeftBound91654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28730⟩⟩) (rawTerms := some (Proof.Events358.exact91655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91659)
      LeftBound91659.bound (LeftBound91659.actual selector witness) := by
  exact .transfer (LeftBound91659.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91654.bound LeftBound91659.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91654.bound, LeftBound91659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91654.actual selector witness) * (LeftBound91659.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91660

namespace LeftBound91675
def owner : Owner := ⟨.program ⟨214⟩, ⟨28512⟩⟩
def transferEvent : Nat := 91675
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91673 .coefficient) (.predecessor 1 91674 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91673 .coefficient)
      LeftBound83552.bound (LeftBound83552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91674 .coefficient)
      LeftAuthority91671.bound (LeftAuthority91671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91671.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83552.bound LeftAuthority91671.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83552.bound, LeftAuthority91671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83552.actual selector witness) * (LeftAuthority91671.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91675

namespace LeftBound91676
def owner : Owner := ⟨.program ⟨214⟩, ⟨28512⟩⟩
def transferEvent : Nat := 91676
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28510⟩⟩]⟩ [⟨.result 91672 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91672 .coefficient)
      LeftAuthority91671.bound (LeftAuthority91671.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28510⟩⟩) (rawTerms := some (Proof.Events358.exact91672RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91671.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91671.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91671.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91671.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91676

namespace LeftBound91677
def owner : Owner := ⟨.program ⟨214⟩, ⟨28512⟩⟩
def transferEvent : Nat := 91677
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 83556 .summary) (.transfer 91676) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 83556 .summary)
      LeftBound83555.bound (LeftBound83555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25144⟩⟩) (rawTerms := some (Proof.Events326.exact83556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91676)
      LeftBound91676.bound (LeftBound91676.actual selector witness) := by
  exact .transfer (LeftBound91676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound83555.bound LeftBound91676.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound83555.bound, LeftBound91676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound83555.actual selector witness) * (LeftBound91676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91677

namespace LeftBound91688
def owner : Owner := ⟨.program ⟨214⟩, ⟨21762⟩⟩
def transferEvent : Nat := 91688
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91686 .coefficient) (.value (.predecessor 1 91687 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91686 .coefficient)
      LeftAuthority91684.bound (LeftAuthority91684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91687 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91684.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91684.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91684.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91688

namespace LeftBound91692
def owner : Owner := ⟨.program ⟨214⟩, ⟨21763⟩⟩
def transferEvent : Nat := 91692
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91690 .coefficient) (.predecessor 1 91691 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91690 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91691 .coefficient)
      LeftBound91688.bound (LeftBound91688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91688.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound91688.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound91688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound91688.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91692

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
