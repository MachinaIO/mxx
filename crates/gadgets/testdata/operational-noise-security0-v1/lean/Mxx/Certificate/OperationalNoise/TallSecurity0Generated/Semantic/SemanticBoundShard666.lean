import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard665

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97444
def owner : Owner := ⟨.program ⟨214⟩, ⟨11743⟩⟩
def transferEvent : Nat := 97444
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97439 .summary) (.transfer 97443) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97439 .summary)
      LeftBound97437.bound (LeftBound97437.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11742⟩⟩) (rawTerms := some (Proof.Events380.exact97439RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97443)
      LeftBound97443.bound (LeftBound97443.actual selector witness) := by
  exact .transfer (LeftBound97443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound97437.bound LeftBound97443.bound
def bound : CoeffClass := .finite ⟨24960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97437.bound, LeftBound97443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound97437.actual selector witness) * (LeftBound97443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97444

namespace LeftBound97450
def owner : Owner := ⟨.program ⟨214⟩, ⟨9596⟩⟩
def transferEvent : Nat := 97450
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 97448 .coefficient) (.predecessor 1 97449 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97448 .coefficient)
      LeftAuthority4729.bound (LeftAuthority4729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97449 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4729.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4729.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4729.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97450

namespace LeftBound97455
def owner : Owner := ⟨.program ⟨214⟩, ⟨7100⟩⟩
def transferEvent : Nat := 97455
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97453 .coefficient) (.predecessor 1 97454 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97453 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97454 .coefficient)
      LeftBound10019.bound (LeftBound10019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound10019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound10019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound10019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97455

namespace LeftBound97460
def owner : Owner := ⟨.program ⟨214⟩, ⟨9597⟩⟩
def transferEvent : Nat := 97460
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97458 .coefficient, .predecessor 1 97459 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97458 .coefficient)
      LeftBound97455.bound (LeftBound97455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97459 .coefficient)
      LeftBound97450.bound (LeftBound97450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97455.bound, LeftBound97450.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97455.bound, LeftBound97450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97455.actual selector witness, LeftBound97450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97460

namespace LeftBound97464
def owner : Owner := ⟨.program ⟨214⟩, ⟨9598⟩⟩
def transferEvent : Nat := 97464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97462 .coefficient, .predecessor 1 97463 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97462 .coefficient)
      LeftBound97460.bound (LeftBound97460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97463 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97460.bound, LeftBound10011.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97460.bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97460.actual selector witness, LeftBound10011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97464

namespace LeftBound97465
def owner : Owner := ⟨.program ⟨214⟩, ⟨9598⟩⟩
def transferEvent : Nat := 97465
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩ [⟨.result 10012 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10012 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨77⟩⟩) (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10011.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10011.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97465

namespace LeftBound97470
def owner : Owner := ⟨.program ⟨214⟩, ⟨9599⟩⟩
def transferEvent : Nat := 97470
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97468 .coefficient) (.predecessor 1 97469 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97468 .coefficient)
      LeftBound97464.bound (LeftBound97464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97469 .coefficient)
      LeftBound10008.bound (LeftBound10008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97464.bound LeftBound10008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97464.bound, LeftBound10008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97464.actual selector witness) * (LeftBound10008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97470

namespace LeftBound97471
def owner : Owner := ⟨.program ⟨214⟩, ⟨9599⟩⟩
def transferEvent : Nat := 97471
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩ [⟨.result 10005 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10005 .coefficient)
      LeftAuthority10004.bound (LeftAuthority10004.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7861⟩⟩) (rawTerms := some (Proof.Events039.exact10005RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10004.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10004.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10004.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97471

namespace LeftBound97472
def owner : Owner := ⟨.program ⟨214⟩, ⟨9599⟩⟩
def transferEvent : Nat := 97472
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97467 .summary) (.transfer 97471) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97467 .summary)
      LeftBound97465.bound (LeftBound97465.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9598⟩⟩) (rawTerms := some (Proof.Events380.exact97467RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97471)
      LeftBound97471.bound (LeftBound97471.actual selector witness) := by
  exact .transfer (LeftBound97471.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97465.bound LeftBound97471.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97465.bound, LeftBound97471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97465.actual selector witness) * (LeftBound97471.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97472

namespace LeftBound97480
def owner : Owner := ⟨.program ⟨214⟩, ⟨11744⟩⟩
def transferEvent : Nat := 97480
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97478 .coefficient, .predecessor 1 97479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97478 .coefficient)
      LeftBound97470.bound (LeftBound97470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97470.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97479 .coefficient)
      LeftBound97442.bound (LeftBound97442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97470.bound, LeftBound97442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97470.bound, LeftBound97442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97470.actual selector witness, LeftBound97442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97480

namespace LeftBound97482
def owner : Owner := ⟨.program ⟨214⟩, ⟨11744⟩⟩
def transferEvent : Nat := 97482
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97477 .summary, .result 97447 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97477 .summary)
      LeftBound97472.bound (LeftBound97472.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9599⟩⟩) (rawTerms := some (Proof.Events380.exact97477RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97447 .summary)
      LeftBound97444.bound (LeftBound97444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11743⟩⟩) (rawTerms := some (Proof.Events380.exact97447RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97472.bound, LeftBound97444.bound]
def bound : CoeffClass := .finite ⟨95445376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97472.bound, LeftBound97444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97472.actual selector witness, LeftBound97444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97482

namespace LeftBound97486
def owner : Owner := ⟨.program ⟨214⟩, ⟨25130⟩⟩
def transferEvent : Nat := 97486
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97484 .coefficient) (.predecessor 1 97485 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97484 .coefficient)
      LeftBound97480.bound (LeftBound97480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97485 .coefficient)
      LeftAuthority97418.bound (LeftAuthority97418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97418.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97480.bound LeftAuthority97418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97480.bound, LeftAuthority97418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97480.actual selector witness) * (LeftAuthority97418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97486

namespace LeftBound97487
def owner : Owner := ⟨.program ⟨214⟩, ⟨25130⟩⟩
def transferEvent : Nat := 97487
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25129⟩⟩]⟩ [⟨.result 97419 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97419 .coefficient)
      LeftAuthority97418.bound (LeftAuthority97418.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25129⟩⟩) (rawTerms := some (Proof.Events380.exact97419RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97418.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97418.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97418.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97487

namespace LeftBound97488
def owner : Owner := ⟨.program ⟨214⟩, ⟨25130⟩⟩
def transferEvent : Nat := 97488
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97483 .summary) (.transfer 97487) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97483 .summary)
      LeftBound97482.bound (LeftBound97482.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11744⟩⟩) (rawTerms := some (Proof.Events380.exact97483RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97487)
      LeftBound97487.bound (LeftBound97487.actual selector witness) := by
  exact .transfer (LeftBound97487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97482.bound LeftBound97487.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97482.bound, LeftBound97487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97482.actual selector witness) * (LeftBound97487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97488

namespace LeftBound97499
def owner : Owner := ⟨.program ⟨214⟩, ⟨19735⟩⟩
def transferEvent : Nat := 97499
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 97497 .coefficient) (.value (.predecessor 1 97498 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97497 .coefficient)
      LeftAuthority97495.bound (LeftAuthority97495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97498 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority97495.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97495.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97495.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97499

namespace LeftBound97503
def owner : Owner := ⟨.program ⟨214⟩, ⟨19736⟩⟩
def transferEvent : Nat := 97503
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97501 .coefficient) (.predecessor 1 97502 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97501 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97502 .coefficient)
      LeftBound97499.bound (LeftBound97499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound97499.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound97499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound97499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97503

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
