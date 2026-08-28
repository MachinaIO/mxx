import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard090

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14283
def owner : Owner := ⟨.program ⟨214⟩, ⟨20699⟩⟩
def transferEvent : Nat := 14283
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20696⟩⟩]⟩ [⟨.result 14275 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14275 .coefficient)
      LeftAuthority14274.bound (LeftAuthority14274.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20696⟩⟩) (rawTerms := some (Proof.Events055.exact14275RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14274.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14274.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14274.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14283

namespace LeftBound14284
def owner : Owner := ⟨.program ⟨214⟩, ⟨20699⟩⟩
def transferEvent : Nat := 14284
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 14283) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14283)
      LeftBound14283.bound (LeftBound14283.actual selector witness) := by
  exact .transfer (LeftBound14283.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound14283.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound14283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound14283.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14284

namespace LeftBound14379
def owner : Owner := ⟨.program ⟨214⟩, ⟨15131⟩⟩
def transferEvent : Nat := 14379
def frameStart : Nat := 14340
def rule : BoundRule := .identity (.predecessor 0 14378 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14378 .coefficient)
      LeftAuthority14376.bound (LeftAuthority14376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14376.derived selector witness)

def rawBound : CoeffClass := LeftAuthority14376.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority14376.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14379

namespace LeftBound14396
def owner : Owner := ⟨.program ⟨214⟩, ⟨15170⟩⟩
def transferEvent : Nat := 14396
def frameStart : Nat := 14340
def rule : BoundRule := .sum [.predecessor 0 14394 .coefficient, .predecessor 1 14395 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14394 .coefficient)
      LeftBound14379.bound (LeftBound14379.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound14379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14395 .coefficient)
      LeftAuthority14392.bound (LeftAuthority14392.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority14392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14379.bound, LeftAuthority14392.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14379.bound, LeftAuthority14392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14379.actual selector witness, LeftAuthority14392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14396

namespace LeftBound14399
def owner : Owner := ⟨.program ⟨214⟩, ⟨15171⟩⟩
def transferEvent : Nat := 14399
def frameStart : Nat := 14340
def rule : BoundRule := .identity (.predecessor 0 14398 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14398 .coefficient)
      LeftBound14396.bound (LeftBound14396.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound14396.derived selector witness)

def rawBound : CoeffClass := LeftBound14396.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound14396.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14399

namespace LeftBound14405
def owner : Owner := ⟨.program ⟨214⟩, ⟨15172⟩⟩
def transferEvent : Nat := 14405
def frameStart : Nat := 14340
def rule : BoundRule := .product (.predecessor 0 14403 .coefficient) (.predecessor 1 14404 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14403 .coefficient)
      LeftAuthority14401.bound (LeftAuthority14401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14404 .coefficient)
      LeftBound14399.bound (LeftBound14399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14399.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority14401.bound LeftBound14399.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14401.bound, LeftBound14399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority14401.actual selector witness) * (LeftBound14399.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14405

namespace LeftBound14413
def owner : Owner := ⟨.program ⟨214⟩, ⟨15173⟩⟩
def transferEvent : Nat := 14413
def frameStart : Nat := 14340
def rule : BoundRule := .sum [.predecessor 0 14411 .coefficient, .predecessor 1 14412 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14411 .coefficient)
      LeftAuthority14409.bound (LeftAuthority14409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14409.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14412 .coefficient)
      LeftBound14405.bound (LeftBound14405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority14409.bound, LeftBound14405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14409.bound, LeftBound14405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority14409.actual selector witness, LeftBound14405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14413

namespace LeftBound14417
def owner : Owner := ⟨.program ⟨214⟩, ⟨26834⟩⟩
def transferEvent : Nat := 14417
def frameStart : Nat := 14340
def rule : BoundRule := .product (.predecessor 0 14415 .coefficient) (.predecessor 1 14416 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14415 .coefficient)
      LeftBound14413.bound (LeftBound14413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14416 .coefficient)
      LeftAuthority14390.bound (LeftAuthority14390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14390.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14390.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14413.bound LeftAuthority14390.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14413.bound, LeftAuthority14390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14413.actual selector witness) * (LeftAuthority14390.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14417

namespace LeftBound14428
def owner : Owner := ⟨.program ⟨214⟩, ⟨15384⟩⟩
def transferEvent : Nat := 14428
def frameStart : Nat := 14340
def rule : BoundRule := .product (.predecessor 0 14426 .coefficient) (.predecessor 1 14427 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14426 .coefficient)
      LeftAuthority14401.bound (LeftAuthority14401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14427 .coefficient)
      LeftAuthority14424.bound (LeftAuthority14424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14424.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14401.bound LeftAuthority14424.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14401.bound, LeftAuthority14424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority14401.actual selector witness) * (LeftAuthority14424.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14428

namespace LeftBound14436
def owner : Owner := ⟨.program ⟨214⟩, ⟨15385⟩⟩
def transferEvent : Nat := 14436
def frameStart : Nat := 14340
def rule : BoundRule := .sum [.predecessor 0 14434 .coefficient, .predecessor 1 14435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14434 .coefficient)
      LeftAuthority14432.bound (LeftAuthority14432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14435 .coefficient)
      LeftBound14428.bound (LeftBound14428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14428.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority14432.bound, LeftBound14428.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14432.bound, LeftBound14428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority14432.actual selector witness, LeftBound14428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14436

namespace LeftBound14440
def owner : Owner := ⟨.program ⟨214⟩, ⟨26838⟩⟩
def transferEvent : Nat := 14440
def frameStart : Nat := 14340
def rule : BoundRule := .sum [.predecessor 0 14438 .coefficient, .predecessor 1 14439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14438 .coefficient)
      LeftBound14436.bound (LeftBound14436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14439 .coefficient)
      LeftBound14417.bound (LeftBound14417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14417.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14436.bound, LeftBound14417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14436.bound, LeftBound14417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14436.actual selector witness, LeftBound14417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14440

namespace LeftBound14453
def owner : Owner := ⟨.program ⟨214⟩, ⟨26836⟩⟩
def transferEvent : Nat := 14453
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14451 .coefficient, .predecessor 1 14452 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14451 .coefficient)
      LeftBound14282.bound (LeftBound14282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14452 .coefficient)
      LeftBound14265.bound (LeftBound14265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14282.bound, LeftBound14265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14282.bound, LeftBound14265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14282.actual selector witness, LeftBound14265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14453

namespace LeftBound14456
def owner : Owner := ⟨.program ⟨214⟩, ⟨26836⟩⟩
def transferEvent : Nat := 14456
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 14450 .summary, .result 14272 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14450 .summary)
      LeftBound14284.bound (LeftBound14284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20699⟩⟩) (rawTerms := some (Proof.Events056.exact14450RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14272 .summary)
      LeftBound14267.bound (LeftBound14267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26835⟩⟩) (rawTerms := some (Proof.Events055.exact14272RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14284.bound, LeftBound14267.bound]
def bound : CoeffClass := .finite ⟨1291911586824442228736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14284.bound, LeftBound14267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14284.actual selector witness, LeftBound14267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14456

namespace LeftBound14479
def owner : Owner := ⟨.program ⟨214⟩, ⟨87⟩⟩
def transferEvent : Nat := 14479
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 14478 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14478 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14479

namespace LeftBound14483
def owner : Owner := ⟨.program ⟨214⟩, ⟨10711⟩⟩
def transferEvent : Nat := 14483
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 14481 .coefficient) (.predecessor 1 14482 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14481 .coefficient)
      LeftAuthority418.bound (LeftAuthority418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events001.exact419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14482 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority418.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority418.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority418.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14483

namespace LeftBound14487
def owner : Owner := ⟨.program ⟨214⟩, ⟨6773⟩⟩
def transferEvent : Nat := 14487
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 14486 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14486 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14487

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
