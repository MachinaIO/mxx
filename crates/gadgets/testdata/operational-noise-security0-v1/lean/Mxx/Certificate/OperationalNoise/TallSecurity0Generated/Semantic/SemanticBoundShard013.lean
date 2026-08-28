import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard001

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound4396
def owner : Owner := ⟨.program ⟨214⟩, ⟨18036⟩⟩
def transferEvent : Nat := 4396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4394 .coefficient) (.predecessor 1 4395 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4394 .coefficient)
      LeftAuthority4392.bound (LeftAuthority4392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4395 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4392.bound LeftAuthority632.bound
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4392.bound, LeftAuthority632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4392.actual selector witness) * (LeftAuthority632.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4396

namespace LeftBound4404
def owner : Owner := ⟨.program ⟨214⟩, ⟨17166⟩⟩
def transferEvent : Nat := 4404
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4402 .coefficient) (.predecessor 1 4403 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4402 .coefficient)
      LeftAuthority4400.bound (LeftAuthority4400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4400.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4403 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4400.bound LeftAuthority642.bound
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4400.bound, LeftAuthority642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4400.actual selector witness) * (LeftAuthority642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4404

namespace LeftBound4412
def owner : Owner := ⟨.program ⟨214⟩, ⟨17222⟩⟩
def transferEvent : Nat := 4412
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4410 .coefficient) (.predecessor 1 4411 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4410 .coefficient)
      LeftAuthority4408.bound (LeftAuthority4408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4411 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4408.bound LeftAuthority652.bound
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4408.bound, LeftAuthority652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4408.actual selector witness) * (LeftAuthority652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4412

namespace LeftBound4420
def owner : Owner := ⟨.program ⟨214⟩, ⟨17439⟩⟩
def transferEvent : Nat := 4420
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4418 .coefficient) (.predecessor 1 4419 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4418 .coefficient)
      LeftAuthority4416.bound (LeftAuthority4416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4419 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4416.bound LeftAuthority662.bound
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4416.bound, LeftAuthority662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4416.actual selector witness) * (LeftAuthority662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4420

namespace LeftBound4428
def owner : Owner := ⟨.program ⟨214⟩, ⟨17815⟩⟩
def transferEvent : Nat := 4428
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4426 .coefficient) (.predecessor 1 4427 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4426 .coefficient)
      LeftAuthority4424.bound (LeftAuthority4424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4424.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4427 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4424.bound LeftAuthority672.bound
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4424.bound, LeftAuthority672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4424.actual selector witness) * (LeftAuthority672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4428

namespace LeftBound4436
def owner : Owner := ⟨.program ⟨214⟩, ⟨15517⟩⟩
def transferEvent : Nat := 4436
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4434 .coefficient) (.predecessor 1 4435 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4434 .coefficient)
      LeftAuthority4432.bound (LeftAuthority4432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4435 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4432.bound LeftAuthority682.bound
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4432.bound, LeftAuthority682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4432.actual selector witness) * (LeftAuthority682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4436

namespace LeftBound4444
def owner : Owner := ⟨.program ⟨214⟩, ⟨15209⟩⟩
def transferEvent : Nat := 4444
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4442 .coefficient) (.predecessor 1 4443 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4442 .coefficient)
      LeftAuthority4440.bound (LeftAuthority4440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4443 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4440.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4440.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4440.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4444

namespace LeftBound4452
def owner : Owner := ⟨.program ⟨214⟩, ⟨15048⟩⟩
def transferEvent : Nat := 4452
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4450 .coefficient) (.predecessor 1 4451 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4450 .coefficient)
      LeftAuthority4448.bound (LeftAuthority4448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4451 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4448.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4448.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4448.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4452

namespace LeftBound4460
def owner : Owner := ⟨.program ⟨214⟩, ⟨14887⟩⟩
def transferEvent : Nat := 4460
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4458 .coefficient) (.predecessor 1 4459 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4458 .coefficient)
      LeftAuthority4456.bound (LeftAuthority4456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4459 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4456.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4456.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4456.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4460

namespace LeftBound4465
def owner : Owner := ⟨.program ⟨214⟩, ⟨14888⟩⟩
def transferEvent : Nat := 4465
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4463 .coefficient, .predecessor 1 4464 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4463 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4464 .coefficient)
      LeftBound4460.bound (LeftBound4460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound4460.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound4460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound726.actual selector witness, LeftBound4460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4465

namespace LeftBound4469
def owner : Owner := ⟨.program ⟨214⟩, ⟨15049⟩⟩
def transferEvent : Nat := 4469
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4467 .coefficient, .predecessor 1 4468 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4467 .coefficient)
      LeftBound4465.bound (LeftBound4465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4468 .coefficient)
      LeftBound4452.bound (LeftBound4452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4465.bound, LeftBound4452.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4465.bound, LeftBound4452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4465.actual selector witness, LeftBound4452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4469

namespace LeftBound4473
def owner : Owner := ⟨.program ⟨214⟩, ⟨15210⟩⟩
def transferEvent : Nat := 4473
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4471 .coefficient, .predecessor 1 4472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4471 .coefficient)
      LeftBound4469.bound (LeftBound4469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4472 .coefficient)
      LeftBound4444.bound (LeftBound4444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4469.bound, LeftBound4444.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4469.bound, LeftBound4444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4469.actual selector witness, LeftBound4444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4473

namespace LeftBound4477
def owner : Owner := ⟨.program ⟨214⟩, ⟨15518⟩⟩
def transferEvent : Nat := 4477
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4475 .coefficient, .predecessor 1 4476 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4475 .coefficient)
      LeftBound4473.bound (LeftBound4473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4476 .coefficient)
      LeftBound4436.bound (LeftBound4436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4473.bound, LeftBound4436.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4473.bound, LeftBound4436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4473.actual selector witness, LeftBound4436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4477

namespace LeftBound4481
def owner : Owner := ⟨.program ⟨214⟩, ⟨17816⟩⟩
def transferEvent : Nat := 4481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4479 .coefficient, .predecessor 1 4480 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4479 .coefficient)
      LeftBound4477.bound (LeftBound4477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4480 .coefficient)
      LeftBound4428.bound (LeftBound4428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4428.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4477.bound, LeftBound4428.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4477.bound, LeftBound4428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4477.actual selector witness, LeftBound4428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4481

namespace LeftBound4485
def owner : Owner := ⟨.program ⟨214⟩, ⟨17817⟩⟩
def transferEvent : Nat := 4485
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4483 .coefficient, .predecessor 1 4484 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4483 .coefficient)
      LeftBound4481.bound (LeftBound4481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4484 .coefficient)
      LeftBound4420.bound (LeftBound4420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4481.bound, LeftBound4420.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4481.bound, LeftBound4420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4481.actual selector witness, LeftBound4420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4485

namespace LeftBound4489
def owner : Owner := ⟨.program ⟨214⟩, ⟨17818⟩⟩
def transferEvent : Nat := 4489
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4487 .coefficient, .predecessor 1 4488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4487 .coefficient)
      LeftBound4485.bound (LeftBound4485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4488 .coefficient)
      LeftBound4412.bound (LeftBound4412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4412.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4485.bound, LeftBound4412.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4485.bound, LeftBound4412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound4485.actual selector witness, LeftBound4412.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4489

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
