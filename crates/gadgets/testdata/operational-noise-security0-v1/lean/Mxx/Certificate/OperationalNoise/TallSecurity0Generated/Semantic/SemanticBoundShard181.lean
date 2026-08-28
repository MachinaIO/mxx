import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard180

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27380
def owner : Owner := ⟨.program ⟨214⟩, ⟨13801⟩⟩
def transferEvent : Nat := 27380
def frameStart : Nat := 27351
def rule : BoundRule := .product (.predecessor 0 27378 .coefficient) (.predecessor 1 27379 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27378 .coefficient)
      LeftAuthority27376.bound (LeftAuthority27376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27379 .coefficient)
      LeftAuthority27373.bound (LeftAuthority27373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27373.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27376.bound LeftAuthority27373.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27376.bound, LeftAuthority27373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27376.actual selector witness) * (LeftAuthority27373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27380

namespace LeftBound27384
def owner : Owner := ⟨.program ⟨214⟩, ⟨13802⟩⟩
def transferEvent : Nat := 27384
def frameStart : Nat := 27351
def rule : BoundRule := .identity (.predecessor 0 27383 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27383 .coefficient)
      LeftBound27380.bound (LeftBound27380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27380.derived selector witness)

def rawBound : CoeffClass := LeftBound27380.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27380.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27384

namespace LeftBound27401
def owner : Owner := ⟨.program ⟨214⟩, ⟨13892⟩⟩
def transferEvent : Nat := 27401
def frameStart : Nat := 27351
def rule : BoundRule := .sum [.predecessor 0 27399 .coefficient, .predecessor 1 27400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27399 .coefficient)
      LeftBound27384.bound (LeftBound27384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27400 .coefficient)
      LeftAuthority27397.bound (LeftAuthority27397.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27384.bound, LeftAuthority27397.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27384.bound, LeftAuthority27397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27384.actual selector witness, LeftAuthority27397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27401

namespace LeftBound27404
def owner : Owner := ⟨.program ⟨214⟩, ⟨13893⟩⟩
def transferEvent : Nat := 27404
def frameStart : Nat := 27351
def rule : BoundRule := .identity (.predecessor 0 27403 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27403 .coefficient)
      LeftBound27401.bound (LeftBound27401.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27401.derived selector witness)

def rawBound : CoeffClass := LeftBound27401.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound27401.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27404

namespace LeftBound27410
def owner : Owner := ⟨.program ⟨214⟩, ⟨13894⟩⟩
def transferEvent : Nat := 27410
def frameStart : Nat := 27351
def rule : BoundRule := .product (.predecessor 0 27408 .coefficient) (.predecessor 1 27409 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27408 .coefficient)
      LeftAuthority27406.bound (LeftAuthority27406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27409 .coefficient)
      LeftBound27404.bound (LeftBound27404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority27406.bound LeftBound27404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27406.bound, LeftBound27404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority27406.actual selector witness) * (LeftBound27404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27410

namespace LeftBound27426
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 27426
def frameStart : Nat := 27351
def rule : BoundRule := .scale (.predecessor 0 27424 .coefficient) (.value (.predecessor 1 27425 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27424 .coefficient)
      LeftAuthority27422.bound (LeftAuthority27422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27425 .coefficient)
      LeftAuthority27413.bound (LeftAuthority27413.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority27413.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27422.bound LeftAuthority27413.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27422.bound, LeftAuthority27413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority27422.actual selector witness) * (LeftAuthority27413.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27426

namespace LeftBound27429
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 27429
def frameStart : Nat := 27351
def rule : BoundRule := .identity (.predecessor 0 27428 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27428 .coefficient)
      LeftAuthority27416.bound (LeftAuthority27416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27416.derived selector witness)

def rawBound : CoeffClass := LeftAuthority27416.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority27416.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27429

namespace LeftBound27433
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 27433
def frameStart : Nat := 27351
def rule : BoundRule := .product (.predecessor 0 27431 .coefficient) (.predecessor 1 27432 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27431 .coefficient)
      LeftBound27429.bound (LeftBound27429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27432 .coefficient)
      LeftBound27426.bound (LeftBound27426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27429.bound LeftBound27426.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27429.bound, LeftBound27426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27429.actual selector witness) * (LeftBound27426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27433

namespace LeftBound27438
def owner : Owner := ⟨.program ⟨214⟩, ⟨13895⟩⟩
def transferEvent : Nat := 27438
def frameStart : Nat := 27351
def rule : BoundRule := .sum [.predecessor 0 27436 .coefficient, .predecessor 1 27437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27436 .coefficient)
      LeftBound27433.bound (LeftBound27433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27437 .coefficient)
      LeftBound27410.bound (LeftBound27410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27433.bound, LeftBound27410.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27433.bound, LeftBound27410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27433.actual selector witness, LeftBound27410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27438

namespace LeftBound27442
def owner : Owner := ⟨.program ⟨214⟩, ⟨25930⟩⟩
def transferEvent : Nat := 27442
def frameStart : Nat := 27351
def rule : BoundRule := .product (.predecessor 0 27440 .coefficient) (.predecessor 1 27441 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27440 .coefficient)
      LeftBound27438.bound (LeftBound27438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27441 .coefficient)
      LeftAuthority27395.bound (LeftAuthority27395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27395.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27438.bound LeftAuthority27395.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27438.bound, LeftAuthority27395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27438.actual selector witness) * (LeftAuthority27395.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27442

namespace LeftBound27453
def owner : Owner := ⟨.program ⟨214⟩, ⟨15716⟩⟩
def transferEvent : Nat := 27453
def frameStart : Nat := 27351
def rule : BoundRule := .product (.predecessor 0 27451 .coefficient) (.predecessor 1 27452 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27451 .coefficient)
      LeftAuthority27406.bound (LeftAuthority27406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27452 .coefficient)
      LeftAuthority27449.bound (LeftAuthority27449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27406.bound LeftAuthority27449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27406.bound, LeftAuthority27449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority27406.actual selector witness) * (LeftAuthority27449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27453

namespace LeftBound27461
def owner : Owner := ⟨.program ⟨214⟩, ⟨15717⟩⟩
def transferEvent : Nat := 27461
def frameStart : Nat := 27351
def rule : BoundRule := .sum [.predecessor 0 27459 .coefficient, .predecessor 1 27460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27459 .coefficient)
      LeftAuthority27457.bound (LeftAuthority27457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27460 .coefficient)
      LeftBound27453.bound (LeftBound27453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27457.bound, LeftBound27453.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27457.bound, LeftBound27453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority27457.actual selector witness, LeftBound27453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27461

namespace LeftBound27465
def owner : Owner := ⟨.program ⟨214⟩, ⟨25931⟩⟩
def transferEvent : Nat := 27465
def frameStart : Nat := 27351
def rule : BoundRule := .sum [.predecessor 0 27463 .coefficient, .predecessor 1 27464 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27463 .coefficient)
      LeftBound27461.bound (LeftBound27461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27464 .coefficient)
      LeftBound27442.bound (LeftBound27442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27461.bound, LeftBound27442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27461.bound, LeftBound27442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27461.actual selector witness, LeftBound27442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27465

namespace LeftBound27478
def owner : Owner := ⟨.program ⟨214⟩, ⟨25929⟩⟩
def transferEvent : Nat := 27478
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27476 .coefficient, .predecessor 1 27477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27476 .coefficient)
      LeftBound27299.bound (LeftBound27299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27477 .coefficient)
      LeftBound27282.bound (LeftBound27282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27299.bound, LeftBound27282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27299.bound, LeftBound27282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27299.actual selector witness, LeftBound27282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27478

namespace LeftBound27481
def owner : Owner := ⟨.program ⟨214⟩, ⟨25929⟩⟩
def transferEvent : Nat := 27481
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27475 .summary, .result 27289 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27475 .summary)
      LeftBound27301.bound (LeftBound27301.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19399⟩⟩) (rawTerms := some (Proof.Events107.exact27475RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 27289 .summary)
      LeftBound27284.bound (LeftBound27284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25928⟩⟩) (rawTerms := some (Proof.Events106.exact27289RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27301.bound, LeftBound27284.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27301.bound, LeftBound27284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound27301.actual selector witness, LeftBound27284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27481

namespace LeftBound27485
def owner : Owner := ⟨.program ⟨214⟩, ⟨27473⟩⟩
def transferEvent : Nat := 27485
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27483 .coefficient) (.predecessor 1 27484 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 27483 .coefficient)
      LeftBound27478.bound (LeftBound27478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 27484 .coefficient)
      LeftAuthority27204.bound (LeftAuthority27204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27204.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27204.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound27478.bound LeftAuthority27204.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27478.bound, LeftAuthority27204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound27478.actual selector witness) * (LeftAuthority27204.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27485

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
