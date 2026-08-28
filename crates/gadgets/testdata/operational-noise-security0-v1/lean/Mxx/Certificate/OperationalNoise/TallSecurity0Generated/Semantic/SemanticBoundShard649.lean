import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard648

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95390
def owner : Owner := ⟨.program ⟨214⟩, ⟨12935⟩⟩
def transferEvent : Nat := 95390
def frameStart : Nat := 95373
def rule : BoundRule := .product (.predecessor 0 95388 .coefficient) (.predecessor 1 95389 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95388 .coefficient)
      LeftAuthority95386.bound (LeftAuthority95386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95389 .coefficient)
      LeftAuthority95383.bound (LeftAuthority95383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95383.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95383.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95386.bound LeftAuthority95383.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95386.bound, LeftAuthority95383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95386.actual selector witness) * (LeftAuthority95383.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95390

namespace LeftBound95394
def owner : Owner := ⟨.program ⟨214⟩, ⟨12936⟩⟩
def transferEvent : Nat := 95394
def frameStart : Nat := 95373
def rule : BoundRule := .identity (.predecessor 0 95393 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95393 .coefficient)
      LeftBound95390.bound (LeftBound95390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95390.derived selector witness)

def rawBound : CoeffClass := LeftBound95390.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95390.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95394

namespace LeftBound95411
def owner : Owner := ⟨.program ⟨214⟩, ⟨13046⟩⟩
def transferEvent : Nat := 95411
def frameStart : Nat := 95373
def rule : BoundRule := .sum [.predecessor 0 95409 .coefficient, .predecessor 1 95410 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95409 .coefficient)
      LeftBound95394.bound (LeftBound95394.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95410 .coefficient)
      LeftAuthority95407.bound (LeftAuthority95407.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95394.bound, LeftAuthority95407.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95394.bound, LeftAuthority95407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95394.actual selector witness, LeftAuthority95407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95411

namespace LeftBound95414
def owner : Owner := ⟨.program ⟨214⟩, ⟨13047⟩⟩
def transferEvent : Nat := 95414
def frameStart : Nat := 95373
def rule : BoundRule := .identity (.predecessor 0 95413 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95413 .coefficient)
      LeftBound95411.bound (LeftBound95411.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound95411.derived selector witness)

def rawBound : CoeffClass := LeftBound95411.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound95411.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95414

namespace LeftBound95420
def owner : Owner := ⟨.program ⟨214⟩, ⟨13048⟩⟩
def transferEvent : Nat := 95420
def frameStart : Nat := 95373
def rule : BoundRule := .product (.predecessor 0 95418 .coefficient) (.predecessor 1 95419 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95418 .coefficient)
      LeftAuthority95416.bound (LeftAuthority95416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95419 .coefficient)
      LeftBound95414.bound (LeftBound95414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority95416.bound LeftBound95414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95416.bound, LeftBound95414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority95416.actual selector witness) * (LeftBound95414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95420

namespace LeftBound95436
def owner : Owner := ⟨.program ⟨214⟩, ⟨7877⟩⟩
def transferEvent : Nat := 95436
def frameStart : Nat := 95373
def rule : BoundRule := .scale (.predecessor 0 95434 .coefficient) (.value (.predecessor 1 95435 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95434 .coefficient)
      LeftAuthority95432.bound (LeftAuthority95432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95435 .coefficient)
      LeftAuthority95423.bound (LeftAuthority95423.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority95423.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95432.bound LeftAuthority95423.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95432.bound, LeftAuthority95423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95432.actual selector witness) * (LeftAuthority95423.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95436

namespace LeftBound95439
def owner : Owner := ⟨.program ⟨214⟩, ⟨6768⟩⟩
def transferEvent : Nat := 95439
def frameStart : Nat := 95373
def rule : BoundRule := .identity (.predecessor 0 95438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95438 .coefficient)
      LeftAuthority95426.bound (LeftAuthority95426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95426.derived selector witness)

def rawBound : CoeffClass := LeftAuthority95426.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority95426.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound95439

namespace LeftBound95443
def owner : Owner := ⟨.program ⟨214⟩, ⟨7878⟩⟩
def transferEvent : Nat := 95443
def frameStart : Nat := 95373
def rule : BoundRule := .product (.predecessor 0 95441 .coefficient) (.predecessor 1 95442 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95441 .coefficient)
      LeftBound95439.bound (LeftBound95439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95442 .coefficient)
      LeftBound95436.bound (LeftBound95436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95436.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95439.bound LeftBound95436.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95439.bound, LeftBound95436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95439.actual selector witness) * (LeftBound95436.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95443

namespace LeftBound95448
def owner : Owner := ⟨.program ⟨214⟩, ⟨13049⟩⟩
def transferEvent : Nat := 95448
def frameStart : Nat := 95373
def rule : BoundRule := .sum [.predecessor 0 95446 .coefficient, .predecessor 1 95447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95446 .coefficient)
      LeftBound95443.bound (LeftBound95443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95447 .coefficient)
      LeftBound95420.bound (LeftBound95420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95443.bound, LeftBound95420.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95443.bound, LeftBound95420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95443.actual selector witness, LeftBound95420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95448

namespace LeftBound95452
def owner : Owner := ⟨.program ⟨214⟩, ⟨25594⟩⟩
def transferEvent : Nat := 95452
def frameStart : Nat := 95373
def rule : BoundRule := .product (.predecessor 0 95450 .coefficient) (.predecessor 1 95451 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95450 .coefficient)
      LeftBound95448.bound (LeftBound95448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95451 .coefficient)
      LeftAuthority95405.bound (LeftAuthority95405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95405.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95448.bound LeftAuthority95405.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95448.bound, LeftAuthority95405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95448.actual selector witness) * (LeftAuthority95405.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95452

namespace LeftBound95463
def owner : Owner := ⟨.program ⟨214⟩, ⟨16744⟩⟩
def transferEvent : Nat := 95463
def frameStart : Nat := 95373
def rule : BoundRule := .product (.predecessor 0 95461 .coefficient) (.predecessor 1 95462 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95461 .coefficient)
      LeftAuthority95416.bound (LeftAuthority95416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95462 .coefficient)
      LeftAuthority95459.bound (LeftAuthority95459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95459.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority95416.bound LeftAuthority95459.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95416.bound, LeftAuthority95459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority95416.actual selector witness) * (LeftAuthority95459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95463

namespace LeftBound95471
def owner : Owner := ⟨.program ⟨214⟩, ⟨16745⟩⟩
def transferEvent : Nat := 95471
def frameStart : Nat := 95373
def rule : BoundRule := .sum [.predecessor 0 95469 .coefficient, .predecessor 1 95470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95469 .coefficient)
      LeftAuthority95467.bound (LeftAuthority95467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95470 .coefficient)
      LeftBound95463.bound (LeftBound95463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority95467.bound, LeftBound95463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95467.bound, LeftBound95463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority95467.actual selector witness, LeftBound95463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95471

namespace LeftBound95475
def owner : Owner := ⟨.program ⟨214⟩, ⟨25595⟩⟩
def transferEvent : Nat := 95475
def frameStart : Nat := 95373
def rule : BoundRule := .sum [.predecessor 0 95473 .coefficient, .predecessor 1 95474 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95473 .coefficient)
      LeftBound95471.bound (LeftBound95471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95474 .coefficient)
      LeftBound95452.bound (LeftBound95452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95471.bound, LeftBound95452.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95471.bound, LeftBound95452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95471.actual selector witness, LeftBound95452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95475

namespace LeftBound95488
def owner : Owner := ⟨.program ⟨214⟩, ⟨25593⟩⟩
def transferEvent : Nat := 95488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95486 .coefficient, .predecessor 1 95487 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95486 .coefficient)
      LeftBound95333.bound (LeftBound95333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95487 .coefficient)
      LeftBound95316.bound (LeftBound95316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95316.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95333.bound, LeftBound95316.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95333.bound, LeftBound95316.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95333.actual selector witness, LeftBound95316.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95488

namespace LeftBound95491
def owner : Owner := ⟨.program ⟨214⟩, ⟨25593⟩⟩
def transferEvent : Nat := 95491
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95485 .summary, .result 95323 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95485 .summary)
      LeftBound95335.bound (LeftBound95335.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20096⟩⟩) (rawTerms := some (Proof.Events372.exact95485RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95323 .summary)
      LeftBound95318.bound (LeftBound95318.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25592⟩⟩) (rawTerms := some (Proof.Events372.exact95323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95335.bound, LeftBound95318.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95335.bound, LeftBound95318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95335.actual selector witness, LeftBound95318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95491

namespace LeftBound95495
def owner : Owner := ⟨.program ⟨214⟩, ⟨29569⟩⟩
def transferEvent : Nat := 95495
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95493 .coefficient) (.predecessor 1 95494 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95493 .coefficient)
      LeftBound95488.bound (LeftBound95488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95494 .coefficient)
      LeftAuthority95238.bound (LeftAuthority95238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95238.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95488.bound LeftAuthority95238.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95488.bound, LeftAuthority95238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95488.actual selector witness) * (LeftAuthority95238.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95495

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
