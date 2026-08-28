import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65495
def owner : Owner := ⟨.program ⟨214⟩, ⟨13443⟩⟩
def transferEvent : Nat := 65495
def frameStart : Nat := 65442
def rule : BoundRule := .identity (.predecessor 0 65494 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65494 .coefficient)
      LeftBound65492.bound (LeftBound65492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65492.derived selector witness)

def rawBound : CoeffClass := LeftBound65492.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound65492.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65495

namespace LeftBound65501
def owner : Owner := ⟨.program ⟨214⟩, ⟨13444⟩⟩
def transferEvent : Nat := 65501
def frameStart : Nat := 65442
def rule : BoundRule := .product (.predecessor 0 65499 .coefficient) (.predecessor 1 65500 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65499 .coefficient)
      LeftAuthority65497.bound (LeftAuthority65497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65500 .coefficient)
      LeftBound65495.bound (LeftBound65495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65495.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority65497.bound LeftBound65495.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65497.bound, LeftBound65495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority65497.actual selector witness) * (LeftBound65495.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65501

namespace LeftBound65517
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 65517
def frameStart : Nat := 65442
def rule : BoundRule := .scale (.predecessor 0 65515 .coefficient) (.value (.predecessor 1 65516 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65515 .coefficient)
      LeftAuthority65513.bound (LeftAuthority65513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65516 .coefficient)
      LeftAuthority65504.bound (LeftAuthority65504.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65504.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority65513.bound LeftAuthority65504.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65513.bound, LeftAuthority65504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65513.actual selector witness) * (LeftAuthority65504.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65517

namespace LeftBound65520
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 65520
def frameStart : Nat := 65442
def rule : BoundRule := .identity (.predecessor 0 65519 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65519 .coefficient)
      LeftAuthority65507.bound (LeftAuthority65507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65507.derived selector witness)

def rawBound : CoeffClass := LeftAuthority65507.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority65507.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65520

namespace LeftBound65524
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 65524
def frameStart : Nat := 65442
def rule : BoundRule := .product (.predecessor 0 65522 .coefficient) (.predecessor 1 65523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65522 .coefficient)
      LeftBound65520.bound (LeftBound65520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65523 .coefficient)
      LeftBound65517.bound (LeftBound65517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65520.bound LeftBound65517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65520.bound, LeftBound65517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65520.actual selector witness) * (LeftBound65517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65524

namespace LeftBound65529
def owner : Owner := ⟨.program ⟨214⟩, ⟨13445⟩⟩
def transferEvent : Nat := 65529
def frameStart : Nat := 65442
def rule : BoundRule := .sum [.predecessor 0 65527 .coefficient, .predecessor 1 65528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65527 .coefficient)
      LeftBound65524.bound (LeftBound65524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65528 .coefficient)
      LeftBound65501.bound (LeftBound65501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65501.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65524.bound, LeftBound65501.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65524.bound, LeftBound65501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65524.actual selector witness, LeftBound65501.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65529

namespace LeftBound65533
def owner : Owner := ⟨.program ⟨214⟩, ⟨25756⟩⟩
def transferEvent : Nat := 65533
def frameStart : Nat := 65442
def rule : BoundRule := .product (.predecessor 0 65531 .coefficient) (.predecessor 1 65532 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65531 .coefficient)
      LeftBound65529.bound (LeftBound65529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65532 .coefficient)
      LeftAuthority65486.bound (LeftAuthority65486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65486.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65529.bound LeftAuthority65486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65529.bound, LeftAuthority65486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65529.actual selector witness) * (LeftAuthority65486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65533

namespace LeftBound65544
def owner : Owner := ⟨.program ⟨214⟩, ⟨17009⟩⟩
def transferEvent : Nat := 65544
def frameStart : Nat := 65442
def rule : BoundRule := .product (.predecessor 0 65542 .coefficient) (.predecessor 1 65543 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65542 .coefficient)
      LeftAuthority65497.bound (LeftAuthority65497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65543 .coefficient)
      LeftAuthority65540.bound (LeftAuthority65540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65497.bound LeftAuthority65540.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65497.bound, LeftAuthority65540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority65497.actual selector witness) * (LeftAuthority65540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65544

namespace LeftBound65552
def owner : Owner := ⟨.program ⟨214⟩, ⟨17010⟩⟩
def transferEvent : Nat := 65552
def frameStart : Nat := 65442
def rule : BoundRule := .sum [.predecessor 0 65550 .coefficient, .predecessor 1 65551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65550 .coefficient)
      LeftAuthority65548.bound (LeftAuthority65548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65551 .coefficient)
      LeftBound65544.bound (LeftBound65544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65544.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority65548.bound, LeftBound65544.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65548.bound, LeftBound65544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority65548.actual selector witness, LeftBound65544.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65552

namespace LeftBound65556
def owner : Owner := ⟨.program ⟨214⟩, ⟨25757⟩⟩
def transferEvent : Nat := 65556
def frameStart : Nat := 65442
def rule : BoundRule := .sum [.predecessor 0 65554 .coefficient, .predecessor 1 65555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65554 .coefficient)
      LeftBound65552.bound (LeftBound65552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65555 .coefficient)
      LeftBound65533.bound (LeftBound65533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65552.bound, LeftBound65533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65552.bound, LeftBound65533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65552.actual selector witness, LeftBound65533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65556

namespace LeftBound65569
def owner : Owner := ⟨.program ⟨214⟩, ⟨25755⟩⟩
def transferEvent : Nat := 65569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65567 .coefficient, .predecessor 1 65568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65567 .coefficient)
      LeftBound65390.bound (LeftBound65390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65568 .coefficient)
      LeftBound65362.bound (LeftBound65362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65390.bound, LeftBound65362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65390.bound, LeftBound65362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65390.actual selector witness, LeftBound65362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65569

namespace LeftBound65572
def owner : Owner := ⟨.program ⟨214⟩, ⟨25755⟩⟩
def transferEvent : Nat := 65572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 65566 .summary, .result 65369 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65566 .summary)
      LeftBound65392.bound (LeftBound65392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20247⟩⟩) (rawTerms := some (Proof.Events256.exact65566RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65369 .summary)
      LeftBound65364.bound (LeftBound65364.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25754⟩⟩) (rawTerms := some (Proof.Events255.exact65369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65392.bound, LeftBound65364.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65392.bound, LeftBound65364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65392.actual selector witness, LeftBound65364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65572

namespace LeftBound65576
def owner : Owner := ⟨.program ⟨214⟩, ⟨30097⟩⟩
def transferEvent : Nat := 65576
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65574 .coefficient) (.predecessor 1 65575 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65574 .coefficient)
      LeftBound65569.bound (LeftBound65569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65575 .coefficient)
      LeftAuthority65279.bound (LeftAuthority65279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65279.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65569.bound LeftAuthority65279.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65569.bound, LeftAuthority65279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65569.actual selector witness) * (LeftAuthority65279.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65576

namespace LeftBound65577
def owner : Owner := ⟨.program ⟨214⟩, ⟨30097⟩⟩
def transferEvent : Nat := 65577
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30095⟩⟩]⟩ [⟨.result 65280 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65280 .coefficient)
      LeftAuthority65279.bound (LeftAuthority65279.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30095⟩⟩) (rawTerms := some (Proof.Events255.exact65280RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65279.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65279.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65279.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65577

namespace LeftBound65578
def owner : Owner := ⟨.program ⟨214⟩, ⟨30097⟩⟩
def transferEvent : Nat := 65578
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65573 .summary) (.transfer 65577) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65573 .summary)
      LeftBound65572.bound (LeftBound65572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25755⟩⟩) (rawTerms := some (Proof.Events256.exact65573RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 65577)
      LeftBound65577.bound (LeftBound65577.actual selector witness) := by
  exact .transfer (LeftBound65577.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65572.bound LeftBound65577.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65572.bound, LeftBound65577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65572.actual selector witness) * (LeftBound65577.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65578

namespace LeftBound65589
def owner : Owner := ⟨.program ⟨214⟩, ⟨22838⟩⟩
def transferEvent : Nat := 65589
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 65587 .coefficient) (.value (.predecessor 1 65588 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65587 .coefficient)
      LeftAuthority65585.bound (LeftAuthority65585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65588 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority65585.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65585.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65585.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65589

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
