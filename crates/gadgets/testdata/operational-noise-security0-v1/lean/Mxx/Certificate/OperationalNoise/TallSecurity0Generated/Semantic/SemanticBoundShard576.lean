import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard575

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84437
def owner : Owner := ⟨.program ⟨214⟩, ⟨14531⟩⟩
def transferEvent : Nat := 84437
def frameStart : Nat := 84387
def rule : BoundRule := .sum [.predecessor 0 84435 .coefficient, .predecessor 1 84436 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84435 .coefficient)
      LeftBound84420.bound (LeftBound84420.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84436 .coefficient)
      LeftAuthority84433.bound (LeftAuthority84433.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84420.bound, LeftAuthority84433.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84420.bound, LeftAuthority84433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84420.actual selector witness, LeftAuthority84433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84437

namespace LeftBound84440
def owner : Owner := ⟨.program ⟨214⟩, ⟨14532⟩⟩
def transferEvent : Nat := 84440
def frameStart : Nat := 84387
def rule : BoundRule := .identity (.predecessor 0 84439 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84439 .coefficient)
      LeftBound84437.bound (LeftBound84437.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84437.derived selector witness)

def rawBound : CoeffClass := LeftBound84437.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84437.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84440

namespace LeftBound84446
def owner : Owner := ⟨.program ⟨214⟩, ⟨14533⟩⟩
def transferEvent : Nat := 84446
def frameStart : Nat := 84387
def rule : BoundRule := .product (.predecessor 0 84444 .coefficient) (.predecessor 1 84445 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84444 .coefficient)
      LeftAuthority84442.bound (LeftAuthority84442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84445 .coefficient)
      LeftBound84440.bound (LeftBound84440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84440.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority84442.bound LeftBound84440.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84442.bound, LeftBound84440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority84442.actual selector witness) * (LeftBound84440.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84446

namespace LeftBound84460
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 84460
def frameStart : Nat := 84387
def rule : BoundRule := .scale (.predecessor 0 84458 .coefficient) (.value (.predecessor 1 84459 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84458 .coefficient)
      LeftAuthority84456.bound (LeftAuthority84456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84459 .coefficient)
      LeftAuthority84390.bound (LeftAuthority84390.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84390.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84456.bound LeftAuthority84390.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84456.bound, LeftAuthority84390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84456.actual selector witness) * (LeftAuthority84390.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84460

namespace LeftBound84463
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 84463
def frameStart : Nat := 84387
def rule : BoundRule := .identity (.predecessor 0 84462 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84462 .coefficient)
      LeftAuthority84450.bound (LeftAuthority84450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84450.derived selector witness)

def rawBound : CoeffClass := LeftAuthority84450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority84450.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84463

namespace LeftBound84467
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 84467
def frameStart : Nat := 84387
def rule : BoundRule := .product (.predecessor 0 84465 .coefficient) (.predecessor 1 84466 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84465 .coefficient)
      LeftBound84463.bound (LeftBound84463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84466 .coefficient)
      LeftBound84460.bound (LeftBound84460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84460.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84463.bound LeftBound84460.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84463.bound, LeftBound84460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84463.actual selector witness) * (LeftBound84460.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84467

namespace LeftBound84472
def owner : Owner := ⟨.program ⟨214⟩, ⟨14534⟩⟩
def transferEvent : Nat := 84472
def frameStart : Nat := 84387
def rule : BoundRule := .sum [.predecessor 0 84470 .coefficient, .predecessor 1 84471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84470 .coefficient)
      LeftBound84467.bound (LeftBound84467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84471 .coefficient)
      LeftBound84446.bound (LeftBound84446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84446.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84467.bound, LeftBound84446.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84467.bound, LeftBound84446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84467.actual selector witness, LeftBound84446.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84472

namespace LeftBound84476
def owner : Owner := ⟨.program ⟨214⟩, ⟨26146⟩⟩
def transferEvent : Nat := 84476
def frameStart : Nat := 84387
def rule : BoundRule := .product (.predecessor 0 84474 .coefficient) (.predecessor 1 84475 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84474 .coefficient)
      LeftBound84472.bound (LeftBound84472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84475 .coefficient)
      LeftAuthority84431.bound (LeftAuthority84431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84431.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84472.bound LeftAuthority84431.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84472.bound, LeftAuthority84431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84472.actual selector witness) * (LeftAuthority84431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84476

namespace LeftBound84487
def owner : Owner := ⟨.program ⟨214⟩, ⟨16061⟩⟩
def transferEvent : Nat := 84487
def frameStart : Nat := 84387
def rule : BoundRule := .product (.predecessor 0 84485 .coefficient) (.predecessor 1 84486 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84485 .coefficient)
      LeftAuthority84442.bound (LeftAuthority84442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84486 .coefficient)
      LeftAuthority84483.bound (LeftAuthority84483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84442.bound LeftAuthority84483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84442.bound, LeftAuthority84483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84442.actual selector witness) * (LeftAuthority84483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84487

namespace LeftBound84495
def owner : Owner := ⟨.program ⟨214⟩, ⟨16062⟩⟩
def transferEvent : Nat := 84495
def frameStart : Nat := 84387
def rule : BoundRule := .sum [.predecessor 0 84493 .coefficient, .predecessor 1 84494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84493 .coefficient)
      LeftAuthority84491.bound (LeftAuthority84491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84491.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84494 .coefficient)
      LeftBound84487.bound (LeftBound84487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84491.bound, LeftBound84487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84491.bound, LeftBound84487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84491.actual selector witness, LeftBound84487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84495

namespace LeftBound84499
def owner : Owner := ⟨.program ⟨214⟩, ⟨26147⟩⟩
def transferEvent : Nat := 84499
def frameStart : Nat := 84387
def rule : BoundRule := .sum [.predecessor 0 84497 .coefficient, .predecessor 1 84498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84497 .coefficient)
      LeftBound84495.bound (LeftBound84495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84498 .coefficient)
      LeftBound84476.bound (LeftBound84476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84476.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84495.bound, LeftBound84476.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84495.bound, LeftBound84476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84495.actual selector witness, LeftBound84476.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84499

namespace LeftBound84512
def owner : Owner := ⟨.program ⟨214⟩, ⟨26145⟩⟩
def transferEvent : Nat := 84512
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84510 .coefficient, .predecessor 1 84511 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84510 .coefficient)
      LeftBound84335.bound (LeftBound84335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84511 .coefficient)
      LeftBound84318.bound (LeftBound84318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84335.bound, LeftBound84318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84335.bound, LeftBound84318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84335.actual selector witness, LeftBound84318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84512

namespace LeftBound84515
def owner : Owner := ⟨.program ⟨214⟩, ⟨26145⟩⟩
def transferEvent : Nat := 84515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84509 .summary, .result 84325 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84509 .summary)
      LeftBound84337.bound (LeftBound84337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19603⟩⟩) (rawTerms := some (Proof.Events330.exact84509RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84325 .summary)
      LeftBound84320.bound (LeftBound84320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26144⟩⟩) (rawTerms := some (Proof.Events329.exact84325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84337.bound, LeftBound84320.bound]
def bound : CoeffClass := .finite ⟨352072932929536, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84337.bound, LeftBound84320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84337.actual selector witness, LeftBound84320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84515

namespace LeftBound84519
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def transferEvent : Nat := 84519
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84517 .coefficient) (.predecessor 1 84518 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84517 .coefficient)
      LeftBound84512.bound (LeftBound84512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84518 .coefficient)
      LeftAuthority84240.bound (LeftAuthority84240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84512.bound LeftAuthority84240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84512.bound, LeftAuthority84240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84512.actual selector witness) * (LeftAuthority84240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84519

namespace LeftBound84520
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def transferEvent : Nat := 84520
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28083⟩⟩]⟩ [⟨.result 84241 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84241 .coefficient)
      LeftAuthority84240.bound (LeftAuthority84240.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28083⟩⟩) (rawTerms := some (Proof.Events329.exact84241RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84240.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84240.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84240.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84240.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84520

namespace LeftBound84521
def owner : Owner := ⟨.program ⟨214⟩, ⟨28085⟩⟩
def transferEvent : Nat := 84521
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84516 .summary) (.transfer 84520) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84516 .summary)
      LeftBound84515.bound (LeftBound84515.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26145⟩⟩) (rawTerms := some (Proof.Events330.exact84516RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84520)
      LeftBound84520.bound (LeftBound84520.actual selector witness) := by
  exact .transfer (LeftBound84520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84515.bound LeftBound84520.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84515.bound, LeftBound84520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84515.actual selector witness) * (LeftBound84520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84521

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
