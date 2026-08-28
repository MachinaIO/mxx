import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard151

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound23524
def owner : Owner := ⟨.program ⟨214⟩, ⟨12591⟩⟩
def transferEvent : Nat := 23524
def frameStart : Nat := 23495
def rule : BoundRule := .product (.predecessor 0 23522 .coefficient) (.predecessor 1 23523 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23522 .coefficient)
      LeftAuthority23520.bound (LeftAuthority23520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23523 .coefficient)
      LeftAuthority23517.bound (LeftAuthority23517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23520.bound LeftAuthority23517.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23520.bound, LeftAuthority23517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23520.actual selector witness) * (LeftAuthority23517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23524

namespace LeftBound23528
def owner : Owner := ⟨.program ⟨214⟩, ⟨12592⟩⟩
def transferEvent : Nat := 23528
def frameStart : Nat := 23495
def rule : BoundRule := .identity (.predecessor 0 23527 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23527 .coefficient)
      LeftBound23524.bound (LeftBound23524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23524.derived selector witness)

def rawBound : CoeffClass := LeftBound23524.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23524.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23528

namespace LeftBound23545
def owner : Owner := ⟨.program ⟨214⟩, ⟨12674⟩⟩
def transferEvent : Nat := 23545
def frameStart : Nat := 23495
def rule : BoundRule := .sum [.predecessor 0 23543 .coefficient, .predecessor 1 23544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23543 .coefficient)
      LeftBound23528.bound (LeftBound23528.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23544 .coefficient)
      LeftAuthority23541.bound (LeftAuthority23541.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23528.bound, LeftAuthority23541.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23528.bound, LeftAuthority23541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23528.actual selector witness, LeftAuthority23541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23545

namespace LeftBound23548
def owner : Owner := ⟨.program ⟨214⟩, ⟨12675⟩⟩
def transferEvent : Nat := 23548
def frameStart : Nat := 23495
def rule : BoundRule := .identity (.predecessor 0 23547 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23547 .coefficient)
      LeftBound23545.bound (LeftBound23545.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23545.derived selector witness)

def rawBound : CoeffClass := LeftBound23545.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23545.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23548

namespace LeftBound23554
def owner : Owner := ⟨.program ⟨214⟩, ⟨12676⟩⟩
def transferEvent : Nat := 23554
def frameStart : Nat := 23495
def rule : BoundRule := .product (.predecessor 0 23552 .coefficient) (.predecessor 1 23553 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23552 .coefficient)
      LeftAuthority23550.bound (LeftAuthority23550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23553 .coefficient)
      LeftBound23548.bound (LeftBound23548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority23550.bound LeftBound23548.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23550.bound, LeftBound23548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority23550.actual selector witness) * (LeftBound23548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23554

namespace LeftBound23570
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 23570
def frameStart : Nat := 23495
def rule : BoundRule := .scale (.predecessor 0 23568 .coefficient) (.value (.predecessor 1 23569 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23568 .coefficient)
      LeftAuthority23566.bound (LeftAuthority23566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23566.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23569 .coefficient)
      LeftAuthority23557.bound (LeftAuthority23557.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23557.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23566.bound LeftAuthority23557.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23566.bound, LeftAuthority23557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23566.actual selector witness) * (LeftAuthority23557.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23570

namespace LeftBound23573
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 23573
def frameStart : Nat := 23495
def rule : BoundRule := .identity (.predecessor 0 23572 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23572 .coefficient)
      LeftAuthority23560.bound (LeftAuthority23560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23560.derived selector witness)

def rawBound : CoeffClass := LeftAuthority23560.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority23560.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23573

namespace LeftBound23577
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 23577
def frameStart : Nat := 23495
def rule : BoundRule := .product (.predecessor 0 23575 .coefficient) (.predecessor 1 23576 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23575 .coefficient)
      LeftBound23573.bound (LeftBound23573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23576 .coefficient)
      LeftBound23570.bound (LeftBound23570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23570.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23573.bound LeftBound23570.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23573.bound, LeftBound23570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23573.actual selector witness) * (LeftBound23570.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23577

namespace LeftBound23582
def owner : Owner := ⟨.program ⟨214⟩, ⟨12677⟩⟩
def transferEvent : Nat := 23582
def frameStart : Nat := 23495
def rule : BoundRule := .sum [.predecessor 0 23580 .coefficient, .predecessor 1 23581 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23580 .coefficient)
      LeftBound23577.bound (LeftBound23577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23581 .coefficient)
      LeftBound23554.bound (LeftBound23554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23577.bound, LeftBound23554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23577.bound, LeftBound23554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23577.actual selector witness, LeftBound23554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23582

namespace LeftBound23586
def owner : Owner := ⟨.program ⟨214⟩, ⟨25468⟩⟩
def transferEvent : Nat := 23586
def frameStart : Nat := 23495
def rule : BoundRule := .product (.predecessor 0 23584 .coefficient) (.predecessor 1 23585 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23584 .coefficient)
      LeftBound23582.bound (LeftBound23582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23585 .coefficient)
      LeftAuthority23539.bound (LeftAuthority23539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23582.bound LeftAuthority23539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23582.bound, LeftAuthority23539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23582.actual selector witness) * (LeftAuthority23539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23586

namespace LeftBound23597
def owner : Owner := ⟨.program ⟨214⟩, ⟨16563⟩⟩
def transferEvent : Nat := 23597
def frameStart : Nat := 23495
def rule : BoundRule := .product (.predecessor 0 23595 .coefficient) (.predecessor 1 23596 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23595 .coefficient)
      LeftAuthority23550.bound (LeftAuthority23550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23596 .coefficient)
      LeftAuthority23593.bound (LeftAuthority23593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23593.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23550.bound LeftAuthority23593.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23550.bound, LeftAuthority23593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23550.actual selector witness) * (LeftAuthority23593.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23597

namespace LeftBound23605
def owner : Owner := ⟨.program ⟨214⟩, ⟨16564⟩⟩
def transferEvent : Nat := 23605
def frameStart : Nat := 23495
def rule : BoundRule := .sum [.predecessor 0 23603 .coefficient, .predecessor 1 23604 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23603 .coefficient)
      LeftAuthority23601.bound (LeftAuthority23601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23604 .coefficient)
      LeftBound23597.bound (LeftBound23597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23597.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority23601.bound, LeftBound23597.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23601.bound, LeftBound23597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority23601.actual selector witness, LeftBound23597.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23605

namespace LeftBound23609
def owner : Owner := ⟨.program ⟨214⟩, ⟨25469⟩⟩
def transferEvent : Nat := 23609
def frameStart : Nat := 23495
def rule : BoundRule := .sum [.predecessor 0 23607 .coefficient, .predecessor 1 23608 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23607 .coefficient)
      LeftBound23605.bound (LeftBound23605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23608 .coefficient)
      LeftBound23586.bound (LeftBound23586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23586.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23605.bound, LeftBound23586.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23605.bound, LeftBound23586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23605.actual selector witness, LeftBound23586.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23609

namespace LeftBound23622
def owner : Owner := ⟨.program ⟨214⟩, ⟨25467⟩⟩
def transferEvent : Nat := 23622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 23620 .coefficient, .predecessor 1 23621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23620 .coefficient)
      LeftBound23443.bound (LeftBound23443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23443.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23443.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23621 .coefficient)
      LeftBound23426.bound (LeftBound23426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23443.bound, LeftBound23426.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23443.bound, LeftBound23426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23443.actual selector witness, LeftBound23426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23622

namespace LeftBound23625
def owner : Owner := ⟨.program ⟨214⟩, ⟨25467⟩⟩
def transferEvent : Nat := 23625
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 23619 .summary, .result 23433 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23619 .summary)
      LeftBound23445.bound (LeftBound23445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19975⟩⟩) (rawTerms := some (Proof.Events092.exact23619RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23433 .summary)
      LeftBound23428.bound (LeftBound23428.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25466⟩⟩) (rawTerms := some (Proof.Events091.exact23433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23445.bound, LeftBound23428.bound]
def bound : CoeffClass := .finite ⟨352134001995776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23445.bound, LeftBound23428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23445.actual selector witness, LeftBound23428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23625

namespace LeftBound23629
def owner : Owner := ⟨.program ⟨214⟩, ⟨29209⟩⟩
def transferEvent : Nat := 23629
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 23627 .coefficient) (.predecessor 1 23628 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23627 .coefficient)
      LeftBound23622.bound (LeftBound23622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23628 .coefficient)
      LeftAuthority23348.bound (LeftAuthority23348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events091.exact23349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23348.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23622.bound LeftAuthority23348.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23622.bound, LeftAuthority23348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23622.actual selector witness) * (LeftAuthority23348.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23629

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
