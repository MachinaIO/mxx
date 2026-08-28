import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard470

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69351
def owner : Owner := ⟨.program ⟨214⟩, ⟨14745⟩⟩
def transferEvent : Nat := 69351
def frameStart : Nat := 69298
def rule : BoundRule := .identity (.predecessor 0 69350 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69350 .coefficient)
      LeftBound69348.bound (LeftBound69348.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69348.derived selector witness)

def rawBound : CoeffClass := LeftBound69348.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound69348.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69351

namespace LeftBound69357
def owner : Owner := ⟨.program ⟨214⟩, ⟨14746⟩⟩
def transferEvent : Nat := 69357
def frameStart : Nat := 69298
def rule : BoundRule := .product (.predecessor 0 69355 .coefficient) (.predecessor 1 69356 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69355 .coefficient)
      LeftAuthority69353.bound (LeftAuthority69353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69356 .coefficient)
      LeftBound69351.bound (LeftBound69351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69351.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority69353.bound LeftBound69351.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69353.bound, LeftBound69351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority69353.actual selector witness) * (LeftBound69351.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69357

namespace LeftBound69373
def owner : Owner := ⟨.program ⟨214⟩, ⟨7859⟩⟩
def transferEvent : Nat := 69373
def frameStart : Nat := 69298
def rule : BoundRule := .scale (.predecessor 0 69371 .coefficient) (.value (.predecessor 1 69372 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69371 .coefficient)
      LeftAuthority69369.bound (LeftAuthority69369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69372 .coefficient)
      LeftAuthority69360.bound (LeftAuthority69360.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority69360.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority69369.bound LeftAuthority69360.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69369.bound, LeftAuthority69360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69369.actual selector witness) * (LeftAuthority69360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69373

namespace LeftBound69376
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 69376
def frameStart : Nat := 69298
def rule : BoundRule := .identity (.predecessor 0 69375 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69375 .coefficient)
      LeftAuthority69363.bound (LeftAuthority69363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69363.derived selector witness)

def rawBound : CoeffClass := LeftAuthority69363.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority69363.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69376

namespace LeftBound69380
def owner : Owner := ⟨.program ⟨214⟩, ⟨7860⟩⟩
def transferEvent : Nat := 69380
def frameStart : Nat := 69298
def rule : BoundRule := .product (.predecessor 0 69378 .coefficient) (.predecessor 1 69379 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69378 .coefficient)
      LeftBound69376.bound (LeftBound69376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69379 .coefficient)
      LeftBound69373.bound (LeftBound69373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69373.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69376.bound LeftBound69373.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69376.bound, LeftBound69373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69376.actual selector witness) * (LeftBound69373.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69380

namespace LeftBound69385
def owner : Owner := ⟨.program ⟨214⟩, ⟨14747⟩⟩
def transferEvent : Nat := 69385
def frameStart : Nat := 69298
def rule : BoundRule := .sum [.predecessor 0 69383 .coefficient, .predecessor 1 69384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69383 .coefficient)
      LeftBound69380.bound (LeftBound69380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69384 .coefficient)
      LeftBound69357.bound (LeftBound69357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69380.bound, LeftBound69357.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69380.bound, LeftBound69357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69380.actual selector witness, LeftBound69357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69385

namespace LeftBound69389
def owner : Owner := ⟨.program ⟨214⟩, ⟨26218⟩⟩
def transferEvent : Nat := 69389
def frameStart : Nat := 69298
def rule : BoundRule := .product (.predecessor 0 69387 .coefficient) (.predecessor 1 69388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69387 .coefficient)
      LeftBound69385.bound (LeftBound69385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69388 .coefficient)
      LeftAuthority69342.bound (LeftAuthority69342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69385.bound LeftAuthority69342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69385.bound, LeftAuthority69342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69385.actual selector witness) * (LeftAuthority69342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69389

namespace LeftBound69400
def owner : Owner := ⟨.program ⟨214⟩, ⟨16176⟩⟩
def transferEvent : Nat := 69400
def frameStart : Nat := 69298
def rule : BoundRule := .product (.predecessor 0 69398 .coefficient) (.predecessor 1 69399 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69398 .coefficient)
      LeftAuthority69353.bound (LeftAuthority69353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69399 .coefficient)
      LeftAuthority69396.bound (LeftAuthority69396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69396.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority69353.bound LeftAuthority69396.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69353.bound, LeftAuthority69396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority69353.actual selector witness) * (LeftAuthority69396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69400

namespace LeftBound69408
def owner : Owner := ⟨.program ⟨214⟩, ⟨16177⟩⟩
def transferEvent : Nat := 69408
def frameStart : Nat := 69298
def rule : BoundRule := .sum [.predecessor 0 69406 .coefficient, .predecessor 1 69407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69406 .coefficient)
      LeftAuthority69404.bound (LeftAuthority69404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69407 .coefficient)
      LeftBound69400.bound (LeftBound69400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69404.bound, LeftBound69400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69404.bound, LeftBound69400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority69404.actual selector witness, LeftBound69400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69408

namespace LeftBound69412
def owner : Owner := ⟨.program ⟨214⟩, ⟨26219⟩⟩
def transferEvent : Nat := 69412
def frameStart : Nat := 69298
def rule : BoundRule := .sum [.predecessor 0 69410 .coefficient, .predecessor 1 69411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69410 .coefficient)
      LeftBound69408.bound (LeftBound69408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69411 .coefficient)
      LeftBound69389.bound (LeftBound69389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69408.bound, LeftBound69389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69408.bound, LeftBound69389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69408.actual selector witness, LeftBound69389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69412

namespace LeftBound69425
def owner : Owner := ⟨.program ⟨214⟩, ⟨26217⟩⟩
def transferEvent : Nat := 69425
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69423 .coefficient, .predecessor 1 69424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69423 .coefficient)
      LeftBound69246.bound (LeftBound69246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69424 .coefficient)
      LeftBound69229.bound (LeftBound69229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69246.bound, LeftBound69229.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69246.bound, LeftBound69229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69246.actual selector witness, LeftBound69229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69425

namespace LeftBound69428
def owner : Owner := ⟨.program ⟨214⟩, ⟨26217⟩⟩
def transferEvent : Nat := 69428
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69422 .summary, .result 69236 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69422 .summary)
      LeftBound69248.bound (LeftBound69248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19671⟩⟩) (rawTerms := some (Proof.Events271.exact69422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69236 .summary)
      LeftBound69231.bound (LeftBound69231.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26216⟩⟩) (rawTerms := some (Proof.Events270.exact69236RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69231.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69248.bound, LeftBound69231.bound]
def bound : CoeffClass := .finite ⟨352091253649408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69248.bound, LeftBound69231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69248.actual selector witness, LeftBound69231.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69428

namespace LeftBound69432
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def transferEvent : Nat := 69432
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69430 .coefficient) (.predecessor 1 69431 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69430 .coefficient)
      LeftBound69425.bound (LeftBound69425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69431 .coefficient)
      LeftAuthority69151.bound (LeftAuthority69151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events270.exact69152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69151.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69425.bound LeftAuthority69151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69425.bound, LeftAuthority69151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69425.actual selector witness) * (LeftAuthority69151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69432

namespace LeftBound69433
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def transferEvent : Nat := 69433
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28287⟩⟩]⟩ [⟨.result 69152 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69152 .coefficient)
      LeftAuthority69151.bound (LeftAuthority69151.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28287⟩⟩) (rawTerms := some (Proof.Events270.exact69152RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69151.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69151.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69151.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69433

namespace LeftBound69434
def owner : Owner := ⟨.program ⟨214⟩, ⟨28289⟩⟩
def transferEvent : Nat := 69434
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 69429 .summary) (.transfer 69433) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69429 .summary)
      LeftBound69428.bound (LeftBound69428.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26217⟩⟩) (rawTerms := some (Proof.Events271.exact69429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69433)
      LeftBound69433.bound (LeftBound69433.actual selector witness) := by
  exact .transfer (LeftBound69433.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69428.bound LeftBound69433.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69428.bound, LeftBound69433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69428.actual selector witness) * (LeftBound69433.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69434

namespace LeftBound69445
def owner : Owner := ⟨.program ⟨214⟩, ⟨21686⟩⟩
def transferEvent : Nat := 69445
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 69443 .coefficient) (.value (.predecessor 1 69444 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69443 .coefficient)
      LeftAuthority69441.bound (LeftAuthority69441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69444 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority69441.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69441.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69441.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69445

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
