import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard603
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard605

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound88372
def owner : Owner := ⟨.program ⟨214⟩, ⟨20394⟩⟩
def transferEvent : Nat := 88372
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 88370 .coefficient) (.value (.predecessor 1 88371 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88370 .coefficient)
      LeftAuthority88368.bound (LeftAuthority88368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88371 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority88368.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88368.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88368.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound88372

namespace LeftBound88376
def owner : Owner := ⟨.program ⟨214⟩, ⟨20395⟩⟩
def transferEvent : Nat := 88376
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88374 .coefficient) (.predecessor 1 88375 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88374 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88375 .coefficient)
      LeftBound88372.bound (LeftBound88372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound88372.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound88372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound88372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88376

namespace LeftBound88377
def owner : Owner := ⟨.program ⟨214⟩, ⟨20395⟩⟩
def transferEvent : Nat := 88377
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20392⟩⟩]⟩ [⟨.result 88369 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88369 .coefficient)
      LeftAuthority88368.bound (LeftAuthority88368.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20392⟩⟩) (rawTerms := some (Proof.Events345.exact88369RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88368.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority88368.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority88368.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88377

namespace LeftBound88378
def owner : Owner := ⟨.program ⟨214⟩, ⟨20395⟩⟩
def transferEvent : Nat := 88378
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 88377) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 88377)
      LeftBound88377.bound (LeftBound88377.actual selector witness) := by
  exact .transfer (LeftBound88377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound88377.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound88377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound88377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88378

namespace LeftBound88473
def owner : Owner := ⟨.program ⟨214⟩, ⟨14793⟩⟩
def transferEvent : Nat := 88473
def frameStart : Nat := 88434
def rule : BoundRule := .identity (.predecessor 0 88472 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88472 .coefficient)
      LeftAuthority88470.bound (LeftAuthority88470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88470.derived selector witness)

def rawBound : CoeffClass := LeftAuthority88470.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority88470.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88473

namespace LeftBound88490
def owner : Owner := ⟨.program ⟨214⟩, ⟨14832⟩⟩
def transferEvent : Nat := 88490
def frameStart : Nat := 88434
def rule : BoundRule := .sum [.predecessor 0 88488 .coefficient, .predecessor 1 88489 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88488 .coefficient)
      LeftBound88473.bound (LeftBound88473.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88489 .coefficient)
      LeftAuthority88486.bound (LeftAuthority88486.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority88486.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88473.bound, LeftAuthority88486.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88473.bound, LeftAuthority88486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88473.actual selector witness, LeftAuthority88486.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88490

namespace LeftBound88493
def owner : Owner := ⟨.program ⟨214⟩, ⟨14833⟩⟩
def transferEvent : Nat := 88493
def frameStart : Nat := 88434
def rule : BoundRule := .identity (.predecessor 0 88492 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88492 .coefficient)
      LeftBound88490.bound (LeftBound88490.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88490.derived selector witness)

def rawBound : CoeffClass := LeftBound88490.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound88490.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88493

namespace LeftBound88499
def owner : Owner := ⟨.program ⟨214⟩, ⟨14834⟩⟩
def transferEvent : Nat := 88499
def frameStart : Nat := 88434
def rule : BoundRule := .product (.predecessor 0 88497 .coefficient) (.predecessor 1 88498 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88497 .coefficient)
      LeftAuthority88495.bound (LeftAuthority88495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88498 .coefficient)
      LeftBound88493.bound (LeftBound88493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88493.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88493.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority88495.bound LeftBound88493.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88495.bound, LeftBound88493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority88495.actual selector witness) * (LeftBound88493.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88499

namespace LeftBound88507
def owner : Owner := ⟨.program ⟨214⟩, ⟨14835⟩⟩
def transferEvent : Nat := 88507
def frameStart : Nat := 88434
def rule : BoundRule := .sum [.predecessor 0 88505 .coefficient, .predecessor 1 88506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88505 .coefficient)
      LeftAuthority88503.bound (LeftAuthority88503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88506 .coefficient)
      LeftBound88499.bound (LeftBound88499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88499.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88503.bound, LeftBound88499.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88503.bound, LeftBound88499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority88503.actual selector witness, LeftBound88499.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88507

namespace LeftBound88511
def owner : Owner := ⟨.program ⟨214⟩, ⟨26359⟩⟩
def transferEvent : Nat := 88511
def frameStart : Nat := 88434
def rule : BoundRule := .product (.predecessor 0 88509 .coefficient) (.predecessor 1 88510 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88509 .coefficient)
      LeftBound88507.bound (LeftBound88507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88510 .coefficient)
      LeftAuthority88484.bound (LeftAuthority88484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88484.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88507.bound LeftAuthority88484.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88507.bound, LeftAuthority88484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88507.actual selector witness) * (LeftAuthority88484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88511

namespace LeftBound88522
def owner : Owner := ⟨.program ⟨214⟩, ⟨15266⟩⟩
def transferEvent : Nat := 88522
def frameStart : Nat := 88434
def rule : BoundRule := .product (.predecessor 0 88520 .coefficient) (.predecessor 1 88521 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88520 .coefficient)
      LeftAuthority88495.bound (LeftAuthority88495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88521 .coefficient)
      LeftAuthority88518.bound (LeftAuthority88518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88518.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority88495.bound LeftAuthority88518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88495.bound, LeftAuthority88518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority88495.actual selector witness) * (LeftAuthority88518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88522

namespace LeftBound88530
def owner : Owner := ⟨.program ⟨214⟩, ⟨15267⟩⟩
def transferEvent : Nat := 88530
def frameStart : Nat := 88434
def rule : BoundRule := .sum [.predecessor 0 88528 .coefficient, .predecessor 1 88529 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88528 .coefficient)
      LeftAuthority88526.bound (LeftAuthority88526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88529 .coefficient)
      LeftBound88522.bound (LeftBound88522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88522.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88526.bound, LeftBound88522.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88526.bound, LeftBound88522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority88526.actual selector witness, LeftBound88522.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88530

namespace LeftBound88534
def owner : Owner := ⟨.program ⟨214⟩, ⟨26362⟩⟩
def transferEvent : Nat := 88534
def frameStart : Nat := 88434
def rule : BoundRule := .sum [.predecessor 0 88532 .coefficient, .predecessor 1 88533 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88532 .coefficient)
      LeftBound88530.bound (LeftBound88530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88533 .coefficient)
      LeftBound88511.bound (LeftBound88511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88530.bound, LeftBound88511.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88530.bound, LeftBound88511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88530.actual selector witness, LeftBound88511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88534

namespace LeftBound88547
def owner : Owner := ⟨.program ⟨214⟩, ⟨26361⟩⟩
def transferEvent : Nat := 88547
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88545 .coefficient, .predecessor 1 88546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88545 .coefficient)
      LeftBound88376.bound (LeftBound88376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88546 .coefficient)
      LeftBound88359.bound (LeftBound88359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88376.bound, LeftBound88359.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88376.bound, LeftBound88359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88376.actual selector witness, LeftBound88359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88547

namespace LeftBound88550
def owner : Owner := ⟨.program ⟨214⟩, ⟨26361⟩⟩
def transferEvent : Nat := 88550
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88544 .summary, .result 88366 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88544 .summary)
      LeftBound88378.bound (LeftBound88378.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20395⟩⟩) (rawTerms := some (Proof.Events345.exact88544RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88366 .summary)
      LeftBound88361.bound (LeftBound88361.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26360⟩⟩) (rawTerms := some (Proof.Events345.exact88366RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88361.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88378.bound, LeftBound88361.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88378.bound, LeftBound88361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88378.actual selector witness, LeftBound88361.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88550

namespace LeftBound88554
def owner : Owner := ⟨.program ⟨214⟩, ⟨26568⟩⟩
def transferEvent : Nat := 88554
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88552 .coefficient, .predecessor 1 88553 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 88552 .coefficient)
      LeftBound88547.bound (LeftBound88547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 88553 .coefficient)
      LeftBound88067.bound (LeftBound88067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88547.bound, LeftBound88067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88547.bound, LeftBound88067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound88547.actual selector witness, LeftBound88067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88554

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
