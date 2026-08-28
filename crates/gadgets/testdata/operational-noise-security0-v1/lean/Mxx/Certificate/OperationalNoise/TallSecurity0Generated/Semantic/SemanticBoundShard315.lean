import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard314

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47429
def owner : Owner := ⟨.program ⟨214⟩, ⟨22059⟩⟩
def transferEvent : Nat := 47429
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47427 .coefficient) (.predecessor 1 47428 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47427 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47428 .coefficient)
      LeftBound47425.bound (LeftBound47425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47425.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound47425.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound47425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound47425.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47429

namespace LeftBound47430
def owner : Owner := ⟨.program ⟨214⟩, ⟨22059⟩⟩
def transferEvent : Nat := 47430
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22056⟩⟩]⟩ [⟨.result 47422 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47422 .coefficient)
      LeftAuthority47421.bound (LeftAuthority47421.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22056⟩⟩) (rawTerms := some (Proof.Events185.exact47422RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47421.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47421.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47421.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47430

namespace LeftBound47431
def owner : Owner := ⟨.program ⟨214⟩, ⟨22059⟩⟩
def transferEvent : Nat := 47431
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 47430) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47430)
      LeftBound47430.bound (LeftBound47430.actual selector witness) := by
  exact .transfer (LeftBound47430.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound47430.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound47430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound47430.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47431

namespace LeftBound47526
def owner : Owner := ⟨.program ⟨214⟩, ⟨16474⟩⟩
def transferEvent : Nat := 47526
def frameStart : Nat := 47487
def rule : BoundRule := .identity (.predecessor 0 47525 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47525 .coefficient)
      LeftAuthority47523.bound (LeftAuthority47523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47523.derived selector witness)

def rawBound : CoeffClass := LeftAuthority47523.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority47523.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47526

namespace LeftBound47543
def owner : Owner := ⟨.program ⟨214⟩, ⟨16513⟩⟩
def transferEvent : Nat := 47543
def frameStart : Nat := 47487
def rule : BoundRule := .sum [.predecessor 0 47541 .coefficient, .predecessor 1 47542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47541 .coefficient)
      LeftBound47526.bound (LeftBound47526.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47542 .coefficient)
      LeftAuthority47539.bound (LeftAuthority47539.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority47539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47526.bound, LeftAuthority47539.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47526.bound, LeftAuthority47539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47526.actual selector witness, LeftAuthority47539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47543

namespace LeftBound47546
def owner : Owner := ⟨.program ⟨214⟩, ⟨16514⟩⟩
def transferEvent : Nat := 47546
def frameStart : Nat := 47487
def rule : BoundRule := .identity (.predecessor 0 47545 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47545 .coefficient)
      LeftBound47543.bound (LeftBound47543.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47543.derived selector witness)

def rawBound : CoeffClass := LeftBound47543.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound47543.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47546

namespace LeftBound47552
def owner : Owner := ⟨.program ⟨214⟩, ⟨16515⟩⟩
def transferEvent : Nat := 47552
def frameStart : Nat := 47487
def rule : BoundRule := .product (.predecessor 0 47550 .coefficient) (.predecessor 1 47551 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47550 .coefficient)
      LeftAuthority47548.bound (LeftAuthority47548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47551 .coefficient)
      LeftBound47546.bound (LeftBound47546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority47548.bound LeftBound47546.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47548.bound, LeftBound47546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority47548.actual selector witness) * (LeftBound47546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47552

namespace LeftBound47560
def owner : Owner := ⟨.program ⟨214⟩, ⟨16516⟩⟩
def transferEvent : Nat := 47560
def frameStart : Nat := 47487
def rule : BoundRule := .sum [.predecessor 0 47558 .coefficient, .predecessor 1 47559 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47558 .coefficient)
      LeftAuthority47556.bound (LeftAuthority47556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47559 .coefficient)
      LeftBound47552.bound (LeftBound47552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47556.bound, LeftBound47552.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47556.bound, LeftBound47552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47556.actual selector witness, LeftBound47552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47560

namespace LeftBound47564
def owner : Owner := ⟨.program ⟨214⟩, ⟨28971⟩⟩
def transferEvent : Nat := 47564
def frameStart : Nat := 47487
def rule : BoundRule := .product (.predecessor 0 47562 .coefficient) (.predecessor 1 47563 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47562 .coefficient)
      LeftBound47560.bound (LeftBound47560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47560.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47563 .coefficient)
      LeftAuthority47537.bound (LeftAuthority47537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47560.bound LeftAuthority47537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47560.bound, LeftAuthority47537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47560.actual selector witness) * (LeftAuthority47537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47564

namespace LeftBound47575
def owner : Owner := ⟨.program ⟨214⟩, ⟨17560⟩⟩
def transferEvent : Nat := 47575
def frameStart : Nat := 47487
def rule : BoundRule := .product (.predecessor 0 47573 .coefficient) (.predecessor 1 47574 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47573 .coefficient)
      LeftAuthority47548.bound (LeftAuthority47548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47574 .coefficient)
      LeftAuthority47571.bound (LeftAuthority47571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47571.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47571.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47548.bound LeftAuthority47571.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47548.bound, LeftAuthority47571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority47548.actual selector witness) * (LeftAuthority47571.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47575

namespace LeftBound47583
def owner : Owner := ⟨.program ⟨214⟩, ⟨17561⟩⟩
def transferEvent : Nat := 47583
def frameStart : Nat := 47487
def rule : BoundRule := .sum [.predecessor 0 47581 .coefficient, .predecessor 1 47582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47581 .coefficient)
      LeftAuthority47579.bound (LeftAuthority47579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47579.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47582 .coefficient)
      LeftBound47575.bound (LeftBound47575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47575.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47579.bound, LeftBound47575.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47579.bound, LeftBound47575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47579.actual selector witness, LeftBound47575.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47583

namespace LeftBound47587
def owner : Owner := ⟨.program ⟨214⟩, ⟨28976⟩⟩
def transferEvent : Nat := 47587
def frameStart : Nat := 47487
def rule : BoundRule := .sum [.predecessor 0 47585 .coefficient, .predecessor 1 47586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47585 .coefficient)
      LeftBound47583.bound (LeftBound47583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47586 .coefficient)
      LeftBound47564.bound (LeftBound47564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47564.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47583.bound, LeftBound47564.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47583.bound, LeftBound47564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47583.actual selector witness, LeftBound47564.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47587

namespace LeftBound47600
def owner : Owner := ⟨.program ⟨214⟩, ⟨28973⟩⟩
def transferEvent : Nat := 47600
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 47598 .coefficient, .predecessor 1 47599 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47598 .coefficient)
      LeftBound47429.bound (LeftBound47429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47599 .coefficient)
      LeftBound47412.bound (LeftBound47412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47412.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47429.bound, LeftBound47412.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47429.bound, LeftBound47412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47429.actual selector witness, LeftBound47412.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47600

namespace LeftBound47603
def owner : Owner := ⟨.program ⟨214⟩, ⟨28973⟩⟩
def transferEvent : Nat := 47603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 47597 .summary, .result 47419 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47597 .summary)
      LeftBound47431.bound (LeftBound47431.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22059⟩⟩) (rawTerms := some (Proof.Events185.exact47597RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47419 .summary)
      LeftBound47414.bound (LeftBound47414.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28972⟩⟩) (rawTerms := some (Proof.Events185.exact47419RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47431.bound, LeftBound47414.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47431.bound, LeftBound47414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47431.actual selector witness, LeftBound47414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47603

namespace LeftBound47607
def owner : Owner := ⟨.program ⟨214⟩, ⟨28974⟩⟩
def transferEvent : Nat := 47607
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47605 .coefficient) (.predecessor 1 47606 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47605 .coefficient)
      LeftBound47600.bound (LeftBound47600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47606 .coefficient)
      LeftBound5618.bound (LeftBound5618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5618.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5618.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47600.bound LeftBound5618.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47600.bound, LeftBound5618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47600.actual selector witness) * (LeftBound5618.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47607

namespace LeftBound47608
def owner : Owner := ⟨.program ⟨214⟩, ⟨28974⟩⟩
def transferEvent : Nat := 47608
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47608

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
