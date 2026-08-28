import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard582
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard583

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85475
def owner : Owner := ⟨.program ⟨214⟩, ⟨25991⟩⟩
def transferEvent : Nat := 85475
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 85469 .summary, .result 85285 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85469 .summary)
      LeftBound85297.bound (LeftBound85297.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19459⟩⟩) (rawTerms := some (Proof.Events333.exact85469RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85285 .summary)
      LeftBound85280.bound (LeftBound85280.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25990⟩⟩) (rawTerms := some (Proof.Events333.exact85285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85297.bound, LeftBound85280.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85297.bound, LeftBound85280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85297.actual selector witness, LeftBound85280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85475

namespace LeftBound85479
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def transferEvent : Nat := 85479
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85477 .coefficient) (.predecessor 1 85478 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85477 .coefficient)
      LeftBound85472.bound (LeftBound85472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85478 .coefficient)
      LeftAuthority85200.bound (LeftAuthority85200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85472.bound LeftAuthority85200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85472.bound, LeftAuthority85200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85472.actual selector witness) * (LeftAuthority85200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85479

namespace LeftBound85480
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def transferEvent : Nat := 85480
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27649⟩⟩]⟩ [⟨.result 85201 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85201 .coefficient)
      LeftAuthority85200.bound (LeftAuthority85200.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27649⟩⟩) (rawTerms := some (Proof.Events332.exact85201RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85200.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85200.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85200.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85480

namespace LeftBound85481
def owner : Owner := ⟨.program ⟨214⟩, ⟨27651⟩⟩
def transferEvent : Nat := 85481
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 85476 .summary) (.transfer 85480) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85476 .summary)
      LeftBound85475.bound (LeftBound85475.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25991⟩⟩) (rawTerms := some (Proof.Events333.exact85476RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85480)
      LeftBound85480.bound (LeftBound85480.actual selector witness) := by
  exact .transfer (LeftBound85480.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85475.bound LeftBound85480.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85475.bound, LeftBound85480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85475.actual selector witness) * (LeftBound85480.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85481

namespace LeftBound85492
def owner : Owner := ⟨.program ⟨214⟩, ⟨21258⟩⟩
def transferEvent : Nat := 85492
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 85490 .coefficient) (.value (.predecessor 1 85491 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85490 .coefficient)
      LeftAuthority85488.bound (LeftAuthority85488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85491 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority85488.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85488.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85488.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85492

namespace LeftBound85496
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def transferEvent : Nat := 85496
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85494 .coefficient) (.predecessor 1 85495 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85494 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85495 .coefficient)
      LeftBound85492.bound (LeftBound85492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound85492.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound85492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound85492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85496

namespace LeftBound85497
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def transferEvent : Nat := 85497
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21256⟩⟩]⟩ [⟨.result 85489 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85489 .coefficient)
      LeftAuthority85488.bound (LeftAuthority85488.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21256⟩⟩) (rawTerms := some (Proof.Events333.exact85489RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85488.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85488.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85488.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85497

namespace LeftBound85498
def owner : Owner := ⟨.program ⟨214⟩, ⟨21259⟩⟩
def transferEvent : Nat := 85498
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 85497) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85497)
      LeftBound85497.bound (LeftBound85497.actual selector witness) := by
  exact .transfer (LeftBound85497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound85497.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound85497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound85497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85498

namespace LeftBound85593
def owner : Owner := ⟨.program ⟨214⟩, ⟨15822⟩⟩
def transferEvent : Nat := 85593
def frameStart : Nat := 85554
def rule : BoundRule := .identity (.predecessor 0 85592 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85592 .coefficient)
      LeftAuthority85590.bound (LeftAuthority85590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85590.derived selector witness)

def rawBound : CoeffClass := LeftAuthority85590.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority85590.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85593

namespace LeftBound85610
def owner : Owner := ⟨.program ⟨214⟩, ⟨15896⟩⟩
def transferEvent : Nat := 85610
def frameStart : Nat := 85554
def rule : BoundRule := .sum [.predecessor 0 85608 .coefficient, .predecessor 1 85609 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85608 .coefficient)
      LeftBound85593.bound (LeftBound85593.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85609 .coefficient)
      LeftAuthority85606.bound (LeftAuthority85606.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85593.bound, LeftAuthority85606.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85593.bound, LeftAuthority85606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85593.actual selector witness, LeftAuthority85606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85610

namespace LeftBound85613
def owner : Owner := ⟨.program ⟨214⟩, ⟨15897⟩⟩
def transferEvent : Nat := 85613
def frameStart : Nat := 85554
def rule : BoundRule := .identity (.predecessor 0 85612 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85612 .coefficient)
      LeftBound85610.bound (LeftBound85610.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85610.derived selector witness)

def rawBound : CoeffClass := LeftBound85610.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound85610.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85613

namespace LeftBound85619
def owner : Owner := ⟨.program ⟨214⟩, ⟨15898⟩⟩
def transferEvent : Nat := 85619
def frameStart : Nat := 85554
def rule : BoundRule := .product (.predecessor 0 85617 .coefficient) (.predecessor 1 85618 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85617 .coefficient)
      LeftAuthority85615.bound (LeftAuthority85615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85618 .coefficient)
      LeftBound85613.bound (LeftBound85613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85613.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority85615.bound LeftBound85613.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85615.bound, LeftBound85613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority85615.actual selector witness) * (LeftBound85613.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85619

namespace LeftBound85627
def owner : Owner := ⟨.program ⟨214⟩, ⟨15899⟩⟩
def transferEvent : Nat := 85627
def frameStart : Nat := 85554
def rule : BoundRule := .sum [.predecessor 0 85625 .coefficient, .predecessor 1 85626 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85625 .coefficient)
      LeftAuthority85623.bound (LeftAuthority85623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85623.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85626 .coefficient)
      LeftBound85619.bound (LeftBound85619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85619.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85623.bound, LeftBound85619.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85623.bound, LeftBound85619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85623.actual selector witness, LeftBound85619.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85627

namespace LeftBound85631
def owner : Owner := ⟨.program ⟨214⟩, ⟨27650⟩⟩
def transferEvent : Nat := 85631
def frameStart : Nat := 85554
def rule : BoundRule := .product (.predecessor 0 85629 .coefficient) (.predecessor 1 85630 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85629 .coefficient)
      LeftBound85627.bound (LeftBound85627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85630 .coefficient)
      LeftAuthority85604.bound (LeftAuthority85604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85604.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85604.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85627.bound LeftAuthority85604.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85627.bound, LeftAuthority85604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85627.actual selector witness) * (LeftAuthority85604.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85631

namespace LeftBound85642
def owner : Owner := ⟨.program ⟨214⟩, ⟨15868⟩⟩
def transferEvent : Nat := 85642
def frameStart : Nat := 85554
def rule : BoundRule := .product (.predecessor 0 85640 .coefficient) (.predecessor 1 85641 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85640 .coefficient)
      LeftAuthority85615.bound (LeftAuthority85615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85615.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85641 .coefficient)
      LeftAuthority85638.bound (LeftAuthority85638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85638.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85615.bound LeftAuthority85638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85615.bound, LeftAuthority85638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority85615.actual selector witness) * (LeftAuthority85638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85642

namespace LeftBound85650
def owner : Owner := ⟨.program ⟨214⟩, ⟨15869⟩⟩
def transferEvent : Nat := 85650
def frameStart : Nat := 85554
def rule : BoundRule := .sum [.predecessor 0 85648 .coefficient, .predecessor 1 85649 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85648 .coefficient)
      LeftAuthority85646.bound (LeftAuthority85646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85649 .coefficient)
      LeftBound85642.bound (LeftBound85642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85642.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority85646.bound, LeftBound85642.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85646.bound, LeftBound85642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority85646.actual selector witness, LeftBound85642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85650

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
