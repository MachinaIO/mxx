import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard223

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound34496
def owner : Owner := ⟨.program ⟨214⟩, ⟨20910⟩⟩
def transferEvent : Nat := 34496
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 34494 .coefficient) (.value (.predecessor 1 34495 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34494 .coefficient)
      LeftAuthority34492.bound (LeftAuthority34492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34495 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority34492.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34492.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34492.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound34496

namespace LeftBound34500
def owner : Owner := ⟨.program ⟨214⟩, ⟨20911⟩⟩
def transferEvent : Nat := 34500
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34498 .coefficient) (.predecessor 1 34499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34498 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34499 .coefficient)
      LeftBound34496.bound (LeftBound34496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34496.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound34496.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound34496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound34496.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34500

namespace LeftBound34501
def owner : Owner := ⟨.program ⟨214⟩, ⟨20911⟩⟩
def transferEvent : Nat := 34501
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20908⟩⟩]⟩ [⟨.result 34493 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34493 .coefficient)
      LeftAuthority34492.bound (LeftAuthority34492.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20908⟩⟩) (rawTerms := some (Proof.Events134.exact34493RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34492.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34492.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34492.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34501

namespace LeftBound34502
def owner : Owner := ⟨.program ⟨214⟩, ⟨20911⟩⟩
def transferEvent : Nat := 34502
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 34501) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34501)
      LeftBound34501.bound (LeftBound34501.actual selector witness) := by
  exact .transfer (LeftBound34501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound34501.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound34501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound34501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34502

namespace LeftBound34597
def owner : Owner := ⟨.program ⟨214⟩, ⟨15596⟩⟩
def transferEvent : Nat := 34597
def frameStart : Nat := 34558
def rule : BoundRule := .identity (.predecessor 0 34596 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34596 .coefficient)
      LeftAuthority34594.bound (LeftAuthority34594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34594.derived selector witness)

def rawBound : CoeffClass := LeftAuthority34594.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority34594.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34597

namespace LeftBound34614
def owner : Owner := ⟨.program ⟨214⟩, ⟨15670⟩⟩
def transferEvent : Nat := 34614
def frameStart : Nat := 34558
def rule : BoundRule := .sum [.predecessor 0 34612 .coefficient, .predecessor 1 34613 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34612 .coefficient)
      LeftBound34597.bound (LeftBound34597.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34613 .coefficient)
      LeftAuthority34610.bound (LeftAuthority34610.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority34610.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34597.bound, LeftAuthority34610.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34597.bound, LeftAuthority34610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34597.actual selector witness, LeftAuthority34610.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34614

namespace LeftBound34617
def owner : Owner := ⟨.program ⟨214⟩, ⟨15671⟩⟩
def transferEvent : Nat := 34617
def frameStart : Nat := 34558
def rule : BoundRule := .identity (.predecessor 0 34616 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34616 .coefficient)
      LeftBound34614.bound (LeftBound34614.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34614.derived selector witness)

def rawBound : CoeffClass := LeftBound34614.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound34614.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34617

namespace LeftBound34623
def owner : Owner := ⟨.program ⟨214⟩, ⟨15672⟩⟩
def transferEvent : Nat := 34623
def frameStart : Nat := 34558
def rule : BoundRule := .product (.predecessor 0 34621 .coefficient) (.predecessor 1 34622 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34621 .coefficient)
      LeftAuthority34619.bound (LeftAuthority34619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34619.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34622 .coefficient)
      LeftBound34617.bound (LeftBound34617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34617.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority34619.bound LeftBound34617.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34619.bound, LeftBound34617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority34619.actual selector witness) * (LeftBound34617.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34623

namespace LeftBound34631
def owner : Owner := ⟨.program ⟨214⟩, ⟨15673⟩⟩
def transferEvent : Nat := 34631
def frameStart : Nat := 34558
def rule : BoundRule := .sum [.predecessor 0 34629 .coefficient, .predecessor 1 34630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34629 .coefficient)
      LeftAuthority34627.bound (LeftAuthority34627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34630 .coefficient)
      LeftBound34623.bound (LeftBound34623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34627.bound, LeftBound34623.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34627.bound, LeftBound34623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34627.actual selector witness, LeftBound34623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34631

namespace LeftBound34635
def owner : Owner := ⟨.program ⟨214⟩, ⟨27248⟩⟩
def transferEvent : Nat := 34635
def frameStart : Nat := 34558
def rule : BoundRule := .product (.predecessor 0 34633 .coefficient) (.predecessor 1 34634 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34633 .coefficient)
      LeftBound34631.bound (LeftBound34631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34634 .coefficient)
      LeftAuthority34608.bound (LeftAuthority34608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34608.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34631.bound LeftAuthority34608.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34631.bound, LeftAuthority34608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34631.actual selector witness) * (LeftAuthority34608.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34635

namespace LeftBound34646
def owner : Owner := ⟨.program ⟨214⟩, ⟨17844⟩⟩
def transferEvent : Nat := 34646
def frameStart : Nat := 34558
def rule : BoundRule := .product (.predecessor 0 34644 .coefficient) (.predecessor 1 34645 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34644 .coefficient)
      LeftAuthority34619.bound (LeftAuthority34619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34619.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34645 .coefficient)
      LeftAuthority34642.bound (LeftAuthority34642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority34619.bound LeftAuthority34642.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34619.bound, LeftAuthority34642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority34619.actual selector witness) * (LeftAuthority34642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34646

namespace LeftBound34654
def owner : Owner := ⟨.program ⟨214⟩, ⟨17845⟩⟩
def transferEvent : Nat := 34654
def frameStart : Nat := 34558
def rule : BoundRule := .sum [.predecessor 0 34652 .coefficient, .predecessor 1 34653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34652 .coefficient)
      LeftAuthority34650.bound (LeftAuthority34650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34650.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34653 .coefficient)
      LeftBound34646.bound (LeftBound34646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34650.bound, LeftBound34646.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34650.bound, LeftBound34646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34650.actual selector witness, LeftBound34646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34654

namespace LeftBound34658
def owner : Owner := ⟨.program ⟨214⟩, ⟨27253⟩⟩
def transferEvent : Nat := 34658
def frameStart : Nat := 34558
def rule : BoundRule := .sum [.predecessor 0 34656 .coefficient, .predecessor 1 34657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34656 .coefficient)
      LeftBound34654.bound (LeftBound34654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34657 .coefficient)
      LeftBound34635.bound (LeftBound34635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34654.bound, LeftBound34635.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34654.bound, LeftBound34635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34654.actual selector witness, LeftBound34635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34658

namespace LeftBound34671
def owner : Owner := ⟨.program ⟨214⟩, ⟨27250⟩⟩
def transferEvent : Nat := 34671
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 34669 .coefficient, .predecessor 1 34670 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34669 .coefficient)
      LeftBound34500.bound (LeftBound34500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34670 .coefficient)
      LeftBound34483.bound (LeftBound34483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34500.bound, LeftBound34483.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34500.bound, LeftBound34483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34500.actual selector witness, LeftBound34483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34671

namespace LeftBound34674
def owner : Owner := ⟨.program ⟨214⟩, ⟨27250⟩⟩
def transferEvent : Nat := 34674
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 34668 .summary, .result 34490 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34668 .summary)
      LeftBound34502.bound (LeftBound34502.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20911⟩⟩) (rawTerms := some (Proof.Events135.exact34668RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34490 .summary)
      LeftBound34485.bound (LeftBound34485.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27249⟩⟩) (rawTerms := some (Proof.Events134.exact34490RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34485.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34502.bound, LeftBound34485.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34502.bound, LeftBound34485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34502.actual selector witness, LeftBound34485.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34674

namespace LeftBound34678
def owner : Owner := ⟨.program ⟨214⟩, ⟨27251⟩⟩
def transferEvent : Nat := 34678
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34676 .coefficient) (.predecessor 1 34677 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34676 .coefficient)
      LeftBound34671.bound (LeftBound34671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34677 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34671.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34671.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34671.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34678

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
