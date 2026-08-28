import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10512
def owner : Owner := ⟨.program ⟨214⟩, ⟨76⟩⟩
def transferEvent : Nat := 10512
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 10511 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10511 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10512

namespace LeftBound10516
def owner : Owner := ⟨.program ⟨214⟩, ⟨14681⟩⟩
def transferEvent : Nat := 10516
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 10514 .coefficient) (.predecessor 1 10515 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10514 .coefficient)
      LeftAuthority237.bound (LeftAuthority237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10515 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority237.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority237.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10516

namespace LeftBound10520
def owner : Owner := ⟨.program ⟨214⟩, ⟨6762⟩⟩
def transferEvent : Nat := 10520
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 10519 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10519 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound10520

namespace LeftBound10524
def owner : Owner := ⟨.program ⟨214⟩, ⟨7370⟩⟩
def transferEvent : Nat := 10524
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10522 .coefficient) (.predecessor 1 10523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10522 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10523 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10524

namespace LeftBound10529
def owner : Owner := ⟨.program ⟨214⟩, ⟨14682⟩⟩
def transferEvent : Nat := 10529
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10527 .coefficient, .predecessor 1 10528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10527 .coefficient)
      LeftBound10524.bound (LeftBound10524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10528 .coefficient)
      LeftBound10516.bound (LeftBound10516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10524.bound, LeftBound10516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10524.bound, LeftBound10516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10524.actual selector witness, LeftBound10516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10529

namespace LeftBound10533
def owner : Owner := ⟨.program ⟨214⟩, ⟨14683⟩⟩
def transferEvent : Nat := 10533
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10531 .coefficient, .predecessor 1 10532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10531 .coefficient)
      LeftBound10529.bound (LeftBound10529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10532 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10529.bound, LeftBound10512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10529.bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10529.actual selector witness, LeftBound10512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10533

namespace LeftBound10534
def owner : Owner := ⟨.program ⟨214⟩, ⟨14683⟩⟩
def transferEvent : Nat := 10534
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩ [⟨.result 10513 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10513 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨76⟩⟩) (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10512.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10512.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10534

namespace LeftBound10539
def owner : Owner := ⟨.program ⟨214⟩, ⟨14684⟩⟩
def transferEvent : Nat := 10539
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10537 .coefficient) (.predecessor 1 10538 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10537 .coefficient)
      LeftBound10533.bound (LeftBound10533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10538 .coefficient)
      LeftBound10509.bound (LeftBound10509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10533.bound LeftBound10509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10533.bound, LeftBound10509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10533.actual selector witness) * (LeftBound10509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10539

namespace LeftBound10540
def owner : Owner := ⟨.program ⟨214⟩, ⟨14684⟩⟩
def transferEvent : Nat := 10540
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩ [⟨.result 10506 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10506 .coefficient)
      LeftAuthority10505.bound (LeftAuthority10505.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7858⟩⟩) (rawTerms := some (Proof.Events041.exact10506RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10505.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10505.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10505.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10540

namespace LeftBound10541
def owner : Owner := ⟨.program ⟨214⟩, ⟨14684⟩⟩
def transferEvent : Nat := 10541
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10536 .summary) (.transfer 10540) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10536 .summary)
      LeftBound10534.bound (LeftBound10534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14683⟩⟩) (rawTerms := some (Proof.Events041.exact10536RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10540)
      LeftBound10540.bound (LeftBound10540.actual selector witness) := by
  exact .transfer (LeftBound10540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10534.bound LeftBound10540.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10534.bound, LeftBound10540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10534.actual selector witness) * (LeftBound10540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10541

namespace LeftBound10549
def owner : Owner := ⟨.program ⟨214⟩, ⟨14685⟩⟩
def transferEvent : Nat := 10549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10547 .coefficient, .predecessor 1 10548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10547 .coefficient)
      LeftBound10539.bound (LeftBound10539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10548 .coefficient)
      LeftBound10498.bound (LeftBound10498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10539.bound, LeftBound10498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10539.bound, LeftBound10498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10539.actual selector witness, LeftBound10498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10549

namespace LeftBound10551
def owner : Owner := ⟨.program ⟨214⟩, ⟨14685⟩⟩
def transferEvent : Nat := 10551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 10546 .summary, .result 10503 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10546 .summary)
      LeftBound10541.bound (LeftBound10541.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14684⟩⟩) (rawTerms := some (Proof.Events041.exact10546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10503 .summary)
      LeftBound10500.bound (LeftBound10500.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14680⟩⟩) (rawTerms := some (Proof.Events041.exact10503RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10541.bound, LeftBound10500.bound]
def bound : CoeffClass := .finite ⟨95443712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10541.bound, LeftBound10500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10541.actual selector witness, LeftBound10500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10551

namespace LeftBound10555
def owner : Owner := ⟨.program ⟨214⟩, ⟨26241⟩⟩
def transferEvent : Nat := 10555
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10553 .coefficient) (.predecessor 1 10554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10553 .coefficient)
      LeftBound10549.bound (LeftBound10549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10554 .coefficient)
      LeftAuthority10468.bound (LeftAuthority10468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10549.bound LeftAuthority10468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10549.bound, LeftAuthority10468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10549.actual selector witness) * (LeftAuthority10468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10555

namespace LeftBound10556
def owner : Owner := ⟨.program ⟨214⟩, ⟨26241⟩⟩
def transferEvent : Nat := 10556
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26240⟩⟩]⟩ [⟨.result 10469 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10469 .coefficient)
      LeftAuthority10468.bound (LeftAuthority10468.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26240⟩⟩) (rawTerms := some (Proof.Events040.exact10469RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10468.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10468.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10468.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10556

namespace LeftBound10557
def owner : Owner := ⟨.program ⟨214⟩, ⟨26241⟩⟩
def transferEvent : Nat := 10557
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10552 .summary) (.transfer 10556) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10552 .summary)
      LeftBound10551.bound (LeftBound10551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14685⟩⟩) (rawTerms := some (Proof.Events041.exact10552RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10556)
      LeftBound10556.bound (LeftBound10556.actual selector witness) := by
  exact .transfer (LeftBound10556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10551.bound LeftBound10556.bound
def bound : CoeffClass := .finite ⟨350279950139392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10551.bound, LeftBound10556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10551.actual selector witness) * (LeftBound10556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10557

namespace LeftBound10568
def owner : Owner := ⟨.program ⟨214⟩, ⟨19690⟩⟩
def transferEvent : Nat := 10568
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 10566 .coefficient) (.value (.predecessor 1 10567 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10566 .coefficient)
      LeftAuthority10564.bound (LeftAuthority10564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10567 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10564.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10564.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10564.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10568

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
