import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard883
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard954
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard956
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard957
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard958
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard980

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound148622
def owner : Owner := ⟨.program ⟨257⟩, ⟨69620⟩⟩
def transferEvent : Nat := 148622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148618 .summary, .result 145336 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148618 .summary)
      LeftBound148617.bound (LeftBound148617.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69619⟩⟩) (rawTerms := some (Proof.Events580.exact148618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145336 .summary)
      LeftBound145331.bound (LeftBound145331.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44492⟩⟩) (rawTerms := some (Proof.Events567.exact145336RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148617.bound, LeftBound145331.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148617.bound, LeftBound145331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148617.actual selector witness, LeftBound145331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148622

namespace LeftBound148626
def owner : Owner := ⟨.program ⟨257⟩, ⟨69621⟩⟩
def transferEvent : Nat := 148626
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148624 .coefficient, .predecessor 1 148625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148624 .coefficient)
      LeftBound148621.bound (LeftBound148621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148625 .coefficient)
      LeftBound145117.bound (LeftBound145117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events566.exact145124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145117.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148621.bound, LeftBound145117.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148621.bound, LeftBound145117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148621.actual selector witness, LeftBound145117.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148626

namespace LeftBound148627
def owner : Owner := ⟨.program ⟨257⟩, ⟨69621⟩⟩
def transferEvent : Nat := 148627
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148623 .summary, .result 145124 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148623 .summary)
      LeftBound148622.bound (LeftBound148622.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69620⟩⟩) (rawTerms := some (Proof.Events580.exact148623RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145124 .summary)
      LeftBound145119.bound (LeftBound145119.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47172⟩⟩) (rawTerms := some (Proof.Events566.exact145124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148622.bound, LeftBound145119.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148622.bound, LeftBound145119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148622.actual selector witness, LeftBound145119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148627

namespace LeftBound148631
def owner : Owner := ⟨.program ⟨257⟩, ⟨69622⟩⟩
def transferEvent : Nat := 148631
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148629 .coefficient, .predecessor 1 148630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148629 .coefficient)
      LeftBound148626.bound (LeftBound148626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148630 .coefficient)
      LeftBound144905.bound (LeftBound144905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events566.exact144912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148626.bound, LeftBound144905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148626.bound, LeftBound144905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148626.actual selector witness, LeftBound144905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148631

namespace LeftBound148632
def owner : Owner := ⟨.program ⟨257⟩, ⟨69622⟩⟩
def transferEvent : Nat := 148632
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148628 .summary, .result 144912 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148628 .summary)
      LeftBound148627.bound (LeftBound148627.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69621⟩⟩) (rawTerms := some (Proof.Events580.exact148628RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 144912 .summary)
      LeftBound144907.bound (LeftBound144907.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49852⟩⟩) (rawTerms := some (Proof.Events566.exact144912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound144907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148627.bound, LeftBound144907.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148627.bound, LeftBound144907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148627.actual selector witness, LeftBound144907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148632

namespace LeftBound148636
def owner : Owner := ⟨.program ⟨257⟩, ⟨71023⟩⟩
def transferEvent : Nat := 148636
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148634 .coefficient, .predecessor 1 148635 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148634 .coefficient)
      LeftBound148631.bound (LeftBound148631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148635 .coefficient)
      LeftBound144693.bound (LeftBound144693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events565.exact144700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148631.bound, LeftBound144693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148631.bound, LeftBound144693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148631.actual selector witness, LeftBound144693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148636

namespace LeftBound148637
def owner : Owner := ⟨.program ⟨257⟩, ⟨71023⟩⟩
def transferEvent : Nat := 148637
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148633 .summary, .result 144700 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148633 .summary)
      LeftBound148632.bound (LeftBound148632.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69622⟩⟩) (rawTerms := some (Proof.Events580.exact148633RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 144700 .summary)
      LeftBound144695.bound (LeftBound144695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71021⟩⟩) (rawTerms := some (Proof.Events565.exact144700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound144695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148632.bound, LeftBound144695.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148632.bound, LeftBound144695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148632.actual selector witness, LeftBound144695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148637

namespace LeftBound148643
def owner : Owner := ⟨.program ⟨257⟩, ⟨7409⟩⟩
def transferEvent : Nat := 148643
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 148641 .coefficient) (.predecessor 1 148642 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148641 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148642 .coefficient)
      LeftAuthority16306.bound (LeftAuthority16306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events063.exact16307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148643

namespace LeftBound148648
def owner : Owner := ⟨.program ⟨257⟩, ⟨9221⟩⟩
def transferEvent : Nat := 148648
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148646 .coefficient, .predecessor 1 148647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148646 .coefficient)
      LeftBound148643.bound (LeftBound148643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148647 .coefficient)
      LeftBound134401.bound (LeftBound134401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148643.bound, LeftBound134401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148643.bound, LeftBound134401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148643.actual selector witness, LeftBound134401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148648

namespace LeftBound148652
def owner : Owner := ⟨.program ⟨257⟩, ⟨9222⟩⟩
def transferEvent : Nat := 148652
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148650 .coefficient, .predecessor 1 148651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148650 .coefficient)
      LeftBound148648.bound (LeftBound148648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148651 .coefficient)
      LeftAuthority148639.bound (LeftAuthority148639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148648.bound, LeftAuthority148639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148648.bound, LeftAuthority148639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148648.actual selector witness, LeftAuthority148639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148652

namespace LeftBound148653
def owner : Owner := ⟨.program ⟨257⟩, ⟨9222⟩⟩
def transferEvent : Nat := 148653
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨15⟩⟩]⟩ [⟨.result 148640 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148640 .coefficient)
      LeftAuthority148639.bound (LeftAuthority148639.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨15⟩⟩) (rawTerms := some (Proof.Events580.exact148640RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority148639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority148639.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority148639.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority148639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority148639.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound148653

namespace LeftBound148658
def owner : Owner := ⟨.program ⟨257⟩, ⟨9621⟩⟩
def transferEvent : Nat := 148658
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 148656 .coefficient) (.predecessor 1 148657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148656 .coefficient)
      LeftBound148652.bound (LeftBound148652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148657 .coefficient)
      LeftBound15983.bound (LeftBound15983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound148652.bound LeftBound15983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148652.bound, LeftBound15983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound148652.actual selector witness) * (LeftBound15983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148658

namespace LeftBound148659
def owner : Owner := ⟨.program ⟨257⟩, ⟨9621⟩⟩
def transferEvent : Nat := 148659
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩ [⟨.result 15980 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15980 .coefficient)
      LeftAuthority15979.bound (LeftAuthority15979.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9583⟩⟩) (rawTerms := some (Proof.Events062.exact15980RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15979.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15979.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15979.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound148659

namespace LeftBound148660
def owner : Owner := ⟨.program ⟨257⟩, ⟨9621⟩⟩
def transferEvent : Nat := 148660
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 148655 .summary) (.transfer 148659) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148655 .summary)
      LeftBound148653.bound (LeftBound148653.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9222⟩⟩) (rawTerms := some (Proof.Events580.exact148655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 148659)
      LeftBound148659.bound (LeftBound148659.actual selector witness) := by
  exact .transfer (LeftBound148659.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound148653.bound LeftBound148659.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148653.bound, LeftBound148659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound148653.actual selector witness) * (LeftBound148659.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound148660

namespace LeftBound148686
def owner : Owner := ⟨.program ⟨257⟩, ⟨71024⟩⟩
def transferEvent : Nat := 148686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148684 .coefficient, .predecessor 1 148685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148684 .coefficient)
      LeftBound148658.bound (LeftBound148658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148685 .coefficient)
      LeftBound148636.bound (LeftBound148636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148636.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148658.bound, LeftBound148636.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148658.bound, LeftBound148636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148658.actual selector witness, LeftBound148636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148686

namespace LeftBound148706
def owner : Owner := ⟨.program ⟨257⟩, ⟨71024⟩⟩
def transferEvent : Nat := 148706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148683 .summary, .result 148638 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148683 .summary)
      LeftBound148660.bound (LeftBound148660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9621⟩⟩) (rawTerms := some (Proof.Events580.exact148683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148638 .summary)
      LeftBound148637.bound (LeftBound148637.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71023⟩⟩) (rawTerms := some (Proof.Events580.exact148638RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148660.bound, LeftBound148637.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148660.bound, LeftBound148637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148660.actual selector witness, LeftBound148637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148706

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
