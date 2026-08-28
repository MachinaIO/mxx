import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard582
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard647
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard648
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard649

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100631
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 100631
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100629 .coefficient, .predecessor 1 100630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100629 .coefficient)
      LeftBound100627.bound (LeftBound100627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100630 .coefficient)
      LeftAuthority100512.bound (LeftAuthority100512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100627.bound, LeftAuthority100512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100627.bound, LeftAuthority100512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100627.actual selector witness, LeftAuthority100512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100631

namespace LeftBound100635
def owner : Owner := ⟨.program ⟨257⟩, ⟨69110⟩⟩
def transferEvent : Nat := 100635
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100633 .coefficient, .predecessor 1 100634 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100633 .coefficient)
      LeftBound100631.bound (LeftBound100631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100634 .coefficient)
      LeftBound100491.bound (LeftBound100491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100491.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100631.bound, LeftBound100491.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100631.bound, LeftBound100491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100631.actual selector witness, LeftBound100491.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100635

namespace LeftBound100639
def owner : Owner := ⟨.program ⟨257⟩, ⟨71406⟩⟩
def transferEvent : Nat := 100639
def frameStart : Nat := 99961
def rule : BoundRule := .product (.predecessor 0 100637 .coefficient) (.predecessor 1 100638 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100637 .coefficient)
      LeftBound100635.bound (LeftBound100635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100638 .coefficient)
      LeftAuthority100476.bound (LeftAuthority100476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100476.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100476.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound100635.bound LeftAuthority100476.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100635.bound, LeftAuthority100476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound100635.actual selector witness) * (LeftAuthority100476.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100639

namespace LeftBound100718
def owner : Owner := ⟨.program ⟨257⟩, ⟨67568⟩⟩
def transferEvent : Nat := 100718
def frameStart : Nat := 99961
def rule : BoundRule := .product (.predecessor 0 100716 .coefficient) (.predecessor 1 100717 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100716 .coefficient)
      LeftAuthority100487.bound (LeftAuthority100487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100717 .coefficient)
      LeftAuthority100714.bound (LeftAuthority100714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100714.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100487.bound LeftAuthority100714.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100487.bound, LeftAuthority100714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority100487.actual selector witness) * (LeftAuthority100714.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100718

namespace LeftBound100726
def owner : Owner := ⟨.program ⟨257⟩, ⟨67573⟩⟩
def transferEvent : Nat := 100726
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100724 .coefficient, .predecessor 1 100725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100724 .coefficient)
      LeftAuthority100722.bound (LeftAuthority100722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100725 .coefficient)
      LeftBound100718.bound (LeftBound100718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100722.bound, LeftBound100718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100722.bound, LeftBound100718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority100722.actual selector witness, LeftBound100718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100726

namespace LeftBound100730
def owner : Owner := ⟨.program ⟨257⟩, ⟨71410⟩⟩
def transferEvent : Nat := 100730
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100728 .coefficient, .predecessor 1 100729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100728 .coefficient)
      LeftBound100726.bound (LeftBound100726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100729 .coefficient)
      LeftBound100639.bound (LeftBound100639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100726.bound, LeftBound100639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100726.bound, LeftBound100639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100726.actual selector witness, LeftBound100639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100730

namespace LeftBound100777
def owner : Owner := ⟨.program ⟨257⟩, ⟨71408⟩⟩
def transferEvent : Nat := 100777
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100775 .coefficient, .predecessor 1 100776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100775 .coefficient)
      LeftBound99368.bound (LeftBound99368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100776 .coefficient)
      LeftBound99283.bound (LeftBound99283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99368.bound, LeftBound99283.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99368.bound, LeftBound99283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99368.actual selector witness, LeftBound99283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100777

namespace LeftBound100814
def owner : Owner := ⟨.program ⟨257⟩, ⟨71408⟩⟩
def transferEvent : Nat := 100814
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100774 .summary, .result 99358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 100774 .summary)
      LeftBound99370.bound (LeftBound99370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68423⟩⟩) (rawTerms := some (Proof.Events393.exact100774RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 99358 .summary)
      LeftBound99285.bound (LeftBound99285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71407⟩⟩) (rawTerms := some (Proof.Events388.exact99358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99370.bound, LeftBound99285.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469506489977540968448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99370.bound, LeftBound99285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound99370.actual selector witness, LeftBound99285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100814

namespace LeftBound100818
def owner : Owner := ⟨.program ⟨257⟩, ⟨71409⟩⟩
def transferEvent : Nat := 100818
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100816 .coefficient) (.predecessor 1 100817 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100816 .coefficient)
      LeftBound100777.bound (LeftBound100777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100817 .coefficient)
      LeftBound15521.bound (LeftBound15521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound100777.bound LeftBound15521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100777.bound, LeftBound15521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound100777.actual selector witness) * (LeftBound15521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100818

namespace LeftBound100819
def owner : Owner := ⟨.program ⟨257⟩, ⟨71409⟩⟩
def transferEvent : Nat := 100819
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7139⟩⟩]⟩ [⟨.result 15518 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15518 .coefficient)
      LeftAuthority15517.bound (LeftAuthority15517.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7139⟩⟩) (rawTerms := some (Proof.Events060.exact15518RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15517.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15517.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15517.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100819

namespace LeftBound100820
def owner : Owner := ⟨.program ⟨257⟩, ⟨71409⟩⟩
def transferEvent : Nat := 100820
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100815 .summary) (.transfer 100819) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 100815 .summary)
      LeftBound100814.bound (LeftBound100814.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71408⟩⟩) (rawTerms := some (Proof.Events393.exact100815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 100819)
      LeftBound100819.bound (LeftBound100819.actual selector witness) := by
  exact .transfer (LeftBound100819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound100814.bound LeftBound100819.bound
def bound : CoeffClass := .finite ⟨66805187221379434678483228029309283225584960819691520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100814.bound, LeftBound100819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound100814.actual selector witness) * (LeftBound100819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100820

namespace LeftBound100835
def owner : Owner := ⟨.program ⟨257⟩, ⟨50150⟩⟩
def transferEvent : Nat := 100835
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100833 .coefficient) (.predecessor 1 100834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100833 .coefficient)
      LeftBound90802.bound (LeftBound90802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events354.exact90806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100834 .coefficient)
      LeftAuthority100831.bound (LeftAuthority100831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100831.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90802.bound LeftAuthority100831.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90802.bound, LeftAuthority100831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90802.actual selector witness) * (LeftAuthority100831.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100835

namespace LeftBound100836
def owner : Owner := ⟨.program ⟨257⟩, ⟨50150⟩⟩
def transferEvent : Nat := 100836
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨50148⟩⟩]⟩ [⟨.result 100832 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 100832 .coefficient)
      LeftAuthority100831.bound (LeftAuthority100831.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50148⟩⟩) (rawTerms := some (Proof.Events393.exact100832RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100831.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100831.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority100831.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100836

namespace LeftBound100837
def owner : Owner := ⟨.program ⟨257⟩, ⟨50150⟩⟩
def transferEvent : Nat := 100837
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 90806 .summary) (.transfer 100836) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 90806 .summary)
      LeftBound90805.bound (LeftBound90805.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49716⟩⟩) (rawTerms := some (Proof.Events354.exact90806RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 100836)
      LeftBound100836.bound (LeftBound100836.actual selector witness) := by
  exact .transfer (LeftBound100836.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound90805.bound LeftBound100836.bound
def bound : CoeffClass := .finite ⟨32194504275408438756654574469120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90805.bound, LeftBound100836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound90805.actual selector witness) * (LeftBound100836.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100837

namespace LeftBound100848
def owner : Owner := ⟨.program ⟨257⟩, ⟨48994⟩⟩
def transferEvent : Nat := 100848
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 100846 .coefficient) (.value (.predecessor 1 100847 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100846 .coefficient)
      LeftAuthority100844.bound (LeftAuthority100844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100847 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100844.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100844.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority100844.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100848

namespace LeftBound100852
def owner : Owner := ⟨.program ⟨257⟩, ⟨48995⟩⟩
def transferEvent : Nat := 100852
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100850 .coefficient) (.predecessor 1 100851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100850 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100851 .coefficient)
      LeftBound100848.bound (LeftBound100848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100848.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound100848.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound100848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound100848.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100852

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
