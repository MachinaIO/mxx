import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard085
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard147

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28483
def owner : Owner := ⟨.program ⟨257⟩, ⟨41432⟩⟩
def transferEvent : Nat := 28483
def frameStart : Nat := 28418
def rule : BoundRule := .product (.predecessor 0 28481 .coefficient) (.predecessor 1 28482 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28481 .coefficient)
      LeftAuthority28479.bound (LeftAuthority28479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28482 .coefficient)
      LeftBound28477.bound (LeftBound28477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority28479.bound LeftBound28477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28479.bound, LeftBound28477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority28479.actual selector witness) * (LeftBound28477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28483

namespace LeftBound28491
def owner : Owner := ⟨.program ⟨257⟩, ⟨41433⟩⟩
def transferEvent : Nat := 28491
def frameStart : Nat := 28418
def rule : BoundRule := .sum [.predecessor 0 28489 .coefficient, .predecessor 1 28490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28489 .coefficient)
      LeftAuthority28487.bound (LeftAuthority28487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28490 .coefficient)
      LeftBound28483.bound (LeftBound28483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28487.bound, LeftBound28483.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28487.bound, LeftBound28483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority28487.actual selector witness, LeftBound28483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28491

namespace LeftBound28495
def owner : Owner := ⟨.program ⟨257⟩, ⟨41766⟩⟩
def transferEvent : Nat := 28495
def frameStart : Nat := 28418
def rule : BoundRule := .product (.predecessor 0 28493 .coefficient) (.predecessor 1 28494 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28493 .coefficient)
      LeftBound28491.bound (LeftBound28491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28494 .coefficient)
      LeftAuthority28468.bound (LeftAuthority28468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound28491.bound LeftAuthority28468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28491.bound, LeftAuthority28468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound28491.actual selector witness) * (LeftAuthority28468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28495

namespace LeftBound28506
def owner : Owner := ⟨.program ⟨257⟩, ⟨40210⟩⟩
def transferEvent : Nat := 28506
def frameStart : Nat := 28418
def rule : BoundRule := .product (.predecessor 0 28504 .coefficient) (.predecessor 1 28505 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28504 .coefficient)
      LeftAuthority28479.bound (LeftAuthority28479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28505 .coefficient)
      LeftAuthority28502.bound (LeftAuthority28502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28502.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28479.bound LeftAuthority28502.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28479.bound, LeftAuthority28502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority28479.actual selector witness) * (LeftAuthority28502.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28506

namespace LeftBound28514
def owner : Owner := ⟨.program ⟨257⟩, ⟨40211⟩⟩
def transferEvent : Nat := 28514
def frameStart : Nat := 28418
def rule : BoundRule := .sum [.predecessor 0 28512 .coefficient, .predecessor 1 28513 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28512 .coefficient)
      LeftAuthority28510.bound (LeftAuthority28510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28510.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28513 .coefficient)
      LeftBound28506.bound (LeftBound28506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28510.bound, LeftBound28506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28510.bound, LeftBound28506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority28510.actual selector witness, LeftBound28506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28514

namespace LeftBound28518
def owner : Owner := ⟨.program ⟨257⟩, ⟨41770⟩⟩
def transferEvent : Nat := 28518
def frameStart : Nat := 28418
def rule : BoundRule := .sum [.predecessor 0 28516 .coefficient, .predecessor 1 28517 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28516 .coefficient)
      LeftBound28514.bound (LeftBound28514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28517 .coefficient)
      LeftBound28495.bound (LeftBound28495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28514.bound, LeftBound28495.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28514.bound, LeftBound28495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound28514.actual selector witness, LeftBound28495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28518

namespace LeftBound28531
def owner : Owner := ⟨.program ⟨257⟩, ⟨41768⟩⟩
def transferEvent : Nat := 28531
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28529 .coefficient, .predecessor 1 28530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28529 .coefficient)
      LeftBound28360.bound (LeftBound28360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28530 .coefficient)
      LeftBound28343.bound (LeftBound28343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28343.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28360.bound, LeftBound28343.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28360.bound, LeftBound28343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound28360.actual selector witness, LeftBound28343.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28531

namespace LeftBound28534
def owner : Owner := ⟨.program ⟨257⟩, ⟨41768⟩⟩
def transferEvent : Nat := 28534
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28528 .summary, .result 28350 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28528 .summary)
      LeftBound28362.bound (LeftBound28362.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40681⟩⟩) (rawTerms := some (Proof.Events111.exact28528RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28350 .summary)
      LeftBound28345.bound (LeftBound28345.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41767⟩⟩) (rawTerms := some (Proof.Events110.exact28350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28362.bound, LeftBound28345.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28362.bound, LeftBound28345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound28362.actual selector witness, LeftBound28345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28534

namespace LeftBound28538
def owner : Owner := ⟨.program ⟨257⟩, ⟨41769⟩⟩
def transferEvent : Nat := 28538
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28536 .coefficient) (.predecessor 1 28537 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28536 .coefficient)
      LeftBound28531.bound (LeftBound28531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28537 .coefficient)
      LeftBound15601.bound (LeftBound15601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound28531.bound LeftBound15601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28531.bound, LeftBound15601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound28531.actual selector witness) * (LeftBound15601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28538

namespace LeftBound28539
def owner : Owner := ⟨.program ⟨257⟩, ⟨41769⟩⟩
def transferEvent : Nat := 28539
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩ [⟨.result 15598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15598 .coefficient)
      LeftAuthority15597.bound (LeftAuthority15597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7159⟩⟩) (rawTerms := some (Proof.Events060.exact15598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28539

namespace LeftBound28540
def owner : Owner := ⟨.program ⟨257⟩, ⟨41769⟩⟩
def transferEvent : Nat := 28540
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28535 .summary) (.transfer 28539) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28535 .summary)
      LeftBound28534.bound (LeftBound28534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41768⟩⟩) (rawTerms := some (Proof.Events111.exact28535RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 28539)
      LeftBound28539.bound (LeftBound28539.actual selector witness) := by
  exact .transfer (LeftBound28539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound28534.bound LeftBound28539.bound
def bound : CoeffClass := .finite ⟨345671091840339265080175045977281837137920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28534.bound, LeftBound28539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound28534.actual selector witness) * (LeftBound28539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28540

namespace LeftBound28555
def owner : Owner := ⟨.program ⟨257⟩, ⟨39087⟩⟩
def transferEvent : Nat := 28555
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28553 .coefficient) (.predecessor 1 28554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28553 .coefficient)
      LeftBound19355.bound (LeftBound19355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28554 .coefficient)
      LeftAuthority28551.bound (LeftAuthority28551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28551.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound19355.bound LeftAuthority28551.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19355.bound, LeftAuthority28551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound19355.actual selector witness) * (LeftAuthority28551.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28555

namespace LeftBound28556
def owner : Owner := ⟨.program ⟨257⟩, ⟨39087⟩⟩
def transferEvent : Nat := 28556
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39085⟩⟩]⟩ [⟨.result 28552 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28552 .coefficient)
      LeftAuthority28551.bound (LeftAuthority28551.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39085⟩⟩) (rawTerms := some (Proof.Events111.exact28552RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28551.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28551.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority28551.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28556

namespace LeftBound28557
def owner : Owner := ⟨.program ⟨257⟩, ⟨39087⟩⟩
def transferEvent : Nat := 28557
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 19359 .summary) (.transfer 28556) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19359 .summary)
      LeftBound19358.bound (LeftBound19358.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38845⟩⟩) (rawTerms := some (Proof.Events075.exact19359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 28556)
      LeftBound28556.bound (LeftBound28556.actual selector witness) := by
  exact .transfer (LeftBound28556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound19358.bound LeftBound28556.bound
def bound : CoeffClass := .finite ⟨32192736221397252361486566686720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19358.bound, LeftBound28556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound19358.actual selector witness) * (LeftBound28556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28557

namespace LeftBound28568
def owner : Owner := ⟨.program ⟨257⟩, ⟨38000⟩⟩
def transferEvent : Nat := 28568
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 28566 .coefficient) (.value (.predecessor 1 28567 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28566 .coefficient)
      LeftAuthority28564.bound (LeftAuthority28564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28567 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28564.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28564.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority28564.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28568

namespace LeftBound28572
def owner : Owner := ⟨.program ⟨257⟩, ⟨38001⟩⟩
def transferEvent : Nat := 28572
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28570 .coefficient) (.predecessor 1 28571 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 28570 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 28571 .coefficient)
      LeftBound28568.bound (LeftBound28568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28568.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound28568.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound28568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound28568.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28572

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
