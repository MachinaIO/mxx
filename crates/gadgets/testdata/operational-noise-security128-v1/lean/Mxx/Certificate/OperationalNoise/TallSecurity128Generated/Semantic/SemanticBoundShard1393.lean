import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound207708
def owner : Owner := ⟨.program ⟨257⟩, ⟨47836⟩⟩
def transferEvent : Nat := 207708
def frameStart : Nat := 207675
def rule : BoundRule := .identity (.predecessor 0 207707 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207707 .coefficient)
      LeftBound207704.bound (LeftBound207704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207704.derived selector witness)

def rawBound : CoeffClass := LeftBound207704.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound207704.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound207708

namespace LeftBound207725
def owner : Owner := ⟨.program ⟨257⟩, ⟨49426⟩⟩
def transferEvent : Nat := 207725
def frameStart : Nat := 207675
def rule : BoundRule := .sum [.predecessor 0 207723 .coefficient, .predecessor 1 207724 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207723 .coefficient)
      LeftBound207708.bound (LeftBound207708.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound207708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207724 .coefficient)
      LeftAuthority207721.bound (LeftAuthority207721.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority207721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207708.bound, LeftAuthority207721.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207708.bound, LeftAuthority207721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207708.actual selector witness, LeftAuthority207721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207725

namespace LeftBound207728
def owner : Owner := ⟨.program ⟨257⟩, ⟨49427⟩⟩
def transferEvent : Nat := 207728
def frameStart : Nat := 207675
def rule : BoundRule := .identity (.predecessor 0 207727 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207727 .coefficient)
      LeftBound207725.bound (LeftBound207725.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound207725.derived selector witness)

def rawBound : CoeffClass := LeftBound207725.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207725.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound207725.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound207728

namespace LeftBound207734
def owner : Owner := ⟨.program ⟨257⟩, ⟨49428⟩⟩
def transferEvent : Nat := 207734
def frameStart : Nat := 207675
def rule : BoundRule := .product (.predecessor 0 207732 .coefficient) (.predecessor 1 207733 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207732 .coefficient)
      LeftAuthority207730.bound (LeftAuthority207730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207733 .coefficient)
      LeftBound207728.bound (LeftBound207728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority207730.bound LeftBound207728.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207730.bound, LeftBound207728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority207730.actual selector witness) * (LeftBound207728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207734

namespace LeftBound207750
def owner : Owner := ⟨.program ⟨257⟩, ⟨9566⟩⟩
def transferEvent : Nat := 207750
def frameStart : Nat := 207675
def rule : BoundRule := .scale (.predecessor 0 207748 .coefficient) (.value (.predecessor 1 207749 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207748 .coefficient)
      LeftAuthority207746.bound (LeftAuthority207746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207749 .coefficient)
      LeftAuthority207737.bound (LeftAuthority207737.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority207737.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority207746.bound LeftAuthority207737.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207746.bound, LeftAuthority207737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority207746.actual selector witness) * (LeftAuthority207737.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound207750

namespace LeftBound207753
def owner : Owner := ⟨.program ⟨257⟩, ⟨7302⟩⟩
def transferEvent : Nat := 207753
def frameStart : Nat := 207675
def rule : BoundRule := .identity (.predecessor 0 207752 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207752 .coefficient)
      LeftAuthority207740.bound (LeftAuthority207740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207740.derived selector witness)

def rawBound : CoeffClass := LeftAuthority207740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority207740.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound207753

namespace LeftBound207757
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def transferEvent : Nat := 207757
def frameStart : Nat := 207675
def rule : BoundRule := .product (.predecessor 0 207755 .coefficient) (.predecessor 1 207756 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207755 .coefficient)
      LeftBound207753.bound (LeftBound207753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207756 .coefficient)
      LeftBound207750.bound (LeftBound207750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207750.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound207753.bound LeftBound207750.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207753.bound, LeftBound207750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound207753.actual selector witness) * (LeftBound207750.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207757

namespace LeftBound207762
def owner : Owner := ⟨.program ⟨257⟩, ⟨49429⟩⟩
def transferEvent : Nat := 207762
def frameStart : Nat := 207675
def rule : BoundRule := .sum [.predecessor 0 207760 .coefficient, .predecessor 1 207761 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207760 .coefficient)
      LeftBound207757.bound (LeftBound207757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207761 .coefficient)
      LeftBound207734.bound (LeftBound207734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207734.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207757.bound, LeftBound207734.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207757.bound, LeftBound207734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207757.actual selector witness, LeftBound207734.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207762

namespace LeftBound207766
def owner : Owner := ⟨.program ⟨257⟩, ⟨49662⟩⟩
def transferEvent : Nat := 207766
def frameStart : Nat := 207675
def rule : BoundRule := .product (.predecessor 0 207764 .coefficient) (.predecessor 1 207765 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207764 .coefficient)
      LeftBound207762.bound (LeftBound207762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207765 .coefficient)
      LeftAuthority207719.bound (LeftAuthority207719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207719.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound207762.bound LeftAuthority207719.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207762.bound, LeftAuthority207719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound207762.actual selector witness) * (LeftAuthority207719.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207766

namespace LeftBound207777
def owner : Owner := ⟨.program ⟨257⟩, ⟨48150⟩⟩
def transferEvent : Nat := 207777
def frameStart : Nat := 207675
def rule : BoundRule := .product (.predecessor 0 207775 .coefficient) (.predecessor 1 207776 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207775 .coefficient)
      LeftAuthority207730.bound (LeftAuthority207730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207776 .coefficient)
      LeftAuthority207773.bound (LeftAuthority207773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority207730.bound LeftAuthority207773.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207730.bound, LeftAuthority207773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority207730.actual selector witness) * (LeftAuthority207773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207777

namespace LeftBound207785
def owner : Owner := ⟨.program ⟨257⟩, ⟨48151⟩⟩
def transferEvent : Nat := 207785
def frameStart : Nat := 207675
def rule : BoundRule := .sum [.predecessor 0 207783 .coefficient, .predecessor 1 207784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207783 .coefficient)
      LeftAuthority207781.bound (LeftAuthority207781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207781.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207784 .coefficient)
      LeftBound207777.bound (LeftBound207777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority207781.bound, LeftBound207777.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207781.bound, LeftBound207777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority207781.actual selector witness, LeftBound207777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207785

namespace LeftBound207789
def owner : Owner := ⟨.program ⟨257⟩, ⟨49663⟩⟩
def transferEvent : Nat := 207789
def frameStart : Nat := 207675
def rule : BoundRule := .sum [.predecessor 0 207787 .coefficient, .predecessor 1 207788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207787 .coefficient)
      LeftBound207785.bound (LeftBound207785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207788 .coefficient)
      LeftBound207766.bound (LeftBound207766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207785.bound, LeftBound207766.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207785.bound, LeftBound207766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207785.actual selector witness, LeftBound207766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207789

namespace LeftBound207802
def owner : Owner := ⟨.program ⟨257⟩, ⟨49661⟩⟩
def transferEvent : Nat := 207802
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207800 .coefficient, .predecessor 1 207801 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207800 .coefficient)
      LeftBound207623.bound (LeftBound207623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207801 .coefficient)
      LeftBound207595.bound (LeftBound207595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207623.bound, LeftBound207595.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207623.bound, LeftBound207595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207623.actual selector witness, LeftBound207595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207802

namespace LeftBound207805
def owner : Owner := ⟨.program ⟨257⟩, ⟨49661⟩⟩
def transferEvent : Nat := 207805
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207799 .summary, .result 207602 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207799 .summary)
      LeftBound207625.bound (LeftBound207625.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48592⟩⟩) (rawTerms := some (Proof.Events811.exact207799RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207602 .summary)
      LeftBound207597.bound (LeftBound207597.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49660⟩⟩) (rawTerms := some (Proof.Events810.exact207602RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207597.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207625.bound, LeftBound207597.bound]
def bound : CoeffClass := .finite ⟨2998346861024241778688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207625.bound, LeftBound207597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207625.actual selector witness, LeftBound207597.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207805

namespace LeftBound207809
def owner : Owner := ⟨.program ⟨257⟩, ⟨50031⟩⟩
def transferEvent : Nat := 207809
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 207807 .coefficient) (.predecessor 1 207808 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207807 .coefficient)
      LeftBound207802.bound (LeftBound207802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207808 .coefficient)
      LeftAuthority207512.bound (LeftAuthority207512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207512.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound207802.bound LeftAuthority207512.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207802.bound, LeftAuthority207512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound207802.actual selector witness) * (LeftAuthority207512.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound207809

namespace LeftBound207810
def owner : Owner := ⟨.program ⟨257⟩, ⟨50031⟩⟩
def transferEvent : Nat := 207810
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨50029⟩⟩]⟩ [⟨.result 207513 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207513 .coefficient)
      LeftAuthority207512.bound (LeftAuthority207512.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50029⟩⟩) (rawTerms := some (Proof.Events810.exact207513RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority207512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority207512.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority207512.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority207512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority207512.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound207810

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
