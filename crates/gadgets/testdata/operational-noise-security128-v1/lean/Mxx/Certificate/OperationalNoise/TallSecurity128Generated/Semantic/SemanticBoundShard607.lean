import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard606

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94102
def owner : Owner := ⟨.program ⟨257⟩, ⟨27707⟩⟩
def transferEvent : Nat := 94102
def frameStart : Nat := 94049
def rule : BoundRule := .identity (.predecessor 0 94101 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94101 .coefficient)
      LeftBound94099.bound (LeftBound94099.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound94099.derived selector witness)

def rawBound : CoeffClass := LeftBound94099.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound94099.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94102

namespace LeftBound94108
def owner : Owner := ⟨.program ⟨257⟩, ⟨27708⟩⟩
def transferEvent : Nat := 94108
def frameStart : Nat := 94049
def rule : BoundRule := .product (.predecessor 0 94106 .coefficient) (.predecessor 1 94107 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94106 .coefficient)
      LeftAuthority94104.bound (LeftAuthority94104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94104.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94107 .coefficient)
      LeftBound94102.bound (LeftBound94102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94102.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority94104.bound LeftBound94102.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94104.bound, LeftBound94102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority94104.actual selector witness) * (LeftBound94102.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94108

namespace LeftBound94124
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 94124
def frameStart : Nat := 94049
def rule : BoundRule := .scale (.predecessor 0 94122 .coefficient) (.value (.predecessor 1 94123 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94122 .coefficient)
      LeftAuthority94120.bound (LeftAuthority94120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94123 .coefficient)
      LeftAuthority94111.bound (LeftAuthority94111.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority94111.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority94120.bound LeftAuthority94111.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94120.bound, LeftAuthority94111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority94120.actual selector witness) * (LeftAuthority94111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94124

namespace LeftBound94127
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 94127
def frameStart : Nat := 94049
def rule : BoundRule := .identity (.predecessor 0 94126 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94126 .coefficient)
      LeftAuthority94114.bound (LeftAuthority94114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94114.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94114.derived selector witness)

def rawBound : CoeffClass := LeftAuthority94114.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority94114.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94127

namespace LeftBound94131
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 94131
def frameStart : Nat := 94049
def rule : BoundRule := .product (.predecessor 0 94129 .coefficient) (.predecessor 1 94130 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94129 .coefficient)
      LeftBound94127.bound (LeftBound94127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94130 .coefficient)
      LeftBound94124.bound (LeftBound94124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound94127.bound LeftBound94124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94127.bound, LeftBound94124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound94127.actual selector witness) * (LeftBound94124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94131

namespace LeftBound94136
def owner : Owner := ⟨.program ⟨257⟩, ⟨27709⟩⟩
def transferEvent : Nat := 94136
def frameStart : Nat := 94049
def rule : BoundRule := .sum [.predecessor 0 94134 .coefficient, .predecessor 1 94135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94134 .coefficient)
      LeftBound94131.bound (LeftBound94131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94135 .coefficient)
      LeftBound94108.bound (LeftBound94108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94131.bound, LeftBound94108.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94131.bound, LeftBound94108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound94131.actual selector witness, LeftBound94108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94136

namespace LeftBound94140
def owner : Owner := ⟨.program ⟨257⟩, ⟨27977⟩⟩
def transferEvent : Nat := 94140
def frameStart : Nat := 94049
def rule : BoundRule := .product (.predecessor 0 94138 .coefficient) (.predecessor 1 94139 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94138 .coefficient)
      LeftBound94136.bound (LeftBound94136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94139 .coefficient)
      LeftAuthority94093.bound (LeftAuthority94093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94093.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound94136.bound LeftAuthority94093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94136.bound, LeftAuthority94093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound94136.actual selector witness) * (LeftAuthority94093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94140

namespace LeftBound94151
def owner : Owner := ⟨.program ⟨257⟩, ⟨26450⟩⟩
def transferEvent : Nat := 94151
def frameStart : Nat := 94049
def rule : BoundRule := .product (.predecessor 0 94149 .coefficient) (.predecessor 1 94150 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94149 .coefficient)
      LeftAuthority94104.bound (LeftAuthority94104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94104.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94150 .coefficient)
      LeftAuthority94147.bound (LeftAuthority94147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94147.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority94104.bound LeftAuthority94147.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94104.bound, LeftAuthority94147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority94104.actual selector witness) * (LeftAuthority94147.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94151

namespace LeftBound94159
def owner : Owner := ⟨.program ⟨257⟩, ⟨26451⟩⟩
def transferEvent : Nat := 94159
def frameStart : Nat := 94049
def rule : BoundRule := .sum [.predecessor 0 94157 .coefficient, .predecessor 1 94158 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94157 .coefficient)
      LeftAuthority94155.bound (LeftAuthority94155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94158 .coefficient)
      LeftBound94151.bound (LeftBound94151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority94155.bound, LeftBound94151.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94155.bound, LeftBound94151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority94155.actual selector witness, LeftBound94151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94159

namespace LeftBound94163
def owner : Owner := ⟨.program ⟨257⟩, ⟨27978⟩⟩
def transferEvent : Nat := 94163
def frameStart : Nat := 94049
def rule : BoundRule := .sum [.predecessor 0 94161 .coefficient, .predecessor 1 94162 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94161 .coefficient)
      LeftBound94159.bound (LeftBound94159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94162 .coefficient)
      LeftBound94140.bound (LeftBound94140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94159.bound, LeftBound94140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94159.bound, LeftBound94140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound94159.actual selector witness, LeftBound94140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94163

namespace LeftBound94176
def owner : Owner := ⟨.program ⟨257⟩, ⟨27976⟩⟩
def transferEvent : Nat := 94176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94174 .coefficient, .predecessor 1 94175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94174 .coefficient)
      LeftBound93997.bound (LeftBound93997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94175 .coefficient)
      LeftBound93980.bound (LeftBound93980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact93987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93997.bound, LeftBound93980.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93997.bound, LeftBound93980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound93997.actual selector witness, LeftBound93980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94176

namespace LeftBound94179
def owner : Owner := ⟨.program ⟨257⟩, ⟨27976⟩⟩
def transferEvent : Nat := 94179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94173 .summary, .result 93987 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 94173 .summary)
      LeftBound93999.bound (LeftBound93999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26902⟩⟩) (rawTerms := some (Proof.Events367.exact94173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 93987 .summary)
      LeftBound93982.bound (LeftBound93982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27975⟩⟩) (rawTerms := some (Proof.Events367.exact93987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93999.bound, LeftBound93982.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93999.bound, LeftBound93982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound93999.actual selector witness, LeftBound93982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94179

namespace LeftBound94183
def owner : Owner := ⟨.program ⟨257⟩, ⟨28416⟩⟩
def transferEvent : Nat := 94183
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94181 .coefficient) (.predecessor 1 94182 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94181 .coefficient)
      LeftBound94176.bound (LeftBound94176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94182 .coefficient)
      LeftAuthority93902.bound (LeftAuthority93902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93902.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound94176.bound LeftAuthority93902.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94176.bound, LeftAuthority93902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound94176.actual selector witness) * (LeftAuthority93902.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94183

namespace LeftBound94184
def owner : Owner := ⟨.program ⟨257⟩, ⟨28416⟩⟩
def transferEvent : Nat := 94184
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28414⟩⟩]⟩ [⟨.result 93903 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 93903 .coefficient)
      LeftAuthority93902.bound (LeftAuthority93902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28414⟩⟩) (rawTerms := some (Proof.Events366.exact93903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93902.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority93902.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94184

namespace LeftBound94185
def owner : Owner := ⟨.program ⟨257⟩, ⟨28416⟩⟩
def transferEvent : Nat := 94185
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94180 .summary) (.transfer 94184) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 94180 .summary)
      LeftBound94179.bound (LeftBound94179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27976⟩⟩) (rawTerms := some (Proof.Events367.exact94180RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 94184)
      LeftBound94184.bound (LeftBound94184.actual selector witness) := by
  exact .transfer (LeftBound94184.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound94179.bound LeftBound94184.bound
def bound : CoeffClass := .finite ⟨32191557518723128098041228165120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94179.bound, LeftBound94184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound94179.actual selector witness) * (LeftBound94184.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94185

namespace LeftBound94196
def owner : Owner := ⟨.program ⟨257⟩, ⟨27258⟩⟩
def transferEvent : Nat := 94196
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 94194 .coefficient) (.value (.predecessor 1 94195 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 94194 .coefficient)
      LeftAuthority94192.bound (LeftAuthority94192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events367.exact94193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 94195 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority94192.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94192.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority94192.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94196

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
