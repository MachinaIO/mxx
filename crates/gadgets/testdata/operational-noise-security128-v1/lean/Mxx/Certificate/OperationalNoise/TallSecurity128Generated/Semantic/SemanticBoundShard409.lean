import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard408

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65546
def owner : Owner := ⟨.program ⟨257⟩, ⟨69035⟩⟩
def transferEvent : Nat := 65546
def frameStart : Nat := 65490
def rule : BoundRule := .sum [.predecessor 0 65544 .coefficient, .predecessor 1 65545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65544 .coefficient)
      LeftBound65529.bound (LeftBound65529.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65545 .coefficient)
      LeftAuthority65542.bound (LeftAuthority65542.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority65542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65529.bound, LeftAuthority65542.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65529.bound, LeftAuthority65542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65529.actual selector witness, LeftAuthority65542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65546

namespace LeftBound65549
def owner : Owner := ⟨.program ⟨257⟩, ⟨69036⟩⟩
def transferEvent : Nat := 65549
def frameStart : Nat := 65490
def rule : BoundRule := .identity (.predecessor 0 65548 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65548 .coefficient)
      LeftBound65546.bound (LeftBound65546.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound65546.derived selector witness)

def rawBound : CoeffClass := LeftBound65546.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound65546.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound65549

namespace LeftBound65555
def owner : Owner := ⟨.program ⟨257⟩, ⟨69037⟩⟩
def transferEvent : Nat := 65555
def frameStart : Nat := 65490
def rule : BoundRule := .product (.predecessor 0 65553 .coefficient) (.predecessor 1 65554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65553 .coefficient)
      LeftAuthority65551.bound (LeftAuthority65551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65554 .coefficient)
      LeftBound65549.bound (LeftBound65549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority65551.bound LeftBound65549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65551.bound, LeftBound65549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority65551.actual selector witness) * (LeftBound65549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65555

namespace LeftBound65563
def owner : Owner := ⟨.program ⟨257⟩, ⟨69038⟩⟩
def transferEvent : Nat := 65563
def frameStart : Nat := 65490
def rule : BoundRule := .sum [.predecessor 0 65561 .coefficient, .predecessor 1 65562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65561 .coefficient)
      LeftAuthority65559.bound (LeftAuthority65559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65559.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65562 .coefficient)
      LeftBound65555.bound (LeftBound65555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority65559.bound, LeftBound65555.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65559.bound, LeftBound65555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority65559.actual selector witness, LeftBound65555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65563

namespace LeftBound65567
def owner : Owner := ⟨.program ⟨257⟩, ⟨70731⟩⟩
def transferEvent : Nat := 65567
def frameStart : Nat := 65490
def rule : BoundRule := .product (.predecessor 0 65565 .coefficient) (.predecessor 1 65566 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65565 .coefficient)
      LeftBound65563.bound (LeftBound65563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65566 .coefficient)
      LeftAuthority65540.bound (LeftAuthority65540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound65563.bound LeftAuthority65540.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65563.bound, LeftAuthority65540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound65563.actual selector witness) * (LeftAuthority65540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65567

namespace LeftBound65578
def owner : Owner := ⟨.program ⟨257⟩, ⟨67102⟩⟩
def transferEvent : Nat := 65578
def frameStart : Nat := 65490
def rule : BoundRule := .product (.predecessor 0 65576 .coefficient) (.predecessor 1 65577 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65576 .coefficient)
      LeftAuthority65551.bound (LeftAuthority65551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65577 .coefficient)
      LeftAuthority65574.bound (LeftAuthority65574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65551.bound LeftAuthority65574.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65551.bound, LeftAuthority65574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority65551.actual selector witness) * (LeftAuthority65574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65578

namespace LeftBound65586
def owner : Owner := ⟨.program ⟨257⟩, ⟨67103⟩⟩
def transferEvent : Nat := 65586
def frameStart : Nat := 65490
def rule : BoundRule := .sum [.predecessor 0 65584 .coefficient, .predecessor 1 65585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65584 .coefficient)
      LeftAuthority65582.bound (LeftAuthority65582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65585 .coefficient)
      LeftBound65578.bound (LeftBound65578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority65582.bound, LeftBound65578.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65582.bound, LeftBound65578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority65582.actual selector witness, LeftBound65578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65586

namespace LeftBound65590
def owner : Owner := ⟨.program ⟨257⟩, ⟨70743⟩⟩
def transferEvent : Nat := 65590
def frameStart : Nat := 65490
def rule : BoundRule := .sum [.predecessor 0 65588 .coefficient, .predecessor 1 65589 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65588 .coefficient)
      LeftBound65586.bound (LeftBound65586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65589 .coefficient)
      LeftBound65567.bound (LeftBound65567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65586.bound, LeftBound65567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65586.bound, LeftBound65567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65586.actual selector witness, LeftBound65567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65590

namespace LeftBound65603
def owner : Owner := ⟨.program ⟨257⟩, ⟨70733⟩⟩
def transferEvent : Nat := 65603
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65601 .coefficient, .predecessor 1 65602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65601 .coefficient)
      LeftBound65432.bound (LeftBound65432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65602 .coefficient)
      LeftBound65415.bound (LeftBound65415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65415.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65432.bound, LeftBound65415.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65432.bound, LeftBound65415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65432.actual selector witness, LeftBound65415.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65603

namespace LeftBound65606
def owner : Owner := ⟨.program ⟨257⟩, ⟨70733⟩⟩
def transferEvent : Nat := 65606
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 65600 .summary, .result 65422 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65600 .summary)
      LeftBound65434.bound (LeftBound65434.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68220⟩⟩) (rawTerms := some (Proof.Events256.exact65600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65422 .summary)
      LeftBound65417.bound (LeftBound65417.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70732⟩⟩) (rawTerms := some (Proof.Events255.exact65422RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65434.bound, LeftBound65417.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65434.bound, LeftBound65417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65434.actual selector witness, LeftBound65417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65606

namespace LeftBound65630
def owner : Owner := ⟨.program ⟨257⟩, ⟨25575⟩⟩
def transferEvent : Nat := 65630
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 65628 .coefficient) (.predecessor 1 65629 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65628 .coefficient)
      LeftAuthority2544.bound (LeftAuthority2544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65629 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2544.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2544.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2544.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65630

namespace LeftBound65635
def owner : Owner := ⟨.program ⟨257⟩, ⟨10757⟩⟩
def transferEvent : Nat := 65635
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65633 .coefficient) (.predecessor 1 65634 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65633 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65634 .coefficient)
      LeftBound21588.bound (LeftBound21588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound21588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound21588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound21588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65635

namespace LeftBound65640
def owner : Owner := ⟨.program ⟨257⟩, ⟨25576⟩⟩
def transferEvent : Nat := 65640
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65638 .coefficient, .predecessor 1 65639 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65638 .coefficient)
      LeftBound65635.bound (LeftBound65635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65639 .coefficient)
      LeftBound65630.bound (LeftBound65630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65635.bound, LeftBound65630.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65635.bound, LeftBound65630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65635.actual selector witness, LeftBound65630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65640

namespace LeftBound65644
def owner : Owner := ⟨.program ⟨257⟩, ⟨25577⟩⟩
def transferEvent : Nat := 65644
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65642 .coefficient, .predecessor 1 65643 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65642 .coefficient)
      LeftBound65640.bound (LeftBound65640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65643 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65640.bound, LeftBound21580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65640.bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65640.actual selector witness, LeftBound21580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65644

namespace LeftBound65645
def owner : Owner := ⟨.program ⟨257⟩, ⟨25577⟩⟩
def transferEvent : Nat := 65645
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩ [⟨.result 21581 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21581 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21580.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21580.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65645

namespace LeftBound65650
def owner : Owner := ⟨.program ⟨257⟩, ⟨62657⟩⟩
def transferEvent : Nat := 65650
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65648 .coefficient) (.predecessor 1 65649 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65648 .coefficient)
      LeftBound65644.bound (LeftBound65644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65649 .coefficient)
      LeftAuthority2547.bound (LeftAuthority2547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound65644.bound LeftAuthority2547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65644.bound, LeftAuthority2547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound65644.actual selector witness) * (LeftAuthority2547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65650

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
