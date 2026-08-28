import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1845

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound272494
def owner : Owner := ⟨.program ⟨257⟩, ⟨52255⟩⟩
def transferEvent : Nat := 272494
def frameStart : Nat := 272441
def rule : BoundRule := .identity (.predecessor 0 272493 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272493 .coefficient)
      LeftBound272491.bound (LeftBound272491.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound272491.derived selector witness)

def rawBound : CoeffClass := LeftBound272491.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound272491.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound272494

namespace LeftBound272500
def owner : Owner := ⟨.program ⟨257⟩, ⟨52256⟩⟩
def transferEvent : Nat := 272500
def frameStart : Nat := 272441
def rule : BoundRule := .product (.predecessor 0 272498 .coefficient) (.predecessor 1 272499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272498 .coefficient)
      LeftAuthority272496.bound (LeftAuthority272496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272499 .coefficient)
      LeftBound272494.bound (LeftBound272494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272494.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority272496.bound LeftBound272494.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272496.bound, LeftBound272494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority272496.actual selector witness) * (LeftBound272494.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272500

namespace LeftBound272516
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 272516
def frameStart : Nat := 272441
def rule : BoundRule := .scale (.predecessor 0 272514 .coefficient) (.value (.predecessor 1 272515 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272514 .coefficient)
      LeftAuthority272512.bound (LeftAuthority272512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272515 .coefficient)
      LeftAuthority272503.bound (LeftAuthority272503.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority272503.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority272512.bound LeftAuthority272503.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272512.bound, LeftAuthority272503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority272512.actual selector witness) * (LeftAuthority272503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound272516

namespace LeftBound272519
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 272519
def frameStart : Nat := 272441
def rule : BoundRule := .identity (.predecessor 0 272518 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272518 .coefficient)
      LeftAuthority272506.bound (LeftAuthority272506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272506.derived selector witness)

def rawBound : CoeffClass := LeftAuthority272506.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority272506.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound272519

namespace LeftBound272523
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 272523
def frameStart : Nat := 272441
def rule : BoundRule := .product (.predecessor 0 272521 .coefficient) (.predecessor 1 272522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272521 .coefficient)
      LeftBound272519.bound (LeftBound272519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272522 .coefficient)
      LeftBound272516.bound (LeftBound272516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound272519.bound LeftBound272516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272519.bound, LeftBound272516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound272519.actual selector witness) * (LeftBound272516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272523

namespace LeftBound272528
def owner : Owner := ⟨.program ⟨257⟩, ⟨52257⟩⟩
def transferEvent : Nat := 272528
def frameStart : Nat := 272441
def rule : BoundRule := .sum [.predecessor 0 272526 .coefficient, .predecessor 1 272527 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272526 .coefficient)
      LeftBound272523.bound (LeftBound272523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272527 .coefficient)
      LeftBound272500.bound (LeftBound272500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272502RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272500.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound272523.bound, LeftBound272500.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272523.bound, LeftBound272500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound272523.actual selector witness, LeftBound272500.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound272528

namespace LeftBound272532
def owner : Owner := ⟨.program ⟨257⟩, ⟨52431⟩⟩
def transferEvent : Nat := 272532
def frameStart : Nat := 272441
def rule : BoundRule := .product (.predecessor 0 272530 .coefficient) (.predecessor 1 272531 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272530 .coefficient)
      LeftBound272528.bound (LeftBound272528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272528.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272531 .coefficient)
      LeftAuthority272485.bound (LeftAuthority272485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound272528.bound LeftAuthority272485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272528.bound, LeftAuthority272485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound272528.actual selector witness) * (LeftAuthority272485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272532

namespace LeftBound272543
def owner : Owner := ⟨.program ⟨257⟩, ⟨50824⟩⟩
def transferEvent : Nat := 272543
def frameStart : Nat := 272441
def rule : BoundRule := .product (.predecessor 0 272541 .coefficient) (.predecessor 1 272542 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272541 .coefficient)
      LeftAuthority272496.bound (LeftAuthority272496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272542 .coefficient)
      LeftAuthority272539.bound (LeftAuthority272539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority272496.bound LeftAuthority272539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272496.bound, LeftAuthority272539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority272496.actual selector witness) * (LeftAuthority272539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272543

namespace LeftBound272551
def owner : Owner := ⟨.program ⟨257⟩, ⟨50825⟩⟩
def transferEvent : Nat := 272551
def frameStart : Nat := 272441
def rule : BoundRule := .sum [.predecessor 0 272549 .coefficient, .predecessor 1 272550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272549 .coefficient)
      LeftAuthority272547.bound (LeftAuthority272547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272550 .coefficient)
      LeftBound272543.bound (LeftBound272543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority272547.bound, LeftBound272543.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272547.bound, LeftBound272543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority272547.actual selector witness, LeftBound272543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound272551

namespace LeftBound272555
def owner : Owner := ⟨.program ⟨257⟩, ⟨52432⟩⟩
def transferEvent : Nat := 272555
def frameStart : Nat := 272441
def rule : BoundRule := .sum [.predecessor 0 272553 .coefficient, .predecessor 1 272554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272553 .coefficient)
      LeftBound272551.bound (LeftBound272551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272554 .coefficient)
      LeftBound272532.bound (LeftBound272532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272532.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound272551.bound, LeftBound272532.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272551.bound, LeftBound272532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound272551.actual selector witness, LeftBound272532.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound272555

namespace LeftBound272568
def owner : Owner := ⟨.program ⟨257⟩, ⟨52430⟩⟩
def transferEvent : Nat := 272568
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 272566 .coefficient, .predecessor 1 272567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272566 .coefficient)
      LeftBound272389.bound (LeftBound272389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272567 .coefficient)
      LeftBound272372.bound (LeftBound272372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1063.exact272379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272372.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound272389.bound, LeftBound272372.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272389.bound, LeftBound272372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound272389.actual selector witness, LeftBound272372.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound272568

namespace LeftBound272571
def owner : Owner := ⟨.program ⟨257⟩, ⟨52430⟩⟩
def transferEvent : Nat := 272571
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 272565 .summary, .result 272379 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272565 .summary)
      LeftBound272391.bound (LeftBound272391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51369⟩⟩) (rawTerms := some (Proof.Events1064.exact272565RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound272391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272379 .summary)
      LeftBound272374.bound (LeftBound272374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52429⟩⟩) (rawTerms := some (Proof.Events1063.exact272379RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound272374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound272391.bound, LeftBound272374.bound]
def bound : CoeffClass := .finite ⟨2997889464187086962688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272391.bound, LeftBound272374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound272391.actual selector witness, LeftBound272374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound272571

namespace LeftBound272575
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def transferEvent : Nat := 272575
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 272573 .coefficient) (.predecessor 1 272574 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272573 .coefficient)
      LeftBound272568.bound (LeftBound272568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound272568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound272568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272574 .coefficient)
      LeftAuthority272294.bound (LeftAuthority272294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1063.exact272295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272294.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound272568.bound LeftAuthority272294.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272568.bound, LeftAuthority272294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound272568.actual selector witness) * (LeftAuthority272294.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272575

namespace LeftBound272576
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def transferEvent : Nat := 272576
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52695⟩⟩]⟩ [⟨.result 272295 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272295 .coefficient)
      LeftAuthority272294.bound (LeftAuthority272294.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52695⟩⟩) (rawTerms := some (Proof.Events1063.exact272295RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272294.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority272294.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority272294.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound272576

namespace LeftBound272577
def owner : Owner := ⟨.program ⟨257⟩, ⟨52697⟩⟩
def transferEvent : Nat := 272577
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 272572 .summary) (.transfer 272576) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 272572 .summary)
      LeftBound272571.bound (LeftBound272571.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52430⟩⟩) (rawTerms := some (Proof.Events1064.exact272572RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound272571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 272576)
      LeftBound272576.bound (LeftBound272576.actual selector witness) := by
  exact .transfer (LeftBound272576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound272571.bound LeftBound272576.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound272571.bound, LeftBound272576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound272571.actual selector witness) * (LeftBound272576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound272577

namespace LeftBound272588
def owner : Owner := ⟨.program ⟨257⟩, ⟨51592⟩⟩
def transferEvent : Nat := 272588
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 272586 .coefficient) (.value (.predecessor 1 272587 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 272586 .coefficient)
      LeftAuthority272584.bound (LeftAuthority272584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1064.exact272585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority272584.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority272584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 272587 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority272584.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority272584.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority272584.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound272588

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
