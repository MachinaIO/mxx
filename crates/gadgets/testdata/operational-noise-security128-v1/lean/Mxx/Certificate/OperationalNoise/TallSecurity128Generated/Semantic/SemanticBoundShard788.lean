import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard787

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound120440
def owner : Owner := ⟨.program ⟨257⟩, ⟨45060⟩⟩
def transferEvent : Nat := 120440
def frameStart : Nat := 120407
def rule : BoundRule := .identity (.predecessor 0 120439 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120439 .coefficient)
      LeftBound120436.bound (LeftBound120436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120436.derived selector witness)

def rawBound : CoeffClass := LeftBound120436.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound120436.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound120440

namespace LeftBound120457
def owner : Owner := ⟨.program ⟨257⟩, ⟨46730⟩⟩
def transferEvent : Nat := 120457
def frameStart : Nat := 120407
def rule : BoundRule := .sum [.predecessor 0 120455 .coefficient, .predecessor 1 120456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120455 .coefficient)
      LeftBound120440.bound (LeftBound120440.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound120440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120456 .coefficient)
      LeftAuthority120453.bound (LeftAuthority120453.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority120453.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound120440.bound, LeftAuthority120453.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120440.bound, LeftAuthority120453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound120440.actual selector witness, LeftAuthority120453.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120457

namespace LeftBound120460
def owner : Owner := ⟨.program ⟨257⟩, ⟨46731⟩⟩
def transferEvent : Nat := 120460
def frameStart : Nat := 120407
def rule : BoundRule := .identity (.predecessor 0 120459 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120459 .coefficient)
      LeftBound120457.bound (LeftBound120457.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound120457.derived selector witness)

def rawBound : CoeffClass := LeftBound120457.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound120457.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound120460

namespace LeftBound120466
def owner : Owner := ⟨.program ⟨257⟩, ⟨46732⟩⟩
def transferEvent : Nat := 120466
def frameStart : Nat := 120407
def rule : BoundRule := .product (.predecessor 0 120464 .coefficient) (.predecessor 1 120465 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120464 .coefficient)
      LeftAuthority120462.bound (LeftAuthority120462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120465 .coefficient)
      LeftBound120460.bound (LeftBound120460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120460.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority120462.bound LeftBound120460.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120462.bound, LeftBound120460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority120462.actual selector witness) * (LeftBound120460.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound120466

namespace LeftBound120482
def owner : Owner := ⟨.program ⟨257⟩, ⟨9563⟩⟩
def transferEvent : Nat := 120482
def frameStart : Nat := 120407
def rule : BoundRule := .scale (.predecessor 0 120480 .coefficient) (.value (.predecessor 1 120481 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120480 .coefficient)
      LeftAuthority120478.bound (LeftAuthority120478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120481 .coefficient)
      LeftAuthority120469.bound (LeftAuthority120469.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority120469.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority120478.bound LeftAuthority120469.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120478.bound, LeftAuthority120469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority120478.actual selector witness) * (LeftAuthority120469.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound120482

namespace LeftBound120485
def owner : Owner := ⟨.program ⟨257⟩, ⟨7301⟩⟩
def transferEvent : Nat := 120485
def frameStart : Nat := 120407
def rule : BoundRule := .identity (.predecessor 0 120484 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120484 .coefficient)
      LeftAuthority120472.bound (LeftAuthority120472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120472.derived selector witness)

def rawBound : CoeffClass := LeftAuthority120472.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority120472.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound120485

namespace LeftBound120489
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def transferEvent : Nat := 120489
def frameStart : Nat := 120407
def rule : BoundRule := .product (.predecessor 0 120487 .coefficient) (.predecessor 1 120488 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120487 .coefficient)
      LeftBound120485.bound (LeftBound120485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120488 .coefficient)
      LeftBound120482.bound (LeftBound120482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120482.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound120485.bound LeftBound120482.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120485.bound, LeftBound120482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound120485.actual selector witness) * (LeftBound120482.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound120489

namespace LeftBound120494
def owner : Owner := ⟨.program ⟨257⟩, ⟨46733⟩⟩
def transferEvent : Nat := 120494
def frameStart : Nat := 120407
def rule : BoundRule := .sum [.predecessor 0 120492 .coefficient, .predecessor 1 120493 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120492 .coefficient)
      LeftBound120489.bound (LeftBound120489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120493 .coefficient)
      LeftBound120466.bound (LeftBound120466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound120489.bound, LeftBound120466.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120489.bound, LeftBound120466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound120489.actual selector witness, LeftBound120466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120494

namespace LeftBound120498
def owner : Owner := ⟨.program ⟨257⟩, ⟨46938⟩⟩
def transferEvent : Nat := 120498
def frameStart : Nat := 120407
def rule : BoundRule := .product (.predecessor 0 120496 .coefficient) (.predecessor 1 120497 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120496 .coefficient)
      LeftBound120494.bound (LeftBound120494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120497 .coefficient)
      LeftAuthority120451.bound (LeftAuthority120451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound120494.bound LeftAuthority120451.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120494.bound, LeftAuthority120451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound120494.actual selector witness) * (LeftAuthority120451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound120498

namespace LeftBound120509
def owner : Owner := ⟨.program ⟨257⟩, ⟨45438⟩⟩
def transferEvent : Nat := 120509
def frameStart : Nat := 120407
def rule : BoundRule := .product (.predecessor 0 120507 .coefficient) (.predecessor 1 120508 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120507 .coefficient)
      LeftAuthority120462.bound (LeftAuthority120462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120508 .coefficient)
      LeftAuthority120505.bound (LeftAuthority120505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority120462.bound LeftAuthority120505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120462.bound, LeftAuthority120505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority120462.actual selector witness) * (LeftAuthority120505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound120509

namespace LeftBound120517
def owner : Owner := ⟨.program ⟨257⟩, ⟨45439⟩⟩
def transferEvent : Nat := 120517
def frameStart : Nat := 120407
def rule : BoundRule := .sum [.predecessor 0 120515 .coefficient, .predecessor 1 120516 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120515 .coefficient)
      LeftAuthority120513.bound (LeftAuthority120513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120513.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120516 .coefficient)
      LeftBound120509.bound (LeftBound120509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120509.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority120513.bound, LeftBound120509.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120513.bound, LeftBound120509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority120513.actual selector witness, LeftBound120509.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120517

namespace LeftBound120521
def owner : Owner := ⟨.program ⟨257⟩, ⟨46939⟩⟩
def transferEvent : Nat := 120521
def frameStart : Nat := 120407
def rule : BoundRule := .sum [.predecessor 0 120519 .coefficient, .predecessor 1 120520 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120519 .coefficient)
      LeftBound120517.bound (LeftBound120517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120520 .coefficient)
      LeftBound120498.bound (LeftBound120498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound120517.bound, LeftBound120498.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120517.bound, LeftBound120498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound120517.actual selector witness, LeftBound120498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120521

namespace LeftBound120534
def owner : Owner := ⟨.program ⟨257⟩, ⟨46937⟩⟩
def transferEvent : Nat := 120534
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 120532 .coefficient, .predecessor 1 120533 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120532 .coefficient)
      LeftBound120355.bound (LeftBound120355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120533 .coefficient)
      LeftBound120338.bound (LeftBound120338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound120355.bound, LeftBound120338.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120355.bound, LeftBound120338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound120355.actual selector witness, LeftBound120338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120534

namespace LeftBound120537
def owner : Owner := ⟨.program ⟨257⟩, ⟨46937⟩⟩
def transferEvent : Nat := 120537
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 120531 .summary, .result 120345 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 120531 .summary)
      LeftBound120357.bound (LeftBound120357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45872⟩⟩) (rawTerms := some (Proof.Events470.exact120531RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound120357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 120345 .summary)
      LeftBound120340.bound (LeftBound120340.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46936⟩⟩) (rawTerms := some (Proof.Events470.exact120345RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound120340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound120357.bound, LeftBound120340.bound]
def bound : CoeffClass := .finite ⟨2998328565150755586048, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120357.bound, LeftBound120340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound120357.actual selector witness, LeftBound120340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound120537

namespace LeftBound120541
def owner : Owner := ⟨.program ⟨257⟩, ⟨47251⟩⟩
def transferEvent : Nat := 120541
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 120539 .coefficient) (.predecessor 1 120540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 120539 .coefficient)
      LeftBound120534.bound (LeftBound120534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events470.exact120538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound120534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound120534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 120540 .coefficient)
      LeftAuthority120260.bound (LeftAuthority120260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events469.exact120261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120260.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound120534.bound LeftAuthority120260.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120534.bound, LeftAuthority120260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound120534.actual selector witness) * (LeftAuthority120260.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound120541

namespace LeftBound120542
def owner : Owner := ⟨.program ⟨257⟩, ⟨47251⟩⟩
def transferEvent : Nat := 120542
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47249⟩⟩]⟩ [⟨.result 120261 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 120261 .coefficient)
      LeftAuthority120260.bound (LeftAuthority120260.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47249⟩⟩) (rawTerms := some (Proof.Events469.exact120261RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority120260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority120260.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority120260.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority120260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority120260.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound120542

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
