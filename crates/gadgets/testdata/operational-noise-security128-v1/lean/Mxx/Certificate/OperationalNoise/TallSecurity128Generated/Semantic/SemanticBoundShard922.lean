import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard921

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound139399
def owner : Owner := ⟨.program ⟨257⟩, ⟨59297⟩⟩
def transferEvent : Nat := 139399
def frameStart : Nat := 139370
def rule : BoundRule := .product (.predecessor 0 139397 .coefficient) (.predecessor 1 139398 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139397 .coefficient)
      LeftAuthority139395.bound (LeftAuthority139395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139398 .coefficient)
      LeftAuthority139392.bound (LeftAuthority139392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority139395.bound LeftAuthority139392.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139395.bound, LeftAuthority139392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority139395.actual selector witness) * (LeftAuthority139392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139399

namespace LeftBound139403
def owner : Owner := ⟨.program ⟨257⟩, ⟨59298⟩⟩
def transferEvent : Nat := 139403
def frameStart : Nat := 139370
def rule : BoundRule := .identity (.predecessor 0 139402 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139402 .coefficient)
      LeftBound139399.bound (LeftBound139399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139399.derived selector witness)

def rawBound : CoeffClass := LeftBound139399.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound139399.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound139403

namespace LeftBound139420
def owner : Owner := ⟨.program ⟨257⟩, ⟨61198⟩⟩
def transferEvent : Nat := 139420
def frameStart : Nat := 139370
def rule : BoundRule := .sum [.predecessor 0 139418 .coefficient, .predecessor 1 139419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139418 .coefficient)
      LeftBound139403.bound (LeftBound139403.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound139403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139419 .coefficient)
      LeftAuthority139416.bound (LeftAuthority139416.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority139416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound139403.bound, LeftAuthority139416.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139403.bound, LeftAuthority139416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound139403.actual selector witness, LeftAuthority139416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139420

namespace LeftBound139423
def owner : Owner := ⟨.program ⟨257⟩, ⟨61199⟩⟩
def transferEvent : Nat := 139423
def frameStart : Nat := 139370
def rule : BoundRule := .identity (.predecessor 0 139422 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139422 .coefficient)
      LeftBound139420.bound (LeftBound139420.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound139420.derived selector witness)

def rawBound : CoeffClass := LeftBound139420.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound139420.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound139423

namespace LeftBound139429
def owner : Owner := ⟨.program ⟨257⟩, ⟨61200⟩⟩
def transferEvent : Nat := 139429
def frameStart : Nat := 139370
def rule : BoundRule := .product (.predecessor 0 139427 .coefficient) (.predecessor 1 139428 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139427 .coefficient)
      LeftAuthority139425.bound (LeftAuthority139425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139428 .coefficient)
      LeftBound139423.bound (LeftBound139423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139423.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority139425.bound LeftBound139423.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139425.bound, LeftBound139423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority139425.actual selector witness) * (LeftBound139423.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139429

namespace LeftBound139445
def owner : Owner := ⟨.program ⟨257⟩, ⟨9536⟩⟩
def transferEvent : Nat := 139445
def frameStart : Nat := 139370
def rule : BoundRule := .scale (.predecessor 0 139443 .coefficient) (.value (.predecessor 1 139444 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139443 .coefficient)
      LeftAuthority139441.bound (LeftAuthority139441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139444 .coefficient)
      LeftAuthority139432.bound (LeftAuthority139432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority139432.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority139441.bound LeftAuthority139432.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139441.bound, LeftAuthority139432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority139441.actual selector witness) * (LeftAuthority139432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound139445

namespace LeftBound139448
def owner : Owner := ⟨.program ⟨257⟩, ⟨7291⟩⟩
def transferEvent : Nat := 139448
def frameStart : Nat := 139370
def rule : BoundRule := .identity (.predecessor 0 139447 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139447 .coefficient)
      LeftAuthority139435.bound (LeftAuthority139435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139435.derived selector witness)

def rawBound : CoeffClass := LeftAuthority139435.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority139435.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound139448

namespace LeftBound139452
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def transferEvent : Nat := 139452
def frameStart : Nat := 139370
def rule : BoundRule := .product (.predecessor 0 139450 .coefficient) (.predecessor 1 139451 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139450 .coefficient)
      LeftBound139448.bound (LeftBound139448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139451 .coefficient)
      LeftBound139445.bound (LeftBound139445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound139448.bound LeftBound139445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139448.bound, LeftBound139445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound139448.actual selector witness) * (LeftBound139445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139452

namespace LeftBound139457
def owner : Owner := ⟨.program ⟨257⟩, ⟨61201⟩⟩
def transferEvent : Nat := 139457
def frameStart : Nat := 139370
def rule : BoundRule := .sum [.predecessor 0 139455 .coefficient, .predecessor 1 139456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139455 .coefficient)
      LeftBound139452.bound (LeftBound139452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139456 .coefficient)
      LeftBound139429.bound (LeftBound139429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound139452.bound, LeftBound139429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139452.bound, LeftBound139429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound139452.actual selector witness, LeftBound139429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139457

namespace LeftBound139461
def owner : Owner := ⟨.program ⟨257⟩, ⟨61385⟩⟩
def transferEvent : Nat := 139461
def frameStart : Nat := 139370
def rule : BoundRule := .product (.predecessor 0 139459 .coefficient) (.predecessor 1 139460 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139459 .coefficient)
      LeftBound139457.bound (LeftBound139457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139460 .coefficient)
      LeftAuthority139414.bound (LeftAuthority139414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound139457.bound LeftAuthority139414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139457.bound, LeftAuthority139414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound139457.actual selector witness) * (LeftAuthority139414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139461

namespace LeftBound139472
def owner : Owner := ⟨.program ⟨257⟩, ⟨59774⟩⟩
def transferEvent : Nat := 139472
def frameStart : Nat := 139370
def rule : BoundRule := .product (.predecessor 0 139470 .coefficient) (.predecessor 1 139471 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139470 .coefficient)
      LeftAuthority139425.bound (LeftAuthority139425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139471 .coefficient)
      LeftAuthority139468.bound (LeftAuthority139468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority139425.bound LeftAuthority139468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139425.bound, LeftAuthority139468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority139425.actual selector witness) * (LeftAuthority139468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139472

namespace LeftBound139480
def owner : Owner := ⟨.program ⟨257⟩, ⟨59775⟩⟩
def transferEvent : Nat := 139480
def frameStart : Nat := 139370
def rule : BoundRule := .sum [.predecessor 0 139478 .coefficient, .predecessor 1 139479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139478 .coefficient)
      LeftAuthority139476.bound (LeftAuthority139476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139476.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139479 .coefficient)
      LeftBound139472.bound (LeftBound139472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority139476.bound, LeftBound139472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority139476.bound, LeftBound139472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority139476.actual selector witness, LeftBound139472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139480

namespace LeftBound139484
def owner : Owner := ⟨.program ⟨257⟩, ⟨61386⟩⟩
def transferEvent : Nat := 139484
def frameStart : Nat := 139370
def rule : BoundRule := .sum [.predecessor 0 139482 .coefficient, .predecessor 1 139483 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139482 .coefficient)
      LeftBound139480.bound (LeftBound139480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139481RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139483 .coefficient)
      LeftBound139461.bound (LeftBound139461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound139480.bound, LeftBound139461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139480.bound, LeftBound139461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound139480.actual selector witness, LeftBound139461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139484

namespace LeftBound139497
def owner : Owner := ⟨.program ⟨257⟩, ⟨61384⟩⟩
def transferEvent : Nat := 139497
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 139495 .coefficient, .predecessor 1 139496 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139495 .coefficient)
      LeftBound139318.bound (LeftBound139318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139496 .coefficient)
      LeftBound139301.bound (LeftBound139301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound139318.bound, LeftBound139301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139318.bound, LeftBound139301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound139318.actual selector witness, LeftBound139301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139497

namespace LeftBound139500
def owner : Owner := ⟨.program ⟨257⟩, ⟨61384⟩⟩
def transferEvent : Nat := 139500
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 139494 .summary, .result 139308 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 139494 .summary)
      LeftBound139320.bound (LeftBound139320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60322⟩⟩) (rawTerms := some (Proof.Events544.exact139494RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound139320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 139308 .summary)
      LeftBound139303.bound (LeftBound139303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61383⟩⟩) (rawTerms := some (Proof.Events544.exact139308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound139303.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound139320.bound, LeftBound139303.bound]
def bound : CoeffClass := .finite ⟨2997962647681031733248, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139320.bound, LeftBound139303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound139320.actual selector witness, LeftBound139303.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound139500

namespace LeftBound139504
def owner : Owner := ⟨.program ⟨257⟩, ⟨61677⟩⟩
def transferEvent : Nat := 139504
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 139502 .coefficient) (.predecessor 1 139503 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 139502 .coefficient)
      LeftBound139497.bound (LeftBound139497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events544.exact139501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 139503 .coefficient)
      LeftAuthority139223.bound (LeftAuthority139223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events543.exact139224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority139223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority139223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound139497.bound LeftAuthority139223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound139497.bound, LeftAuthority139223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound139497.actual selector witness) * (LeftAuthority139223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound139504

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
