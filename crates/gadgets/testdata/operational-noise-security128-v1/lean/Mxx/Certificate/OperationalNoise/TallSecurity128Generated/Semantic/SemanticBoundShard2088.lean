import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2057
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2087

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound307432
def owner : Owner := ⟨.program ⟨257⟩, ⟨21729⟩⟩
def transferEvent : Nat := 307432
def frameStart : Nat := 307405
def rule : BoundRule := .identity (.predecessor 0 307431 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307431 .coefficient)
      LeftAuthority307429.bound (LeftAuthority307429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307429.derived selector witness)

def rawBound : CoeffClass := LeftAuthority307429.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority307429.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound307432

namespace LeftBound307449
def owner : Owner := ⟨.program ⟨257⟩, ⟨23246⟩⟩
def transferEvent : Nat := 307449
def frameStart : Nat := 307405
def rule : BoundRule := .sum [.predecessor 0 307447 .coefficient, .predecessor 1 307448 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307447 .coefficient)
      LeftBound307432.bound (LeftBound307432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound307432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307448 .coefficient)
      LeftAuthority307445.bound (LeftAuthority307445.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority307445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307432.bound, LeftAuthority307445.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307432.bound, LeftAuthority307445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307432.actual selector witness, LeftAuthority307445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307449

namespace LeftBound307452
def owner : Owner := ⟨.program ⟨257⟩, ⟨23247⟩⟩
def transferEvent : Nat := 307452
def frameStart : Nat := 307405
def rule : BoundRule := .identity (.predecessor 0 307451 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307451 .coefficient)
      LeftBound307449.bound (LeftBound307449.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound307449.derived selector witness)

def rawBound : CoeffClass := LeftBound307449.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound307449.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound307452

namespace LeftBound307458
def owner : Owner := ⟨.program ⟨257⟩, ⟨23248⟩⟩
def transferEvent : Nat := 307458
def frameStart : Nat := 307405
def rule : BoundRule := .product (.predecessor 0 307456 .coefficient) (.predecessor 1 307457 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307456 .coefficient)
      LeftAuthority307454.bound (LeftAuthority307454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307457 .coefficient)
      LeftBound307452.bound (LeftBound307452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307452.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority307454.bound LeftBound307452.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307454.bound, LeftBound307452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority307454.actual selector witness) * (LeftBound307452.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307458

namespace LeftBound307466
def owner : Owner := ⟨.program ⟨257⟩, ⟨23249⟩⟩
def transferEvent : Nat := 307466
def frameStart : Nat := 307405
def rule : BoundRule := .sum [.predecessor 0 307464 .coefficient, .predecessor 1 307465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307464 .coefficient)
      LeftAuthority307462.bound (LeftAuthority307462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307465 .coefficient)
      LeftBound307458.bound (LeftBound307458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307458.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority307462.bound, LeftBound307458.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307462.bound, LeftBound307458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority307462.actual selector witness, LeftBound307458.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307466

namespace LeftBound307470
def owner : Owner := ⟨.program ⟨257⟩, ⟨23556⟩⟩
def transferEvent : Nat := 307470
def frameStart : Nat := 307405
def rule : BoundRule := .product (.predecessor 0 307468 .coefficient) (.predecessor 1 307469 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307468 .coefficient)
      LeftBound307466.bound (LeftBound307466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307469 .coefficient)
      LeftAuthority307443.bound (LeftAuthority307443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound307466.bound LeftAuthority307443.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307466.bound, LeftAuthority307443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound307466.actual selector witness) * (LeftAuthority307443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307470

namespace LeftBound307481
def owner : Owner := ⟨.program ⟨257⟩, ⟨21894⟩⟩
def transferEvent : Nat := 307481
def frameStart : Nat := 307405
def rule : BoundRule := .product (.predecessor 0 307479 .coefficient) (.predecessor 1 307480 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307479 .coefficient)
      LeftAuthority307454.bound (LeftAuthority307454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307480 .coefficient)
      LeftAuthority307477.bound (LeftAuthority307477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority307454.bound LeftAuthority307477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307454.bound, LeftAuthority307477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority307454.actual selector witness) * (LeftAuthority307477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307481

namespace LeftBound307489
def owner : Owner := ⟨.program ⟨257⟩, ⟨21895⟩⟩
def transferEvent : Nat := 307489
def frameStart : Nat := 307405
def rule : BoundRule := .sum [.predecessor 0 307487 .coefficient, .predecessor 1 307488 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307487 .coefficient)
      LeftAuthority307485.bound (LeftAuthority307485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307485.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307488 .coefficient)
      LeftBound307481.bound (LeftBound307481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority307485.bound, LeftBound307481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307485.bound, LeftBound307481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority307485.actual selector witness, LeftBound307481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307489

namespace LeftBound307493
def owner : Owner := ⟨.program ⟨257⟩, ⟨23561⟩⟩
def transferEvent : Nat := 307493
def frameStart : Nat := 307405
def rule : BoundRule := .sum [.predecessor 0 307491 .coefficient, .predecessor 1 307492 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307491 .coefficient)
      LeftBound307489.bound (LeftBound307489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307489.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307492 .coefficient)
      LeftBound307470.bound (LeftBound307470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307470.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307489.bound, LeftBound307470.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307489.bound, LeftBound307470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307489.actual selector witness, LeftBound307470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307493

namespace LeftBound307506
def owner : Owner := ⟨.program ⟨257⟩, ⟨23558⟩⟩
def transferEvent : Nat := 307506
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307504 .coefficient, .predecessor 1 307505 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307504 .coefficient)
      LeftBound307359.bound (LeftBound307359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307359.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307505 .coefficient)
      LeftBound307342.bound (LeftBound307342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1200.exact307349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307342.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307359.bound, LeftBound307342.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307359.bound, LeftBound307342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307359.actual selector witness, LeftBound307342.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307506

namespace LeftBound307509
def owner : Owner := ⟨.program ⟨257⟩, ⟨23558⟩⟩
def transferEvent : Nat := 307509
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307503 .summary, .result 307349 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307503 .summary)
      LeftBound307361.bound (LeftBound307361.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22475⟩⟩) (rawTerms := some (Proof.Events1201.exact307503RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307349 .summary)
      LeftBound307344.bound (LeftBound307344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23557⟩⟩) (rawTerms := some (Proof.Events1200.exact307349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307344.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307361.bound, LeftBound307344.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307361.bound, LeftBound307344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307361.actual selector witness, LeftBound307344.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307509

namespace LeftBound307513
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def transferEvent : Nat := 307513
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 307511 .coefficient) (.predecessor 1 307512 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307511 .coefficient)
      LeftBound307506.bound (LeftBound307506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307512 .coefficient)
      LeftBound15841.bound (LeftBound15841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound307506.bound LeftBound15841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307506.bound, LeftBound15841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound307506.actual selector witness) * (LeftBound15841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307513

namespace LeftBound307514
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def transferEvent : Nat := 307514
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩ [⟨.result 15838 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15838 .coefficient)
      LeftAuthority15837.bound (LeftAuthority15837.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7155⟩⟩) (rawTerms := some (Proof.Events061.exact15838RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15837.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15837.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15837.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound307514

namespace LeftBound307515
def owner : Owner := ⟨.program ⟨257⟩, ⟨23559⟩⟩
def transferEvent : Nat := 307515
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 307510 .summary) (.transfer 307514) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307510 .summary)
      LeftBound307509.bound (LeftBound307509.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23558⟩⟩) (rawTerms := some (Proof.Events1201.exact307510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 307514)
      LeftBound307514.bound (LeftBound307514.actual selector witness) := by
  exact .transfer (LeftBound307514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound307509.bound LeftBound307514.bound
def bound : CoeffClass := .finite ⟨345626795057764889831969145180473178193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307509.bound, LeftBound307514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound307509.actual selector witness) * (LeftBound307514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307515

namespace LeftBound307530
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def transferEvent : Nat := 307530
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 307528 .coefficient) (.predecessor 1 307529 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307528 .coefficient)
      LeftBound302297.bound (LeftBound302297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1180.exact302301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound302297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound302297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307529 .coefficient)
      LeftAuthority307526.bound (LeftAuthority307526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1201.exact307527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound302297.bound LeftAuthority307526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound302297.bound, LeftAuthority307526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound302297.actual selector witness) * (LeftAuthority307526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound307530

namespace LeftBound307531
def owner : Owner := ⟨.program ⟨257⟩, ⟨20337⟩⟩
def transferEvent : Nat := 307531
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20335⟩⟩]⟩ [⟨.result 307527 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307527 .coefficient)
      LeftAuthority307526.bound (LeftAuthority307526.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20335⟩⟩) (rawTerms := some (Proof.Events1201.exact307527RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority307526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority307526.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority307526.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority307526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority307526.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound307531

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
