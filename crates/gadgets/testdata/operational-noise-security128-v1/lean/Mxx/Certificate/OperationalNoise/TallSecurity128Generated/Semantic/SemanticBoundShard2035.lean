import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2034

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound299616
def owner : Owner := ⟨.program ⟨257⟩, ⟨61186⟩⟩
def transferEvent : Nat := 299616
def frameStart : Nat := 299578
def rule : BoundRule := .sum [.predecessor 0 299614 .coefficient, .predecessor 1 299615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299614 .coefficient)
      LeftBound299599.bound (LeftBound299599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound299599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299615 .coefficient)
      LeftAuthority299612.bound (LeftAuthority299612.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority299612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound299599.bound, LeftAuthority299612.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299599.bound, LeftAuthority299612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound299599.actual selector witness, LeftAuthority299612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299616

namespace LeftBound299619
def owner : Owner := ⟨.program ⟨257⟩, ⟨61187⟩⟩
def transferEvent : Nat := 299619
def frameStart : Nat := 299578
def rule : BoundRule := .identity (.predecessor 0 299618 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299618 .coefficient)
      LeftBound299616.bound (LeftBound299616.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound299616.derived selector witness)

def rawBound : CoeffClass := LeftBound299616.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound299616.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound299619

namespace LeftBound299625
def owner : Owner := ⟨.program ⟨257⟩, ⟨61188⟩⟩
def transferEvent : Nat := 299625
def frameStart : Nat := 299578
def rule : BoundRule := .product (.predecessor 0 299623 .coefficient) (.predecessor 1 299624 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299623 .coefficient)
      LeftAuthority299621.bound (LeftAuthority299621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299624 .coefficient)
      LeftBound299619.bound (LeftBound299619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority299621.bound LeftBound299619.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299621.bound, LeftBound299619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority299621.actual selector witness) * (LeftBound299619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299625

namespace LeftBound299641
def owner : Owner := ⟨.program ⟨257⟩, ⟨9536⟩⟩
def transferEvent : Nat := 299641
def frameStart : Nat := 299578
def rule : BoundRule := .scale (.predecessor 0 299639 .coefficient) (.value (.predecessor 1 299640 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299639 .coefficient)
      LeftAuthority299637.bound (LeftAuthority299637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299640 .coefficient)
      LeftAuthority299628.bound (LeftAuthority299628.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority299628.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority299637.bound LeftAuthority299628.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299637.bound, LeftAuthority299628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority299637.actual selector witness) * (LeftAuthority299628.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound299641

namespace LeftBound299644
def owner : Owner := ⟨.program ⟨257⟩, ⟨7291⟩⟩
def transferEvent : Nat := 299644
def frameStart : Nat := 299578
def rule : BoundRule := .identity (.predecessor 0 299643 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299643 .coefficient)
      LeftAuthority299631.bound (LeftAuthority299631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299631.derived selector witness)

def rawBound : CoeffClass := LeftAuthority299631.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority299631.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound299644

namespace LeftBound299648
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def transferEvent : Nat := 299648
def frameStart : Nat := 299578
def rule : BoundRule := .product (.predecessor 0 299646 .coefficient) (.predecessor 1 299647 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299646 .coefficient)
      LeftBound299644.bound (LeftBound299644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299647 .coefficient)
      LeftBound299641.bound (LeftBound299641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound299644.bound LeftBound299641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299644.bound, LeftBound299641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound299644.actual selector witness) * (LeftBound299641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299648

namespace LeftBound299653
def owner : Owner := ⟨.program ⟨257⟩, ⟨61189⟩⟩
def transferEvent : Nat := 299653
def frameStart : Nat := 299578
def rule : BoundRule := .sum [.predecessor 0 299651 .coefficient, .predecessor 1 299652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299651 .coefficient)
      LeftBound299648.bound (LeftBound299648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299652 .coefficient)
      LeftBound299625.bound (LeftBound299625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound299648.bound, LeftBound299625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299648.bound, LeftBound299625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound299648.actual selector witness, LeftBound299625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299653

namespace LeftBound299657
def owner : Owner := ⟨.program ⟨257⟩, ⟨61352⟩⟩
def transferEvent : Nat := 299657
def frameStart : Nat := 299578
def rule : BoundRule := .product (.predecessor 0 299655 .coefficient) (.predecessor 1 299656 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299655 .coefficient)
      LeftBound299653.bound (LeftBound299653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299656 .coefficient)
      LeftAuthority299610.bound (LeftAuthority299610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound299653.bound LeftAuthority299610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299653.bound, LeftAuthority299610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound299653.actual selector witness) * (LeftAuthority299610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299657

namespace LeftBound299668
def owner : Owner := ⟨.program ⟨257⟩, ⟨59750⟩⟩
def transferEvent : Nat := 299668
def frameStart : Nat := 299578
def rule : BoundRule := .product (.predecessor 0 299666 .coefficient) (.predecessor 1 299667 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299666 .coefficient)
      LeftAuthority299621.bound (LeftAuthority299621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299667 .coefficient)
      LeftAuthority299664.bound (LeftAuthority299664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority299621.bound LeftAuthority299664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299621.bound, LeftAuthority299664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority299621.actual selector witness) * (LeftAuthority299664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299668

namespace LeftBound299676
def owner : Owner := ⟨.program ⟨257⟩, ⟨59751⟩⟩
def transferEvent : Nat := 299676
def frameStart : Nat := 299578
def rule : BoundRule := .sum [.predecessor 0 299674 .coefficient, .predecessor 1 299675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299674 .coefficient)
      LeftAuthority299672.bound (LeftAuthority299672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299675 .coefficient)
      LeftBound299668.bound (LeftBound299668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority299672.bound, LeftBound299668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299672.bound, LeftBound299668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority299672.actual selector witness, LeftBound299668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299676

namespace LeftBound299680
def owner : Owner := ⟨.program ⟨257⟩, ⟨61353⟩⟩
def transferEvent : Nat := 299680
def frameStart : Nat := 299578
def rule : BoundRule := .sum [.predecessor 0 299678 .coefficient, .predecessor 1 299679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299678 .coefficient)
      LeftBound299676.bound (LeftBound299676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299679 .coefficient)
      LeftBound299657.bound (LeftBound299657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound299676.bound, LeftBound299657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299676.bound, LeftBound299657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound299676.actual selector witness, LeftBound299657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299680

namespace LeftBound299693
def owner : Owner := ⟨.program ⟨257⟩, ⟨61351⟩⟩
def transferEvent : Nat := 299693
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 299691 .coefficient, .predecessor 1 299692 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299691 .coefficient)
      LeftBound299538.bound (LeftBound299538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299692 .coefficient)
      LeftBound299521.bound (LeftBound299521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound299538.bound, LeftBound299521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299538.bound, LeftBound299521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound299538.actual selector witness, LeftBound299521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299693

namespace LeftBound299696
def owner : Owner := ⟨.program ⟨257⟩, ⟨61351⟩⟩
def transferEvent : Nat := 299696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 299690 .summary, .result 299528 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299690 .summary)
      LeftBound299540.bound (LeftBound299540.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60292⟩⟩) (rawTerms := some (Proof.Events1170.exact299690RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound299540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299528 .summary)
      LeftBound299523.bound (LeftBound299523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61350⟩⟩) (rawTerms := some (Proof.Events1170.exact299528RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound299523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound299540.bound, LeftBound299523.bound]
def bound : CoeffClass := .finite ⟨2997962647681031733248, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299540.bound, LeftBound299523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound299540.actual selector witness, LeftBound299523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound299696

namespace LeftBound299700
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def transferEvent : Nat := 299700
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 299698 .coefficient) (.predecessor 1 299699 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 299698 .coefficient)
      LeftBound299693.bound (LeftBound299693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1170.exact299697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound299693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound299693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 299699 .coefficient)
      LeftAuthority299443.bound (LeftAuthority299443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1169.exact299444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound299693.bound LeftAuthority299443.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299693.bound, LeftAuthority299443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound299693.actual selector witness) * (LeftAuthority299443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299700

namespace LeftBound299701
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def transferEvent : Nat := 299701
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61582⟩⟩]⟩ [⟨.result 299444 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299444 .coefficient)
      LeftAuthority299443.bound (LeftAuthority299443.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61582⟩⟩) (rawTerms := some (Proof.Events1169.exact299444RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority299443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority299443.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority299443.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority299443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority299443.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound299701

namespace LeftBound299702
def owner : Owner := ⟨.program ⟨257⟩, ⟨61584⟩⟩
def transferEvent : Nat := 299702
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 299697 .summary) (.transfer 299701) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 299697 .summary)
      LeftBound299696.bound (LeftBound299696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61351⟩⟩) (rawTerms := some (Proof.Events1170.exact299697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound299696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 299701)
      LeftBound299701.bound (LeftBound299701.actual selector witness) := by
  exact .transfer (LeftBound299701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound299696.bound LeftBound299701.bound
def bound : CoeffClass := .finite ⟨32190378816049003834595889643520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound299696.bound, LeftBound299701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound299696.actual selector witness) * (LeftBound299701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound299702

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
