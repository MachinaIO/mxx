import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard193

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound34638
def owner : Owner := ⟨.program ⟨257⟩, ⟨36063⟩⟩
def transferEvent : Nat := 34638
def frameStart : Nat := 34585
def rule : BoundRule := .identity (.predecessor 0 34637 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34637 .coefficient)
      LeftBound34635.bound (LeftBound34635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34635.derived selector witness)

def rawBound : CoeffClass := LeftBound34635.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound34635.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34638

namespace LeftBound34644
def owner : Owner := ⟨.program ⟨257⟩, ⟨36064⟩⟩
def transferEvent : Nat := 34644
def frameStart : Nat := 34585
def rule : BoundRule := .product (.predecessor 0 34642 .coefficient) (.predecessor 1 34643 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34642 .coefficient)
      LeftAuthority34640.bound (LeftAuthority34640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34643 .coefficient)
      LeftBound34638.bound (LeftBound34638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority34640.bound LeftBound34638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34640.bound, LeftBound34638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority34640.actual selector witness) * (LeftBound34638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34644

namespace LeftBound34660
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 34660
def frameStart : Nat := 34585
def rule : BoundRule := .scale (.predecessor 0 34658 .coefficient) (.value (.predecessor 1 34659 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34658 .coefficient)
      LeftAuthority34656.bound (LeftAuthority34656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34659 .coefficient)
      LeftAuthority34647.bound (LeftAuthority34647.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority34647.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority34656.bound LeftAuthority34647.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34656.bound, LeftAuthority34647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34656.actual selector witness) * (LeftAuthority34647.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound34660

namespace LeftBound34663
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 34663
def frameStart : Nat := 34585
def rule : BoundRule := .identity (.predecessor 0 34662 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34662 .coefficient)
      LeftAuthority34650.bound (LeftAuthority34650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34650.derived selector witness)

def rawBound : CoeffClass := LeftAuthority34650.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority34650.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34663

namespace LeftBound34667
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 34667
def frameStart : Nat := 34585
def rule : BoundRule := .product (.predecessor 0 34665 .coefficient) (.predecessor 1 34666 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34665 .coefficient)
      LeftBound34663.bound (LeftBound34663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34666 .coefficient)
      LeftBound34660.bound (LeftBound34660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34663.bound LeftBound34660.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34663.bound, LeftBound34660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34663.actual selector witness) * (LeftBound34660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34667

namespace LeftBound34672
def owner : Owner := ⟨.program ⟨257⟩, ⟨36065⟩⟩
def transferEvent : Nat := 34672
def frameStart : Nat := 34585
def rule : BoundRule := .sum [.predecessor 0 34670 .coefficient, .predecessor 1 34671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34670 .coefficient)
      LeftBound34667.bound (LeftBound34667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34671 .coefficient)
      LeftBound34644.bound (LeftBound34644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34644.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34667.bound, LeftBound34644.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34667.bound, LeftBound34644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34667.actual selector witness, LeftBound34644.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34672

namespace LeftBound34676
def owner : Owner := ⟨.program ⟨257⟩, ⟨36361⟩⟩
def transferEvent : Nat := 34676
def frameStart : Nat := 34585
def rule : BoundRule := .product (.predecessor 0 34674 .coefficient) (.predecessor 1 34675 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34674 .coefficient)
      LeftBound34672.bound (LeftBound34672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34675 .coefficient)
      LeftAuthority34629.bound (LeftAuthority34629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34672.bound LeftAuthority34629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34672.bound, LeftAuthority34629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34672.actual selector witness) * (LeftAuthority34629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34676

namespace LeftBound34687
def owner : Owner := ⟨.program ⟨257⟩, ⟨34822⟩⟩
def transferEvent : Nat := 34687
def frameStart : Nat := 34585
def rule : BoundRule := .product (.predecessor 0 34685 .coefficient) (.predecessor 1 34686 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34685 .coefficient)
      LeftAuthority34640.bound (LeftAuthority34640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34686 .coefficient)
      LeftAuthority34683.bound (LeftAuthority34683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority34640.bound LeftAuthority34683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34640.bound, LeftAuthority34683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority34640.actual selector witness) * (LeftAuthority34683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34687

namespace LeftBound34695
def owner : Owner := ⟨.program ⟨257⟩, ⟨34823⟩⟩
def transferEvent : Nat := 34695
def frameStart : Nat := 34585
def rule : BoundRule := .sum [.predecessor 0 34693 .coefficient, .predecessor 1 34694 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34693 .coefficient)
      LeftAuthority34691.bound (LeftAuthority34691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34694 .coefficient)
      LeftBound34687.bound (LeftBound34687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34691.bound, LeftBound34687.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34691.bound, LeftBound34687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority34691.actual selector witness, LeftBound34687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34695

namespace LeftBound34699
def owner : Owner := ⟨.program ⟨257⟩, ⟨36362⟩⟩
def transferEvent : Nat := 34699
def frameStart : Nat := 34585
def rule : BoundRule := .sum [.predecessor 0 34697 .coefficient, .predecessor 1 34698 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34697 .coefficient)
      LeftBound34695.bound (LeftBound34695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34698 .coefficient)
      LeftBound34676.bound (LeftBound34676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34695.bound, LeftBound34676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34695.bound, LeftBound34676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34695.actual selector witness, LeftBound34676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34699

namespace LeftBound34712
def owner : Owner := ⟨.program ⟨257⟩, ⟨36360⟩⟩
def transferEvent : Nat := 34712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 34710 .coefficient, .predecessor 1 34711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34710 .coefficient)
      LeftBound34533.bound (LeftBound34533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34711 .coefficient)
      LeftBound34516.bound (LeftBound34516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34533.bound, LeftBound34516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34533.bound, LeftBound34516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34533.actual selector witness, LeftBound34516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34712

namespace LeftBound34715
def owner : Owner := ⟨.program ⟨257⟩, ⟨36360⟩⟩
def transferEvent : Nat := 34715
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 34709 .summary, .result 34523 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34709 .summary)
      LeftBound34535.bound (LeftBound34535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35282⟩⟩) (rawTerms := some (Proof.Events135.exact34709RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34523 .summary)
      LeftBound34518.bound (LeftBound34518.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36359⟩⟩) (rawTerms := some (Proof.Events134.exact34523RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34535.bound, LeftBound34518.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34535.bound, LeftBound34518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34535.actual selector witness, LeftBound34518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34715

namespace LeftBound34719
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def transferEvent : Nat := 34719
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34717 .coefficient) (.predecessor 1 34718 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34717 .coefficient)
      LeftBound34712.bound (LeftBound34712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34718 .coefficient)
      LeftAuthority34438.bound (LeftAuthority34438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34438.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34712.bound LeftAuthority34438.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34712.bound, LeftAuthority34438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34712.actual selector witness) * (LeftAuthority34438.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34719

namespace LeftBound34720
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def transferEvent : Nat := 34720
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36854⟩⟩]⟩ [⟨.result 34439 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34439 .coefficient)
      LeftAuthority34438.bound (LeftAuthority34438.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36854⟩⟩) (rawTerms := some (Proof.Events134.exact34439RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34438.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34438.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34438.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34720

namespace LeftBound34721
def owner : Owner := ⟨.program ⟨257⟩, ⟨36856⟩⟩
def transferEvent : Nat := 34721
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 34716 .summary) (.transfer 34720) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34716 .summary)
      LeftBound34715.bound (LeftBound34715.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36360⟩⟩) (rawTerms := some (Proof.Events135.exact34716RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 34720)
      LeftBound34720.bound (LeftBound34720.actual selector witness) := by
  exact .transfer (LeftBound34720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34715.bound LeftBound34720.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34715.bound, LeftBound34720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34715.actual selector witness) * (LeftBound34720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34721

namespace LeftBound34732
def owner : Owner := ⟨.program ⟨257⟩, ⟨35678⟩⟩
def transferEvent : Nat := 34732
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 34730 .coefficient) (.value (.predecessor 1 34731 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34730 .coefficient)
      LeftAuthority34728.bound (LeftAuthority34728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34731 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority34728.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34728.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34728.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound34732

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
