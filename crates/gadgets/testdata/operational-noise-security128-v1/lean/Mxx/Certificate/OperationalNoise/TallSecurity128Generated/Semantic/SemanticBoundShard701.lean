import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard700

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107739
def owner : Owner := ⟨.program ⟨257⟩, ⟨34459⟩⟩
def transferEvent : Nat := 107739
def frameStart : Nat := 107710
def rule : BoundRule := .product (.predecessor 0 107737 .coefficient) (.predecessor 1 107738 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107737 .coefficient)
      LeftAuthority107735.bound (LeftAuthority107735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107738 .coefficient)
      LeftAuthority107732.bound (LeftAuthority107732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107732.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107732.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority107735.bound LeftAuthority107732.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107735.bound, LeftAuthority107732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority107735.actual selector witness) * (LeftAuthority107732.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107739

namespace LeftBound107743
def owner : Owner := ⟨.program ⟨257⟩, ⟨34460⟩⟩
def transferEvent : Nat := 107743
def frameStart : Nat := 107710
def rule : BoundRule := .identity (.predecessor 0 107742 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107742 .coefficient)
      LeftBound107739.bound (LeftBound107739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107739.derived selector witness)

def rawBound : CoeffClass := LeftBound107739.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound107739.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound107743

namespace LeftBound107760
def owner : Owner := ⟨.program ⟨257⟩, ⟨36030⟩⟩
def transferEvent : Nat := 107760
def frameStart : Nat := 107710
def rule : BoundRule := .sum [.predecessor 0 107758 .coefficient, .predecessor 1 107759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107758 .coefficient)
      LeftBound107743.bound (LeftBound107743.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound107743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107759 .coefficient)
      LeftAuthority107756.bound (LeftAuthority107756.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority107756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107743.bound, LeftAuthority107756.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107743.bound, LeftAuthority107756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107743.actual selector witness, LeftAuthority107756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107760

namespace LeftBound107763
def owner : Owner := ⟨.program ⟨257⟩, ⟨36031⟩⟩
def transferEvent : Nat := 107763
def frameStart : Nat := 107710
def rule : BoundRule := .identity (.predecessor 0 107762 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107762 .coefficient)
      LeftBound107760.bound (LeftBound107760.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound107760.derived selector witness)

def rawBound : CoeffClass := LeftBound107760.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound107760.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound107763

namespace LeftBound107769
def owner : Owner := ⟨.program ⟨257⟩, ⟨36032⟩⟩
def transferEvent : Nat := 107769
def frameStart : Nat := 107710
def rule : BoundRule := .product (.predecessor 0 107767 .coefficient) (.predecessor 1 107768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107767 .coefficient)
      LeftAuthority107765.bound (LeftAuthority107765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107768 .coefficient)
      LeftBound107763.bound (LeftBound107763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107763.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority107765.bound LeftBound107763.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107765.bound, LeftBound107763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority107765.actual selector witness) * (LeftBound107763.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107769

namespace LeftBound107785
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 107785
def frameStart : Nat := 107710
def rule : BoundRule := .scale (.predecessor 0 107783 .coefficient) (.value (.predecessor 1 107784 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107783 .coefficient)
      LeftAuthority107781.bound (LeftAuthority107781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107781.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107781.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107784 .coefficient)
      LeftAuthority107772.bound (LeftAuthority107772.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority107772.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority107781.bound LeftAuthority107772.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107781.bound, LeftAuthority107772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority107781.actual selector witness) * (LeftAuthority107772.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound107785

namespace LeftBound107788
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 107788
def frameStart : Nat := 107710
def rule : BoundRule := .identity (.predecessor 0 107787 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107787 .coefficient)
      LeftAuthority107775.bound (LeftAuthority107775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107775.derived selector witness)

def rawBound : CoeffClass := LeftAuthority107775.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority107775.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound107788

namespace LeftBound107792
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 107792
def frameStart : Nat := 107710
def rule : BoundRule := .product (.predecessor 0 107790 .coefficient) (.predecessor 1 107791 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107790 .coefficient)
      LeftBound107788.bound (LeftBound107788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107791 .coefficient)
      LeftBound107785.bound (LeftBound107785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107785.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107788.bound LeftBound107785.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107788.bound, LeftBound107785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107788.actual selector witness) * (LeftBound107785.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107792

namespace LeftBound107797
def owner : Owner := ⟨.program ⟨257⟩, ⟨36033⟩⟩
def transferEvent : Nat := 107797
def frameStart : Nat := 107710
def rule : BoundRule := .sum [.predecessor 0 107795 .coefficient, .predecessor 1 107796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107795 .coefficient)
      LeftBound107792.bound (LeftBound107792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107796 .coefficient)
      LeftBound107769.bound (LeftBound107769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107769.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107792.bound, LeftBound107769.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107792.bound, LeftBound107769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107792.actual selector witness, LeftBound107769.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107797

namespace LeftBound107801
def owner : Owner := ⟨.program ⟨257⟩, ⟨36273⟩⟩
def transferEvent : Nat := 107801
def frameStart : Nat := 107710
def rule : BoundRule := .product (.predecessor 0 107799 .coefficient) (.predecessor 1 107800 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107799 .coefficient)
      LeftBound107797.bound (LeftBound107797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107800 .coefficient)
      LeftAuthority107754.bound (LeftAuthority107754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107797.bound LeftAuthority107754.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107797.bound, LeftAuthority107754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107797.actual selector witness) * (LeftAuthority107754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107801

namespace LeftBound107812
def owner : Owner := ⟨.program ⟨257⟩, ⟨34758⟩⟩
def transferEvent : Nat := 107812
def frameStart : Nat := 107710
def rule : BoundRule := .product (.predecessor 0 107810 .coefficient) (.predecessor 1 107811 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107810 .coefficient)
      LeftAuthority107765.bound (LeftAuthority107765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107811 .coefficient)
      LeftAuthority107808.bound (LeftAuthority107808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority107765.bound LeftAuthority107808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107765.bound, LeftAuthority107808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority107765.actual selector witness) * (LeftAuthority107808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107812

namespace LeftBound107820
def owner : Owner := ⟨.program ⟨257⟩, ⟨34759⟩⟩
def transferEvent : Nat := 107820
def frameStart : Nat := 107710
def rule : BoundRule := .sum [.predecessor 0 107818 .coefficient, .predecessor 1 107819 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107818 .coefficient)
      LeftAuthority107816.bound (LeftAuthority107816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107819 .coefficient)
      LeftBound107812.bound (LeftBound107812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority107816.bound, LeftBound107812.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107816.bound, LeftBound107812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority107816.actual selector witness, LeftBound107812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107820

namespace LeftBound107824
def owner : Owner := ⟨.program ⟨257⟩, ⟨36274⟩⟩
def transferEvent : Nat := 107824
def frameStart : Nat := 107710
def rule : BoundRule := .sum [.predecessor 0 107822 .coefficient, .predecessor 1 107823 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107822 .coefficient)
      LeftBound107820.bound (LeftBound107820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107823 .coefficient)
      LeftBound107801.bound (LeftBound107801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107801.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107820.bound, LeftBound107801.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107820.bound, LeftBound107801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107820.actual selector witness, LeftBound107801.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107824

namespace LeftBound107837
def owner : Owner := ⟨.program ⟨257⟩, ⟨36272⟩⟩
def transferEvent : Nat := 107837
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107835 .coefficient, .predecessor 1 107836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107835 .coefficient)
      LeftBound107658.bound (LeftBound107658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107836 .coefficient)
      LeftBound107641.bound (LeftBound107641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107658.bound, LeftBound107641.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107658.bound, LeftBound107641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107658.actual selector witness, LeftBound107641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107837

namespace LeftBound107840
def owner : Owner := ⟨.program ⟨257⟩, ⟨36272⟩⟩
def transferEvent : Nat := 107840
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107834 .summary, .result 107648 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107834 .summary)
      LeftBound107660.bound (LeftBound107660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35202⟩⟩) (rawTerms := some (Proof.Events421.exact107834RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 107648 .summary)
      LeftBound107643.bound (LeftBound107643.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36271⟩⟩) (rawTerms := some (Proof.Events420.exact107648RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107660.bound, LeftBound107643.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107660.bound, LeftBound107643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound107660.actual selector witness, LeftBound107643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107840

namespace LeftBound107844
def owner : Owner := ⟨.program ⟨257⟩, ⟨36656⟩⟩
def transferEvent : Nat := 107844
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107842 .coefficient) (.predecessor 1 107843 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 107842 .coefficient)
      LeftBound107837.bound (LeftBound107837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events421.exact107841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 107843 .coefficient)
      LeftAuthority107563.bound (LeftAuthority107563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events420.exact107564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107563.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound107837.bound LeftAuthority107563.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107837.bound, LeftAuthority107563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound107837.actual selector witness) * (LeftAuthority107563.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107844

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
