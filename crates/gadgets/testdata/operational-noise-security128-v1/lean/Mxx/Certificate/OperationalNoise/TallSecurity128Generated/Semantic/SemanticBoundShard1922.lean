import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1921

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound283730
def owner : Owner := ⟨.program ⟨257⟩, ⟨30342⟩⟩
def transferEvent : Nat := 283730
def frameStart : Nat := 283680
def rule : BoundRule := .sum [.predecessor 0 283728 .coefficient, .predecessor 1 283729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283728 .coefficient)
      LeftBound283713.bound (LeftBound283713.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound283713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283729 .coefficient)
      LeftAuthority283726.bound (LeftAuthority283726.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority283726.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283713.bound, LeftAuthority283726.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283713.bound, LeftAuthority283726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283713.actual selector witness, LeftAuthority283726.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283730

namespace LeftBound283733
def owner : Owner := ⟨.program ⟨257⟩, ⟨30343⟩⟩
def transferEvent : Nat := 283733
def frameStart : Nat := 283680
def rule : BoundRule := .identity (.predecessor 0 283732 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283732 .coefficient)
      LeftBound283730.bound (LeftBound283730.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound283730.derived selector witness)

def rawBound : CoeffClass := LeftBound283730.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound283730.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound283733

namespace LeftBound283739
def owner : Owner := ⟨.program ⟨257⟩, ⟨30344⟩⟩
def transferEvent : Nat := 283739
def frameStart : Nat := 283680
def rule : BoundRule := .product (.predecessor 0 283737 .coefficient) (.predecessor 1 283738 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283737 .coefficient)
      LeftAuthority283735.bound (LeftAuthority283735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283738 .coefficient)
      LeftBound283733.bound (LeftBound283733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283733.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority283735.bound LeftBound283733.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283735.bound, LeftBound283733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority283735.actual selector witness) * (LeftBound283733.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283739

namespace LeftBound283753
def owner : Owner := ⟨.program ⟨257⟩, ⟨9548⟩⟩
def transferEvent : Nat := 283753
def frameStart : Nat := 283680
def rule : BoundRule := .scale (.predecessor 0 283751 .coefficient) (.value (.predecessor 1 283752 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283751 .coefficient)
      LeftAuthority283749.bound (LeftAuthority283749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283749.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283752 .coefficient)
      LeftAuthority283683.bound (LeftAuthority283683.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority283683.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority283749.bound LeftAuthority283683.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283749.bound, LeftAuthority283683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority283749.actual selector witness) * (LeftAuthority283683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound283753

namespace LeftBound283756
def owner : Owner := ⟨.program ⟨257⟩, ⟨7296⟩⟩
def transferEvent : Nat := 283756
def frameStart : Nat := 283680
def rule : BoundRule := .identity (.predecessor 0 283755 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283755 .coefficient)
      LeftAuthority283743.bound (LeftAuthority283743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283743.derived selector witness)

def rawBound : CoeffClass := LeftAuthority283743.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority283743.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound283756

namespace LeftBound283760
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def transferEvent : Nat := 283760
def frameStart : Nat := 283680
def rule : BoundRule := .product (.predecessor 0 283758 .coefficient) (.predecessor 1 283759 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283758 .coefficient)
      LeftBound283756.bound (LeftBound283756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283759 .coefficient)
      LeftBound283753.bound (LeftBound283753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283756.bound LeftBound283753.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283756.bound, LeftBound283753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283756.actual selector witness) * (LeftBound283753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283760

namespace LeftBound283765
def owner : Owner := ⟨.program ⟨257⟩, ⟨30345⟩⟩
def transferEvent : Nat := 283765
def frameStart : Nat := 283680
def rule : BoundRule := .sum [.predecessor 0 283763 .coefficient, .predecessor 1 283764 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283763 .coefficient)
      LeftBound283760.bound (LeftBound283760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283764 .coefficient)
      LeftBound283739.bound (LeftBound283739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283760.bound, LeftBound283739.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283760.bound, LeftBound283739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283760.actual selector witness, LeftBound283739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283765

namespace LeftBound283769
def owner : Owner := ⟨.program ⟨257⟩, ⟨30536⟩⟩
def transferEvent : Nat := 283769
def frameStart : Nat := 283680
def rule : BoundRule := .product (.predecessor 0 283767 .coefficient) (.predecessor 1 283768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283767 .coefficient)
      LeftBound283765.bound (LeftBound283765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283768 .coefficient)
      LeftAuthority283724.bound (LeftAuthority283724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283765.bound LeftAuthority283724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283765.bound, LeftAuthority283724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283765.actual selector witness) * (LeftAuthority283724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283769

namespace LeftBound283780
def owner : Owner := ⟨.program ⟨257⟩, ⟨29042⟩⟩
def transferEvent : Nat := 283780
def frameStart : Nat := 283680
def rule : BoundRule := .product (.predecessor 0 283778 .coefficient) (.predecessor 1 283779 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283778 .coefficient)
      LeftAuthority283735.bound (LeftAuthority283735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283735.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283779 .coefficient)
      LeftAuthority283776.bound (LeftAuthority283776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority283735.bound LeftAuthority283776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283735.bound, LeftAuthority283776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority283735.actual selector witness) * (LeftAuthority283776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283780

namespace LeftBound283788
def owner : Owner := ⟨.program ⟨257⟩, ⟨29043⟩⟩
def transferEvent : Nat := 283788
def frameStart : Nat := 283680
def rule : BoundRule := .sum [.predecessor 0 283786 .coefficient, .predecessor 1 283787 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283786 .coefficient)
      LeftAuthority283784.bound (LeftAuthority283784.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283784.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283787 .coefficient)
      LeftBound283780.bound (LeftBound283780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority283784.bound, LeftBound283780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283784.bound, LeftBound283780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority283784.actual selector witness, LeftBound283780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283788

namespace LeftBound283792
def owner : Owner := ⟨.program ⟨257⟩, ⟨30537⟩⟩
def transferEvent : Nat := 283792
def frameStart : Nat := 283680
def rule : BoundRule := .sum [.predecessor 0 283790 .coefficient, .predecessor 1 283791 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283790 .coefficient)
      LeftBound283788.bound (LeftBound283788.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283788.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283788.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283791 .coefficient)
      LeftBound283769.bound (LeftBound283769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283769.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283788.bound, LeftBound283769.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283788.bound, LeftBound283769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283788.actual selector witness, LeftBound283769.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283792

namespace LeftBound283805
def owner : Owner := ⟨.program ⟨257⟩, ⟨30535⟩⟩
def transferEvent : Nat := 283805
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 283803 .coefficient, .predecessor 1 283804 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283803 .coefficient)
      LeftBound283628.bound (LeftBound283628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283804 .coefficient)
      LeftBound283611.bound (LeftBound283611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1107.exact283618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283628.bound, LeftBound283611.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283628.bound, LeftBound283611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283628.actual selector witness, LeftBound283611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283805

namespace LeftBound283808
def owner : Owner := ⟨.program ⟨257⟩, ⟨30535⟩⟩
def transferEvent : Nat := 283808
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 283802 .summary, .result 283618 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283802 .summary)
      LeftBound283630.bound (LeftBound283630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29472⟩⟩) (rawTerms := some (Proof.Events1108.exact283802RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound283630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283618 .summary)
      LeftBound283613.bound (LeftBound283613.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30534⟩⟩) (rawTerms := some (Proof.Events1107.exact283618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound283613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound283630.bound, LeftBound283613.bound]
def bound : CoeffClass := .finite ⟨2998127310542407467008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283630.bound, LeftBound283613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound283630.actual selector witness, LeftBound283613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound283808

namespace LeftBound283812
def owner : Owner := ⟨.program ⟨257⟩, ⟨30821⟩⟩
def transferEvent : Nat := 283812
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 283810 .coefficient) (.predecessor 1 283811 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 283810 .coefficient)
      LeftBound283805.bound (LeftBound283805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1108.exact283809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 283811 .coefficient)
      LeftAuthority283533.bound (LeftAuthority283533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1107.exact283534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283805.bound LeftAuthority283533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283805.bound, LeftAuthority283533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283805.actual selector witness) * (LeftAuthority283533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283812

namespace LeftBound283813
def owner : Owner := ⟨.program ⟨257⟩, ⟨30821⟩⟩
def transferEvent : Nat := 283813
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨30819⟩⟩]⟩ [⟨.result 283534 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283534 .coefficient)
      LeftAuthority283533.bound (LeftAuthority283533.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨30819⟩⟩) (rawTerms := some (Proof.Events1107.exact283534RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority283533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority283533.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority283533.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority283533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority283533.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound283813

namespace LeftBound283814
def owner : Owner := ⟨.program ⟨257⟩, ⟨30821⟩⟩
def transferEvent : Nat := 283814
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 283809 .summary) (.transfer 283813) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283809 .summary)
      LeftBound283808.bound (LeftBound283808.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30535⟩⟩) (rawTerms := some (Proof.Events1108.exact283809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound283808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 283813)
      LeftBound283813.bound (LeftBound283813.actual selector witness) := by
  exact .transfer (LeftBound283813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound283808.bound LeftBound283813.bound
def bound : CoeffClass := .finite ⟨32192146870060190229763897425920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound283808.bound, LeftBound283813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound283808.actual selector witness) * (LeftBound283813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound283814

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
