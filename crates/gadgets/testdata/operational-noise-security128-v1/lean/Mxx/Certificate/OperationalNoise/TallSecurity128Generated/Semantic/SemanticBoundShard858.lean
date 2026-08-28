import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard799
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard857

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound130852
def owner : Owner := ⟨.program ⟨257⟩, ⟨41450⟩⟩
def transferEvent : Nat := 130852
def frameStart : Nat := 130796
def rule : BoundRule := .sum [.predecessor 0 130850 .coefficient, .predecessor 1 130851 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130850 .coefficient)
      LeftBound130835.bound (LeftBound130835.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound130835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130851 .coefficient)
      LeftAuthority130848.bound (LeftAuthority130848.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority130848.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130835.bound, LeftAuthority130848.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130835.bound, LeftAuthority130848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130835.actual selector witness, LeftAuthority130848.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130852

namespace LeftBound130855
def owner : Owner := ⟨.program ⟨257⟩, ⟨41451⟩⟩
def transferEvent : Nat := 130855
def frameStart : Nat := 130796
def rule : BoundRule := .identity (.predecessor 0 130854 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130854 .coefficient)
      LeftBound130852.bound (LeftBound130852.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound130852.derived selector witness)

def rawBound : CoeffClass := LeftBound130852.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound130852.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound130855

namespace LeftBound130861
def owner : Owner := ⟨.program ⟨257⟩, ⟨41452⟩⟩
def transferEvent : Nat := 130861
def frameStart : Nat := 130796
def rule : BoundRule := .product (.predecessor 0 130859 .coefficient) (.predecessor 1 130860 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130859 .coefficient)
      LeftAuthority130857.bound (LeftAuthority130857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130860 .coefficient)
      LeftBound130855.bound (LeftBound130855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority130857.bound LeftBound130855.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130857.bound, LeftBound130855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority130857.actual selector witness) * (LeftBound130855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130861

namespace LeftBound130869
def owner : Owner := ⟨.program ⟨257⟩, ⟨41453⟩⟩
def transferEvent : Nat := 130869
def frameStart : Nat := 130796
def rule : BoundRule := .sum [.predecessor 0 130867 .coefficient, .predecessor 1 130868 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130867 .coefficient)
      LeftAuthority130865.bound (LeftAuthority130865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130865.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130865.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130868 .coefficient)
      LeftBound130861.bound (LeftBound130861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130861.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority130865.bound, LeftBound130861.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130865.bound, LeftBound130861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority130865.actual selector witness, LeftBound130861.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130869

namespace LeftBound130873
def owner : Owner := ⟨.program ⟨257⟩, ⟨41884⟩⟩
def transferEvent : Nat := 130873
def frameStart : Nat := 130796
def rule : BoundRule := .product (.predecessor 0 130871 .coefficient) (.predecessor 1 130872 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130871 .coefficient)
      LeftBound130869.bound (LeftBound130869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130872 .coefficient)
      LeftAuthority130846.bound (LeftAuthority130846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130846.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound130869.bound LeftAuthority130846.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130869.bound, LeftAuthority130846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound130869.actual selector witness) * (LeftAuthority130846.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130873

namespace LeftBound130884
def owner : Owner := ⟨.program ⟨257⟩, ⟨40272⟩⟩
def transferEvent : Nat := 130884
def frameStart : Nat := 130796
def rule : BoundRule := .product (.predecessor 0 130882 .coefficient) (.predecessor 1 130883 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130882 .coefficient)
      LeftAuthority130857.bound (LeftAuthority130857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130883 .coefficient)
      LeftAuthority130880.bound (LeftAuthority130880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority130857.bound LeftAuthority130880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130857.bound, LeftAuthority130880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority130857.actual selector witness) * (LeftAuthority130880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130884

namespace LeftBound130892
def owner : Owner := ⟨.program ⟨257⟩, ⟨40273⟩⟩
def transferEvent : Nat := 130892
def frameStart : Nat := 130796
def rule : BoundRule := .sum [.predecessor 0 130890 .coefficient, .predecessor 1 130891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130890 .coefficient)
      LeftAuthority130888.bound (LeftAuthority130888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130891 .coefficient)
      LeftBound130884.bound (LeftBound130884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority130888.bound, LeftBound130884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130888.bound, LeftBound130884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority130888.actual selector witness, LeftBound130884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130892

namespace LeftBound130896
def owner : Owner := ⟨.program ⟨257⟩, ⟨41888⟩⟩
def transferEvent : Nat := 130896
def frameStart : Nat := 130796
def rule : BoundRule := .sum [.predecessor 0 130894 .coefficient, .predecessor 1 130895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130894 .coefficient)
      LeftBound130892.bound (LeftBound130892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130895 .coefficient)
      LeftBound130873.bound (LeftBound130873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130892.bound, LeftBound130873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130892.bound, LeftBound130873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130892.actual selector witness, LeftBound130873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130896

namespace LeftBound130909
def owner : Owner := ⟨.program ⟨257⟩, ⟨41886⟩⟩
def transferEvent : Nat := 130909
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 130907 .coefficient, .predecessor 1 130908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130907 .coefficient)
      LeftBound130738.bound (LeftBound130738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130738.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130908 .coefficient)
      LeftBound130721.bound (LeftBound130721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events510.exact130728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130738.bound, LeftBound130721.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130738.bound, LeftBound130721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130738.actual selector witness, LeftBound130721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130909

namespace LeftBound130912
def owner : Owner := ⟨.program ⟨257⟩, ⟨41886⟩⟩
def transferEvent : Nat := 130912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 130906 .summary, .result 130728 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130906 .summary)
      LeftBound130740.bound (LeftBound130740.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40775⟩⟩) (rawTerms := some (Proof.Events511.exact130906RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130740.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130728 .summary)
      LeftBound130723.bound (LeftBound130723.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41885⟩⟩) (rawTerms := some (Proof.Events510.exact130728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130723.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130740.bound, LeftBound130723.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130740.bound, LeftBound130723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130740.actual selector witness, LeftBound130723.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130912

namespace LeftBound130916
def owner : Owner := ⟨.program ⟨257⟩, ⟨41887⟩⟩
def transferEvent : Nat := 130916
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 130914 .coefficient) (.predecessor 1 130915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130914 .coefficient)
      LeftBound130909.bound (LeftBound130909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130915 .coefficient)
      LeftBound15601.bound (LeftBound15601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound130909.bound LeftBound15601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130909.bound, LeftBound15601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound130909.actual selector witness) * (LeftBound15601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130916

namespace LeftBound130917
def owner : Owner := ⟨.program ⟨257⟩, ⟨41887⟩⟩
def transferEvent : Nat := 130917
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩ [⟨.result 15598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15598 .coefficient)
      LeftAuthority15597.bound (LeftAuthority15597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7159⟩⟩) (rawTerms := some (Proof.Events060.exact15598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound130917

namespace LeftBound130918
def owner : Owner := ⟨.program ⟨257⟩, ⟨41887⟩⟩
def transferEvent : Nat := 130918
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 130913 .summary) (.transfer 130917) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130913 .summary)
      LeftBound130912.bound (LeftBound130912.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41886⟩⟩) (rawTerms := some (Proof.Events511.exact130913RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 130917)
      LeftBound130917.bound (LeftBound130917.actual selector witness) := by
  exact .transfer (LeftBound130917.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound130912.bound LeftBound130917.bound
def bound : CoeffClass := .finite ⟨345671091840339265080175045977281837137920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130912.bound, LeftBound130917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound130912.actual selector witness) * (LeftBound130917.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130918

namespace LeftBound130933
def owner : Owner := ⟨.program ⟨257⟩, ⟨39205⟩⟩
def transferEvent : Nat := 130933
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 130931 .coefficient) (.predecessor 1 130932 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130931 .coefficient)
      LeftBound121980.bound (LeftBound121980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events476.exact121984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound121980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound121980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130932 .coefficient)
      LeftAuthority130929.bound (LeftAuthority130929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events511.exact130930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121980.bound LeftAuthority130929.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121980.bound, LeftAuthority130929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121980.actual selector witness) * (LeftAuthority130929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130933

namespace LeftBound130934
def owner : Owner := ⟨.program ⟨257⟩, ⟨39205⟩⟩
def transferEvent : Nat := 130934
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39203⟩⟩]⟩ [⟨.result 130930 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130930 .coefficient)
      LeftAuthority130929.bound (LeftAuthority130929.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39203⟩⟩) (rawTerms := some (Proof.Events511.exact130930RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130929.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority130929.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority130929.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound130934

namespace LeftBound130935
def owner : Owner := ⟨.program ⟨257⟩, ⟨39205⟩⟩
def transferEvent : Nat := 130935
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 121984 .summary) (.transfer 130934) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 121984 .summary)
      LeftBound121983.bound (LeftBound121983.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38897⟩⟩) (rawTerms := some (Proof.Events476.exact121984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound121983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 130934)
      LeftBound130934.bound (LeftBound130934.actual selector witness) := by
  exact .transfer (LeftBound130934.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound121983.bound LeftBound130934.bound
def bound : CoeffClass := .finite ⟨32192736221397252361486566686720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound121983.bound, LeftBound130934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound121983.actual selector witness) * (LeftBound130934.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130935

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
