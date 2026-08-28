import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1497

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound222811
def owner : Owner := ⟨.program ⟨257⟩, ⟨45131⟩⟩
def transferEvent : Nat := 222811
def frameStart : Nat := 222782
def rule : BoundRule := .product (.predecessor 0 222809 .coefficient) (.predecessor 1 222810 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222809 .coefficient)
      LeftAuthority222807.bound (LeftAuthority222807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222810 .coefficient)
      LeftAuthority222804.bound (LeftAuthority222804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222804.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority222807.bound LeftAuthority222804.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222807.bound, LeftAuthority222804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority222807.actual selector witness) * (LeftAuthority222804.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222811

namespace LeftBound222815
def owner : Owner := ⟨.program ⟨257⟩, ⟨45132⟩⟩
def transferEvent : Nat := 222815
def frameStart : Nat := 222782
def rule : BoundRule := .identity (.predecessor 0 222814 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222814 .coefficient)
      LeftBound222811.bound (LeftBound222811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222811.derived selector witness)

def rawBound : CoeffClass := LeftBound222811.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound222811.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound222815

namespace LeftBound222832
def owner : Owner := ⟨.program ⟨257⟩, ⟨46742⟩⟩
def transferEvent : Nat := 222832
def frameStart : Nat := 222782
def rule : BoundRule := .sum [.predecessor 0 222830 .coefficient, .predecessor 1 222831 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222830 .coefficient)
      LeftBound222815.bound (LeftBound222815.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound222815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222831 .coefficient)
      LeftAuthority222828.bound (LeftAuthority222828.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority222828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound222815.bound, LeftAuthority222828.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222815.bound, LeftAuthority222828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound222815.actual selector witness, LeftAuthority222828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222832

namespace LeftBound222835
def owner : Owner := ⟨.program ⟨257⟩, ⟨46743⟩⟩
def transferEvent : Nat := 222835
def frameStart : Nat := 222782
def rule : BoundRule := .identity (.predecessor 0 222834 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222834 .coefficient)
      LeftBound222832.bound (LeftBound222832.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound222832.derived selector witness)

def rawBound : CoeffClass := LeftBound222832.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound222832.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound222835

namespace LeftBound222841
def owner : Owner := ⟨.program ⟨257⟩, ⟨46744⟩⟩
def transferEvent : Nat := 222841
def frameStart : Nat := 222782
def rule : BoundRule := .product (.predecessor 0 222839 .coefficient) (.predecessor 1 222840 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222839 .coefficient)
      LeftAuthority222837.bound (LeftAuthority222837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222840 .coefficient)
      LeftBound222835.bound (LeftBound222835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222835.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority222837.bound LeftBound222835.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222837.bound, LeftBound222835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority222837.actual selector witness) * (LeftBound222835.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222841

namespace LeftBound222857
def owner : Owner := ⟨.program ⟨257⟩, ⟨9563⟩⟩
def transferEvent : Nat := 222857
def frameStart : Nat := 222782
def rule : BoundRule := .scale (.predecessor 0 222855 .coefficient) (.value (.predecessor 1 222856 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222855 .coefficient)
      LeftAuthority222853.bound (LeftAuthority222853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222856 .coefficient)
      LeftAuthority222844.bound (LeftAuthority222844.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority222844.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority222853.bound LeftAuthority222844.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222853.bound, LeftAuthority222844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority222853.actual selector witness) * (LeftAuthority222844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound222857

namespace LeftBound222860
def owner : Owner := ⟨.program ⟨257⟩, ⟨7301⟩⟩
def transferEvent : Nat := 222860
def frameStart : Nat := 222782
def rule : BoundRule := .identity (.predecessor 0 222859 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222859 .coefficient)
      LeftAuthority222847.bound (LeftAuthority222847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222847.derived selector witness)

def rawBound : CoeffClass := LeftAuthority222847.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority222847.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound222860

namespace LeftBound222864
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def transferEvent : Nat := 222864
def frameStart : Nat := 222782
def rule : BoundRule := .product (.predecessor 0 222862 .coefficient) (.predecessor 1 222863 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222862 .coefficient)
      LeftBound222860.bound (LeftBound222860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222863 .coefficient)
      LeftBound222857.bound (LeftBound222857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222857.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound222860.bound LeftBound222857.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222860.bound, LeftBound222857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound222860.actual selector witness) * (LeftBound222857.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222864

namespace LeftBound222869
def owner : Owner := ⟨.program ⟨257⟩, ⟨46745⟩⟩
def transferEvent : Nat := 222869
def frameStart : Nat := 222782
def rule : BoundRule := .sum [.predecessor 0 222867 .coefficient, .predecessor 1 222868 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222867 .coefficient)
      LeftBound222864.bound (LeftBound222864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222868 .coefficient)
      LeftBound222841.bound (LeftBound222841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222841.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound222864.bound, LeftBound222841.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222864.bound, LeftBound222841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound222864.actual selector witness, LeftBound222841.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222869

namespace LeftBound222873
def owner : Owner := ⟨.program ⟨257⟩, ⟨46971⟩⟩
def transferEvent : Nat := 222873
def frameStart : Nat := 222782
def rule : BoundRule := .product (.predecessor 0 222871 .coefficient) (.predecessor 1 222872 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222871 .coefficient)
      LeftBound222869.bound (LeftBound222869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222872 .coefficient)
      LeftAuthority222826.bound (LeftAuthority222826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222826.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound222869.bound LeftAuthority222826.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222869.bound, LeftAuthority222826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound222869.actual selector witness) * (LeftAuthority222826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222873

namespace LeftBound222884
def owner : Owner := ⟨.program ⟨257⟩, ⟨45462⟩⟩
def transferEvent : Nat := 222884
def frameStart : Nat := 222782
def rule : BoundRule := .product (.predecessor 0 222882 .coefficient) (.predecessor 1 222883 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222882 .coefficient)
      LeftAuthority222837.bound (LeftAuthority222837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222883 .coefficient)
      LeftAuthority222880.bound (LeftAuthority222880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority222837.bound LeftAuthority222880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222837.bound, LeftAuthority222880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority222837.actual selector witness) * (LeftAuthority222880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222884

namespace LeftBound222892
def owner : Owner := ⟨.program ⟨257⟩, ⟨45463⟩⟩
def transferEvent : Nat := 222892
def frameStart : Nat := 222782
def rule : BoundRule := .sum [.predecessor 0 222890 .coefficient, .predecessor 1 222891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222890 .coefficient)
      LeftAuthority222888.bound (LeftAuthority222888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222891 .coefficient)
      LeftBound222884.bound (LeftBound222884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority222888.bound, LeftBound222884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority222888.bound, LeftBound222884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority222888.actual selector witness, LeftBound222884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222892

namespace LeftBound222896
def owner : Owner := ⟨.program ⟨257⟩, ⟨46972⟩⟩
def transferEvent : Nat := 222896
def frameStart : Nat := 222782
def rule : BoundRule := .sum [.predecessor 0 222894 .coefficient, .predecessor 1 222895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222894 .coefficient)
      LeftBound222892.bound (LeftBound222892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222895 .coefficient)
      LeftBound222873.bound (LeftBound222873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound222892.bound, LeftBound222873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222892.bound, LeftBound222873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound222892.actual selector witness, LeftBound222873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222896

namespace LeftBound222909
def owner : Owner := ⟨.program ⟨257⟩, ⟨46970⟩⟩
def transferEvent : Nat := 222909
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 222907 .coefficient, .predecessor 1 222908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222907 .coefficient)
      LeftBound222730.bound (LeftBound222730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222908 .coefficient)
      LeftBound222713.bound (LeftBound222713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound222730.bound, LeftBound222713.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222730.bound, LeftBound222713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound222730.actual selector witness, LeftBound222713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222909

namespace LeftBound222912
def owner : Owner := ⟨.program ⟨257⟩, ⟨46970⟩⟩
def transferEvent : Nat := 222912
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 222906 .summary, .result 222720 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222906 .summary)
      LeftBound222732.bound (LeftBound222732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45902⟩⟩) (rawTerms := some (Proof.Events870.exact222906RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222720 .summary)
      LeftBound222715.bound (LeftBound222715.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46969⟩⟩) (rawTerms := some (Proof.Events870.exact222720RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound222732.bound, LeftBound222715.bound]
def bound : CoeffClass := .finite ⟨2998328565150755586048, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222732.bound, LeftBound222715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound222732.actual selector witness, LeftBound222715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound222912

namespace LeftBound222916
def owner : Owner := ⟨.program ⟨257⟩, ⟨47326⟩⟩
def transferEvent : Nat := 222916
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 222914 .coefficient) (.predecessor 1 222915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 222914 .coefficient)
      LeftBound222909.bound (LeftBound222909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events870.exact222913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 222915 .coefficient)
      LeftAuthority222635.bound (LeftAuthority222635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events869.exact222636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority222635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority222635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound222909.bound LeftAuthority222635.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222909.bound, LeftAuthority222635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound222909.actual selector witness) * (LeftAuthority222635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound222916

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
