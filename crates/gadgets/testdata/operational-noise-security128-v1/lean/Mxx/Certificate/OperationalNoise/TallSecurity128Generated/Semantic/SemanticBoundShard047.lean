import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14842
def owner : Owner := ⟨.program ⟨257⟩, ⟨26493⟩⟩
def transferEvent : Nat := 14842
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14840 .coefficient) (.predecessor 1 14841 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14840 .coefficient)
      LeftAuthority14838.bound (LeftAuthority14838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14841 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14838.bound LeftAuthority612.bound
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14838.bound, LeftAuthority612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14838.actual selector witness) * (LeftAuthority612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14842

namespace LeftBound14850
def owner : Owner := ⟨.program ⟨257⟩, ⟨65889⟩⟩
def transferEvent : Nat := 14850
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14848 .coefficient) (.predecessor 1 14849 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14848 .coefficient)
      LeftAuthority14846.bound (LeftAuthority14846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14849 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14846.bound LeftAuthority622.bound
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14846.bound, LeftAuthority622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14846.actual selector witness) * (LeftAuthority622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14850

namespace LeftBound14858
def owner : Owner := ⟨.program ⟨257⟩, ⟨62896⟩⟩
def transferEvent : Nat := 14858
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14856 .coefficient) (.predecessor 1 14857 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14856 .coefficient)
      LeftAuthority14854.bound (LeftAuthority14854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14857 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14854.bound LeftAuthority632.bound
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14854.bound, LeftAuthority632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14854.actual selector witness) * (LeftAuthority632.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14858

namespace LeftBound14866
def owner : Owner := ⟨.program ⟨257⟩, ⟨59916⟩⟩
def transferEvent : Nat := 14866
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14864 .coefficient) (.predecessor 1 14865 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14864 .coefficient)
      LeftAuthority14862.bound (LeftAuthority14862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14865 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14862.bound LeftAuthority642.bound
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14862.bound, LeftAuthority642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14862.actual selector witness) * (LeftAuthority642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14866

namespace LeftBound14874
def owner : Owner := ⟨.program ⟨257⟩, ⟨56936⟩⟩
def transferEvent : Nat := 14874
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14872 .coefficient) (.predecessor 1 14873 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14872 .coefficient)
      LeftAuthority14870.bound (LeftAuthority14870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14873 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14870.bound LeftAuthority652.bound
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14870.bound, LeftAuthority652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14870.actual selector witness) * (LeftAuthority652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14874

namespace LeftBound14882
def owner : Owner := ⟨.program ⟨257⟩, ⟨53956⟩⟩
def transferEvent : Nat := 14882
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14880 .coefficient) (.predecessor 1 14881 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14880 .coefficient)
      LeftAuthority14878.bound (LeftAuthority14878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14881 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14878.bound LeftAuthority662.bound
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14878.bound, LeftAuthority662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14878.actual selector witness) * (LeftAuthority662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14882

namespace LeftBound14890
def owner : Owner := ⟨.program ⟨257⟩, ⟨50976⟩⟩
def transferEvent : Nat := 14890
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14888 .coefficient) (.predecessor 1 14889 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14888 .coefficient)
      LeftAuthority14886.bound (LeftAuthority14886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14889 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14886.bound LeftAuthority672.bound
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14886.bound, LeftAuthority672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14886.actual selector witness) * (LeftAuthority672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14890

namespace LeftBound14898
def owner : Owner := ⟨.program ⟨257⟩, ⟨31912⟩⟩
def transferEvent : Nat := 14898
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14896 .coefficient) (.predecessor 1 14897 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14896 .coefficient)
      LeftAuthority14894.bound (LeftAuthority14894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14897 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14894.bound LeftAuthority682.bound
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14894.bound, LeftAuthority682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14894.actual selector witness) * (LeftAuthority682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14898

namespace LeftBound14906
def owner : Owner := ⟨.program ⟨257⟩, ⟨21892⟩⟩
def transferEvent : Nat := 14906
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14904 .coefficient) (.predecessor 1 14905 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14904 .coefficient)
      LeftAuthority14902.bound (LeftAuthority14902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14905 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14902.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14902.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14902.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14906

namespace LeftBound14914
def owner : Owner := ⟨.program ⟨257⟩, ⟨18672⟩⟩
def transferEvent : Nat := 14914
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14912 .coefficient) (.predecessor 1 14913 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14912 .coefficient)
      LeftAuthority14910.bound (LeftAuthority14910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14913 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14910.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14910.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14910.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14914

namespace LeftBound14922
def owner : Owner := ⟨.program ⟨257⟩, ⟨15871⟩⟩
def transferEvent : Nat := 14922
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14920 .coefficient) (.predecessor 1 14921 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14920 .coefficient)
      LeftAuthority14918.bound (LeftAuthority14918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14921 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14918.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14918.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority14918.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14922

namespace LeftBound14927
def owner : Owner := ⟨.program ⟨257⟩, ⟨15872⟩⟩
def transferEvent : Nat := 14927
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14925 .coefficient, .predecessor 1 14926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14925 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14926 .coefficient)
      LeftBound14922.bound (LeftBound14922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound14922.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound14922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound14922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14927

namespace LeftBound14931
def owner : Owner := ⟨.program ⟨257⟩, ⟨18673⟩⟩
def transferEvent : Nat := 14931
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14929 .coefficient, .predecessor 1 14930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14929 .coefficient)
      LeftBound14927.bound (LeftBound14927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14930 .coefficient)
      LeftBound14914.bound (LeftBound14914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14927.bound, LeftBound14914.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14927.bound, LeftBound14914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14927.actual selector witness, LeftBound14914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14931

namespace LeftBound14935
def owner : Owner := ⟨.program ⟨257⟩, ⟨21893⟩⟩
def transferEvent : Nat := 14935
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14933 .coefficient, .predecessor 1 14934 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14933 .coefficient)
      LeftBound14931.bound (LeftBound14931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14931.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14934 .coefficient)
      LeftBound14906.bound (LeftBound14906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14931.bound, LeftBound14906.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14931.bound, LeftBound14906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14931.actual selector witness, LeftBound14906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14935

namespace LeftBound14939
def owner : Owner := ⟨.program ⟨257⟩, ⟨31913⟩⟩
def transferEvent : Nat := 14939
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14937 .coefficient, .predecessor 1 14938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14937 .coefficient)
      LeftBound14935.bound (LeftBound14935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14938 .coefficient)
      LeftBound14898.bound (LeftBound14898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14935.bound, LeftBound14898.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14935.bound, LeftBound14898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14935.actual selector witness, LeftBound14898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14939

namespace LeftBound14943
def owner : Owner := ⟨.program ⟨257⟩, ⟨50977⟩⟩
def transferEvent : Nat := 14943
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14941 .coefficient, .predecessor 1 14942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 14941 .coefficient)
      LeftBound14939.bound (LeftBound14939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 14942 .coefficient)
      LeftBound14890.bound (LeftBound14890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact14892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14939.bound, LeftBound14890.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14939.bound, LeftBound14890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound14939.actual selector witness, LeftBound14890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14943

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
