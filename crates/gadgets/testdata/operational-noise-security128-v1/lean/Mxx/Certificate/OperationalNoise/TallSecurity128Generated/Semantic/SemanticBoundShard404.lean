import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard403

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64832
def owner : Owner := ⟨.program ⟨257⟩, ⟨26264⟩⟩
def transferEvent : Nat := 64832
def frameStart : Nat := 64799
def rule : BoundRule := .identity (.predecessor 0 64831 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64831 .coefficient)
      LeftBound64828.bound (LeftBound64828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64828.derived selector witness)

def rawBound : CoeffClass := LeftBound64828.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound64828.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64832

namespace LeftBound64849
def owner : Owner := ⟨.program ⟨257⟩, ⟨27714⟩⟩
def transferEvent : Nat := 64849
def frameStart : Nat := 64799
def rule : BoundRule := .sum [.predecessor 0 64847 .coefficient, .predecessor 1 64848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64847 .coefficient)
      LeftBound64832.bound (LeftBound64832.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64848 .coefficient)
      LeftAuthority64845.bound (LeftAuthority64845.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64832.bound, LeftAuthority64845.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64832.bound, LeftAuthority64845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64832.actual selector witness, LeftAuthority64845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64849

namespace LeftBound64852
def owner : Owner := ⟨.program ⟨257⟩, ⟨27715⟩⟩
def transferEvent : Nat := 64852
def frameStart : Nat := 64799
def rule : BoundRule := .identity (.predecessor 0 64851 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64851 .coefficient)
      LeftBound64849.bound (LeftBound64849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64849.derived selector witness)

def rawBound : CoeffClass := LeftBound64849.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64849.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound64849.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64852

namespace LeftBound64858
def owner : Owner := ⟨.program ⟨257⟩, ⟨27716⟩⟩
def transferEvent : Nat := 64858
def frameStart : Nat := 64799
def rule : BoundRule := .product (.predecessor 0 64856 .coefficient) (.predecessor 1 64857 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64856 .coefficient)
      LeftAuthority64854.bound (LeftAuthority64854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64857 .coefficient)
      LeftBound64852.bound (LeftBound64852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64852.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority64854.bound LeftBound64852.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64854.bound, LeftBound64852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority64854.actual selector witness) * (LeftBound64852.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64858

namespace LeftBound64874
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 64874
def frameStart : Nat := 64799
def rule : BoundRule := .scale (.predecessor 0 64872 .coefficient) (.value (.predecessor 1 64873 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64872 .coefficient)
      LeftAuthority64870.bound (LeftAuthority64870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64873 .coefficient)
      LeftAuthority64861.bound (LeftAuthority64861.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64861.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority64870.bound LeftAuthority64861.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64870.bound, LeftAuthority64861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority64870.actual selector witness) * (LeftAuthority64861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64874

namespace LeftBound64877
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 64877
def frameStart : Nat := 64799
def rule : BoundRule := .identity (.predecessor 0 64876 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64876 .coefficient)
      LeftAuthority64864.bound (LeftAuthority64864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64864.derived selector witness)

def rawBound : CoeffClass := LeftAuthority64864.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority64864.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64877

namespace LeftBound64881
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 64881
def frameStart : Nat := 64799
def rule : BoundRule := .product (.predecessor 0 64879 .coefficient) (.predecessor 1 64880 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64879 .coefficient)
      LeftBound64877.bound (LeftBound64877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64880 .coefficient)
      LeftBound64874.bound (LeftBound64874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64874.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64877.bound LeftBound64874.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64877.bound, LeftBound64874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64877.actual selector witness) * (LeftBound64874.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64881

namespace LeftBound64886
def owner : Owner := ⟨.program ⟨257⟩, ⟨27717⟩⟩
def transferEvent : Nat := 64886
def frameStart : Nat := 64799
def rule : BoundRule := .sum [.predecessor 0 64884 .coefficient, .predecessor 1 64885 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64884 .coefficient)
      LeftBound64881.bound (LeftBound64881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64885 .coefficient)
      LeftBound64858.bound (LeftBound64858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64881.bound, LeftBound64858.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64881.bound, LeftBound64858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64881.actual selector witness, LeftBound64858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64886

namespace LeftBound64890
def owner : Owner := ⟨.program ⟨257⟩, ⟨27999⟩⟩
def transferEvent : Nat := 64890
def frameStart : Nat := 64799
def rule : BoundRule := .product (.predecessor 0 64888 .coefficient) (.predecessor 1 64889 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64888 .coefficient)
      LeftBound64886.bound (LeftBound64886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64886.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64889 .coefficient)
      LeftAuthority64843.bound (LeftAuthority64843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64886.bound LeftAuthority64843.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64886.bound, LeftAuthority64843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64886.actual selector witness) * (LeftAuthority64843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64890

namespace LeftBound64901
def owner : Owner := ⟨.program ⟨257⟩, ⟨26466⟩⟩
def transferEvent : Nat := 64901
def frameStart : Nat := 64799
def rule : BoundRule := .product (.predecessor 0 64899 .coefficient) (.predecessor 1 64900 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64899 .coefficient)
      LeftAuthority64854.bound (LeftAuthority64854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64900 .coefficient)
      LeftAuthority64897.bound (LeftAuthority64897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64854.bound LeftAuthority64897.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64854.bound, LeftAuthority64897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority64854.actual selector witness) * (LeftAuthority64897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64901

namespace LeftBound64909
def owner : Owner := ⟨.program ⟨257⟩, ⟨26467⟩⟩
def transferEvent : Nat := 64909
def frameStart : Nat := 64799
def rule : BoundRule := .sum [.predecessor 0 64907 .coefficient, .predecessor 1 64908 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64907 .coefficient)
      LeftAuthority64905.bound (LeftAuthority64905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64908 .coefficient)
      LeftBound64901.bound (LeftBound64901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64901.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64905.bound, LeftBound64901.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64905.bound, LeftBound64901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority64905.actual selector witness, LeftBound64901.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64909

namespace LeftBound64913
def owner : Owner := ⟨.program ⟨257⟩, ⟨28000⟩⟩
def transferEvent : Nat := 64913
def frameStart : Nat := 64799
def rule : BoundRule := .sum [.predecessor 0 64911 .coefficient, .predecessor 1 64912 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64911 .coefficient)
      LeftBound64909.bound (LeftBound64909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64912 .coefficient)
      LeftBound64890.bound (LeftBound64890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64909.bound, LeftBound64890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64909.bound, LeftBound64890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64909.actual selector witness, LeftBound64890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64913

namespace LeftBound64926
def owner : Owner := ⟨.program ⟨257⟩, ⟨27998⟩⟩
def transferEvent : Nat := 64926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64924 .coefficient, .predecessor 1 64925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64924 .coefficient)
      LeftBound64747.bound (LeftBound64747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64925 .coefficient)
      LeftBound64730.bound (LeftBound64730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64747.bound, LeftBound64730.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64747.bound, LeftBound64730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64747.actual selector witness, LeftBound64730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64926

namespace LeftBound64929
def owner : Owner := ⟨.program ⟨257⟩, ⟨27998⟩⟩
def transferEvent : Nat := 64929
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64923 .summary, .result 64737 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64923 .summary)
      LeftBound64749.bound (LeftBound64749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26922⟩⟩) (rawTerms := some (Proof.Events253.exact64923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64737 .summary)
      LeftBound64732.bound (LeftBound64732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27997⟩⟩) (rawTerms := some (Proof.Events252.exact64737RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64749.bound, LeftBound64732.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64749.bound, LeftBound64732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound64749.actual selector witness, LeftBound64732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64929

namespace LeftBound64933
def owner : Owner := ⟨.program ⟨257⟩, ⟨28466⟩⟩
def transferEvent : Nat := 64933
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64931 .coefficient) (.predecessor 1 64932 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 64931 .coefficient)
      LeftBound64926.bound (LeftBound64926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 64932 .coefficient)
      LeftAuthority64652.bound (LeftAuthority64652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64926.bound LeftAuthority64652.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64926.bound, LeftAuthority64652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64926.actual selector witness) * (LeftAuthority64652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64933

namespace LeftBound64934
def owner : Owner := ⟨.program ⟨257⟩, ⟨28466⟩⟩
def transferEvent : Nat := 64934
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28464⟩⟩]⟩ [⟨.result 64653 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64653 .coefficient)
      LeftAuthority64652.bound (LeftAuthority64652.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28464⟩⟩) (rawTerms := some (Proof.Events252.exact64653RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64652.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64652.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority64652.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64934

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
