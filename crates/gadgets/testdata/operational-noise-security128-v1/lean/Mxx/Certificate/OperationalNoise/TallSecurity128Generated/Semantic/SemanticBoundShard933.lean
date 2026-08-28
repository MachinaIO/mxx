import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard932

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound140866
def owner : Owner := ⟨.program ⟨257⟩, ⟨52258⟩⟩
def transferEvent : Nat := 140866
def frameStart : Nat := 140816
def rule : BoundRule := .sum [.predecessor 0 140864 .coefficient, .predecessor 1 140865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140864 .coefficient)
      LeftBound140849.bound (LeftBound140849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound140849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140865 .coefficient)
      LeftAuthority140862.bound (LeftAuthority140862.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority140862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound140849.bound, LeftAuthority140862.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140849.bound, LeftAuthority140862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound140849.actual selector witness, LeftAuthority140862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140866

namespace LeftBound140869
def owner : Owner := ⟨.program ⟨257⟩, ⟨52259⟩⟩
def transferEvent : Nat := 140869
def frameStart : Nat := 140816
def rule : BoundRule := .identity (.predecessor 0 140868 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140868 .coefficient)
      LeftBound140866.bound (LeftBound140866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound140866.derived selector witness)

def rawBound : CoeffClass := LeftBound140866.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound140866.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound140869

namespace LeftBound140875
def owner : Owner := ⟨.program ⟨257⟩, ⟨52260⟩⟩
def transferEvent : Nat := 140875
def frameStart : Nat := 140816
def rule : BoundRule := .product (.predecessor 0 140873 .coefficient) (.predecessor 1 140874 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140873 .coefficient)
      LeftAuthority140871.bound (LeftAuthority140871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140874 .coefficient)
      LeftBound140869.bound (LeftBound140869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority140871.bound LeftBound140869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140871.bound, LeftBound140869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority140871.actual selector witness) * (LeftBound140869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140875

namespace LeftBound140891
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 140891
def frameStart : Nat := 140816
def rule : BoundRule := .scale (.predecessor 0 140889 .coefficient) (.value (.predecessor 1 140890 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140889 .coefficient)
      LeftAuthority140887.bound (LeftAuthority140887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140890 .coefficient)
      LeftAuthority140878.bound (LeftAuthority140878.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority140878.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority140887.bound LeftAuthority140878.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140887.bound, LeftAuthority140878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority140887.actual selector witness) * (LeftAuthority140878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound140891

namespace LeftBound140894
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 140894
def frameStart : Nat := 140816
def rule : BoundRule := .identity (.predecessor 0 140893 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140893 .coefficient)
      LeftAuthority140881.bound (LeftAuthority140881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140881.derived selector witness)

def rawBound : CoeffClass := LeftAuthority140881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority140881.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound140894

namespace LeftBound140898
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 140898
def frameStart : Nat := 140816
def rule : BoundRule := .product (.predecessor 0 140896 .coefficient) (.predecessor 1 140897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140896 .coefficient)
      LeftBound140894.bound (LeftBound140894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140897 .coefficient)
      LeftBound140891.bound (LeftBound140891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound140894.bound LeftBound140891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140894.bound, LeftBound140891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound140894.actual selector witness) * (LeftBound140891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140898

namespace LeftBound140903
def owner : Owner := ⟨.program ⟨257⟩, ⟨52261⟩⟩
def transferEvent : Nat := 140903
def frameStart : Nat := 140816
def rule : BoundRule := .sum [.predecessor 0 140901 .coefficient, .predecessor 1 140902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140901 .coefficient)
      LeftBound140898.bound (LeftBound140898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140902 .coefficient)
      LeftBound140875.bound (LeftBound140875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound140898.bound, LeftBound140875.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140898.bound, LeftBound140875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound140898.actual selector witness, LeftBound140875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140903

namespace LeftBound140907
def owner : Owner := ⟨.program ⟨257⟩, ⟨52445⟩⟩
def transferEvent : Nat := 140907
def frameStart : Nat := 140816
def rule : BoundRule := .product (.predecessor 0 140905 .coefficient) (.predecessor 1 140906 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140905 .coefficient)
      LeftBound140903.bound (LeftBound140903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140906 .coefficient)
      LeftAuthority140860.bound (LeftAuthority140860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound140903.bound LeftAuthority140860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140903.bound, LeftAuthority140860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound140903.actual selector witness) * (LeftAuthority140860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140907

namespace LeftBound140918
def owner : Owner := ⟨.program ⟨257⟩, ⟨50834⟩⟩
def transferEvent : Nat := 140918
def frameStart : Nat := 140816
def rule : BoundRule := .product (.predecessor 0 140916 .coefficient) (.predecessor 1 140917 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140916 .coefficient)
      LeftAuthority140871.bound (LeftAuthority140871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140917 .coefficient)
      LeftAuthority140914.bound (LeftAuthority140914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority140871.bound LeftAuthority140914.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140871.bound, LeftAuthority140914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority140871.actual selector witness) * (LeftAuthority140914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140918

namespace LeftBound140926
def owner : Owner := ⟨.program ⟨257⟩, ⟨50835⟩⟩
def transferEvent : Nat := 140926
def frameStart : Nat := 140816
def rule : BoundRule := .sum [.predecessor 0 140924 .coefficient, .predecessor 1 140925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140924 .coefficient)
      LeftAuthority140922.bound (LeftAuthority140922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140925 .coefficient)
      LeftBound140918.bound (LeftBound140918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority140922.bound, LeftBound140918.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140922.bound, LeftBound140918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority140922.actual selector witness, LeftBound140918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140926

namespace LeftBound140930
def owner : Owner := ⟨.program ⟨257⟩, ⟨52446⟩⟩
def transferEvent : Nat := 140930
def frameStart : Nat := 140816
def rule : BoundRule := .sum [.predecessor 0 140928 .coefficient, .predecessor 1 140929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140928 .coefficient)
      LeftBound140926.bound (LeftBound140926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140929 .coefficient)
      LeftBound140907.bound (LeftBound140907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound140926.bound, LeftBound140907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140926.bound, LeftBound140907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound140926.actual selector witness, LeftBound140907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140930

namespace LeftBound140943
def owner : Owner := ⟨.program ⟨257⟩, ⟨52444⟩⟩
def transferEvent : Nat := 140943
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 140941 .coefficient, .predecessor 1 140942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140941 .coefficient)
      LeftBound140764.bound (LeftBound140764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140942 .coefficient)
      LeftBound140747.bound (LeftBound140747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events549.exact140754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140747.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound140764.bound, LeftBound140747.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140764.bound, LeftBound140747.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound140764.actual selector witness, LeftBound140747.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140943

namespace LeftBound140946
def owner : Owner := ⟨.program ⟨257⟩, ⟨52444⟩⟩
def transferEvent : Nat := 140946
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 140940 .summary, .result 140754 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140940 .summary)
      LeftBound140766.bound (LeftBound140766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51382⟩⟩) (rawTerms := some (Proof.Events550.exact140940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound140766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140754 .summary)
      LeftBound140749.bound (LeftBound140749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52443⟩⟩) (rawTerms := some (Proof.Events549.exact140754RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound140749.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound140766.bound, LeftBound140749.bound]
def bound : CoeffClass := .finite ⟨2997889464187086962688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140766.bound, LeftBound140749.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound140766.actual selector witness, LeftBound140749.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound140946

namespace LeftBound140950
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def transferEvent : Nat := 140950
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 140948 .coefficient) (.predecessor 1 140949 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 140948 .coefficient)
      LeftBound140943.bound (LeftBound140943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events550.exact140947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 140949 .coefficient)
      LeftAuthority140669.bound (LeftAuthority140669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events549.exact140670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound140943.bound LeftAuthority140669.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140943.bound, LeftAuthority140669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound140943.actual selector witness) * (LeftAuthority140669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140950

namespace LeftBound140951
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def transferEvent : Nat := 140951
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52735⟩⟩]⟩ [⟨.result 140670 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140670 .coefficient)
      LeftAuthority140669.bound (LeftAuthority140669.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52735⟩⟩) (rawTerms := some (Proof.Events549.exact140670RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority140669.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority140669.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority140669.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority140669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority140669.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound140951

namespace LeftBound140952
def owner : Owner := ⟨.program ⟨257⟩, ⟨52737⟩⟩
def transferEvent : Nat := 140952
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 140947 .summary) (.transfer 140951) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140947 .summary)
      LeftBound140946.bound (LeftBound140946.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52444⟩⟩) (rawTerms := some (Proof.Events550.exact140947RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound140946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 140951)
      LeftBound140951.bound (LeftBound140951.actual selector witness) := by
  exact .transfer (LeftBound140951.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound140946.bound LeftBound140951.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound140946.bound, LeftBound140951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound140946.actual selector witness) * (LeftBound140951.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound140952

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
