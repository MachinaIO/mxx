import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1037

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound155956
def owner : Owner := ⟨.program ⟨257⟩, ⟨31406⟩⟩
def transferEvent : Nat := 155956
def frameStart : Nat := 155923
def rule : BoundRule := .identity (.predecessor 0 155955 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155955 .coefficient)
      LeftBound155952.bound (LeftBound155952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155952.derived selector witness)

def rawBound : CoeffClass := LeftBound155952.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound155952.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound155956

namespace LeftBound155973
def owner : Owner := ⟨.program ⟨257⟩, ⟨33214⟩⟩
def transferEvent : Nat := 155973
def frameStart : Nat := 155923
def rule : BoundRule := .sum [.predecessor 0 155971 .coefficient, .predecessor 1 155972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155971 .coefficient)
      LeftBound155956.bound (LeftBound155956.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound155956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155972 .coefficient)
      LeftAuthority155969.bound (LeftAuthority155969.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority155969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound155956.bound, LeftAuthority155969.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155956.bound, LeftAuthority155969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound155956.actual selector witness, LeftAuthority155969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound155973

namespace LeftBound155976
def owner : Owner := ⟨.program ⟨257⟩, ⟨33215⟩⟩
def transferEvent : Nat := 155976
def frameStart : Nat := 155923
def rule : BoundRule := .identity (.predecessor 0 155975 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155975 .coefficient)
      LeftBound155973.bound (LeftBound155973.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound155973.derived selector witness)

def rawBound : CoeffClass := LeftBound155973.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound155973.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound155976

namespace LeftBound155982
def owner : Owner := ⟨.program ⟨257⟩, ⟨33216⟩⟩
def transferEvent : Nat := 155982
def frameStart : Nat := 155923
def rule : BoundRule := .product (.predecessor 0 155980 .coefficient) (.predecessor 1 155981 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155980 .coefficient)
      LeftAuthority155978.bound (LeftAuthority155978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155981 .coefficient)
      LeftBound155976.bound (LeftBound155976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority155978.bound LeftBound155976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155978.bound, LeftBound155976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority155978.actual selector witness) * (LeftBound155976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound155982

namespace LeftBound155998
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 155998
def frameStart : Nat := 155923
def rule : BoundRule := .scale (.predecessor 0 155996 .coefficient) (.value (.predecessor 1 155997 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 155996 .coefficient)
      LeftAuthority155994.bound (LeftAuthority155994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 155997 .coefficient)
      LeftAuthority155985.bound (LeftAuthority155985.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority155985.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority155994.bound LeftAuthority155985.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155994.bound, LeftAuthority155985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority155994.actual selector witness) * (LeftAuthority155985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound155998

namespace LeftBound156001
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 156001
def frameStart : Nat := 155923
def rule : BoundRule := .identity (.predecessor 0 156000 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156000 .coefficient)
      LeftAuthority155988.bound (LeftAuthority155988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155988.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155988.derived selector witness)

def rawBound : CoeffClass := LeftAuthority155988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority155988.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound156001

namespace LeftBound156005
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 156005
def frameStart : Nat := 155923
def rule : BoundRule := .product (.predecessor 0 156003 .coefficient) (.predecessor 1 156004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156003 .coefficient)
      LeftBound156001.bound (LeftBound156001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156004 .coefficient)
      LeftBound155998.bound (LeftBound155998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound156001.bound LeftBound155998.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156001.bound, LeftBound155998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound156001.actual selector witness) * (LeftBound155998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156005

namespace LeftBound156010
def owner : Owner := ⟨.program ⟨257⟩, ⟨33217⟩⟩
def transferEvent : Nat := 156010
def frameStart : Nat := 155923
def rule : BoundRule := .sum [.predecessor 0 156008 .coefficient, .predecessor 1 156009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156008 .coefficient)
      LeftBound156005.bound (LeftBound156005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156009 .coefficient)
      LeftBound155982.bound (LeftBound155982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156005.bound, LeftBound155982.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156005.bound, LeftBound155982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156005.actual selector witness, LeftBound155982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156010

namespace LeftBound156014
def owner : Owner := ⟨.program ⟨257⟩, ⟨33429⟩⟩
def transferEvent : Nat := 156014
def frameStart : Nat := 155923
def rule : BoundRule := .product (.predecessor 0 156012 .coefficient) (.predecessor 1 156013 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156012 .coefficient)
      LeftBound156010.bound (LeftBound156010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156013 .coefficient)
      LeftAuthority155967.bound (LeftAuthority155967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound156010.bound LeftAuthority155967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156010.bound, LeftAuthority155967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound156010.actual selector witness) * (LeftAuthority155967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156014

namespace LeftBound156025
def owner : Owner := ⟨.program ⟨257⟩, ⟨31806⟩⟩
def transferEvent : Nat := 156025
def frameStart : Nat := 155923
def rule : BoundRule := .product (.predecessor 0 156023 .coefficient) (.predecessor 1 156024 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156023 .coefficient)
      LeftAuthority155978.bound (LeftAuthority155978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact155979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156024 .coefficient)
      LeftAuthority156021.bound (LeftAuthority156021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority155978.bound LeftAuthority156021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155978.bound, LeftAuthority156021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority155978.actual selector witness) * (LeftAuthority156021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156025

namespace LeftBound156033
def owner : Owner := ⟨.program ⟨257⟩, ⟨31807⟩⟩
def transferEvent : Nat := 156033
def frameStart : Nat := 155923
def rule : BoundRule := .sum [.predecessor 0 156031 .coefficient, .predecessor 1 156032 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156031 .coefficient)
      LeftAuthority156029.bound (LeftAuthority156029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156032 .coefficient)
      LeftBound156025.bound (LeftBound156025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156025.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority156029.bound, LeftBound156025.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority156029.bound, LeftBound156025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority156029.actual selector witness, LeftBound156025.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156033

namespace LeftBound156037
def owner : Owner := ⟨.program ⟨257⟩, ⟨33430⟩⟩
def transferEvent : Nat := 156037
def frameStart : Nat := 155923
def rule : BoundRule := .sum [.predecessor 0 156035 .coefficient, .predecessor 1 156036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156035 .coefficient)
      LeftBound156033.bound (LeftBound156033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156036 .coefficient)
      LeftBound156014.bound (LeftBound156014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156033.bound, LeftBound156014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156033.bound, LeftBound156014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156033.actual selector witness, LeftBound156014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156037

namespace LeftBound156050
def owner : Owner := ⟨.program ⟨257⟩, ⟨33428⟩⟩
def transferEvent : Nat := 156050
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 156048 .coefficient, .predecessor 1 156049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156048 .coefficient)
      LeftBound155871.bound (LeftBound155871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156049 .coefficient)
      LeftBound155854.bound (LeftBound155854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events608.exact155861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155854.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound155871.bound, LeftBound155854.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155871.bound, LeftBound155854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound155871.actual selector witness, LeftBound155854.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156050

namespace LeftBound156053
def owner : Owner := ⟨.program ⟨257⟩, ⟨33428⟩⟩
def transferEvent : Nat := 156053
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 156047 .summary, .result 155861 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 156047 .summary)
      LeftBound155873.bound (LeftBound155873.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32362⟩⟩) (rawTerms := some (Proof.Events609.exact156047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155861 .summary)
      LeftBound155856.bound (LeftBound155856.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33427⟩⟩) (rawTerms := some (Proof.Events608.exact155861RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound155873.bound, LeftBound155856.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound155873.bound, LeftBound155856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound155873.actual selector witness, LeftBound155856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156053

namespace LeftBound156057
def owner : Owner := ⟨.program ⟨257⟩, ⟨33801⟩⟩
def transferEvent : Nat := 156057
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 156055 .coefficient) (.predecessor 1 156056 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156055 .coefficient)
      LeftBound156050.bound (LeftBound156050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events609.exact156054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156056 .coefficient)
      LeftAuthority155776.bound (LeftAuthority155776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events608.exact155777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155776.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound156050.bound LeftAuthority155776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156050.bound, LeftAuthority155776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound156050.actual selector witness) * (LeftAuthority155776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156057

namespace LeftBound156058
def owner : Owner := ⟨.program ⟨257⟩, ⟨33801⟩⟩
def transferEvent : Nat := 156058
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33799⟩⟩]⟩ [⟨.result 155777 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155777 .coefficient)
      LeftAuthority155776.bound (LeftAuthority155776.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33799⟩⟩) (rawTerms := some (Proof.Events608.exact155777RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority155776.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority155776.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority155776.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority155776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority155776.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound156058

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
