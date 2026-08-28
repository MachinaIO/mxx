import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard338

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55053
def owner : Owner := ⟨.program ⟨257⟩, ⟨17160⟩⟩
def transferEvent : Nat := 55053
def frameStart : Nat := 54994
def rule : BoundRule := .product (.predecessor 0 55051 .coefficient) (.predecessor 1 55052 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55051 .coefficient)
      LeftAuthority55049.bound (LeftAuthority55049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55052 .coefficient)
      LeftBound55047.bound (LeftBound55047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority55049.bound LeftBound55047.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55049.bound, LeftBound55047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority55049.actual selector witness) * (LeftBound55047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55053

namespace LeftBound55069
def owner : Owner := ⟨.program ⟨257⟩, ⟨9569⟩⟩
def transferEvent : Nat := 55069
def frameStart : Nat := 54994
def rule : BoundRule := .scale (.predecessor 0 55067 .coefficient) (.value (.predecessor 1 55068 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55067 .coefficient)
      LeftAuthority55065.bound (LeftAuthority55065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55068 .coefficient)
      LeftAuthority55056.bound (LeftAuthority55056.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55056.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55065.bound LeftAuthority55056.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55065.bound, LeftAuthority55056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority55065.actual selector witness) * (LeftAuthority55056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55069

namespace LeftBound55072
def owner : Owner := ⟨.program ⟨257⟩, ⟨7303⟩⟩
def transferEvent : Nat := 55072
def frameStart : Nat := 54994
def rule : BoundRule := .identity (.predecessor 0 55071 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55071 .coefficient)
      LeftAuthority55059.bound (LeftAuthority55059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55059.derived selector witness)

def rawBound : CoeffClass := LeftAuthority55059.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority55059.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55072

namespace LeftBound55076
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def transferEvent : Nat := 55076
def frameStart : Nat := 54994
def rule : BoundRule := .product (.predecessor 0 55074 .coefficient) (.predecessor 1 55075 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55074 .coefficient)
      LeftBound55072.bound (LeftBound55072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55075 .coefficient)
      LeftBound55069.bound (LeftBound55069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55069.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55072.bound LeftBound55069.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55072.bound, LeftBound55069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55072.actual selector witness) * (LeftBound55069.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55076

namespace LeftBound55081
def owner : Owner := ⟨.program ⟨257⟩, ⟨17161⟩⟩
def transferEvent : Nat := 55081
def frameStart : Nat := 54994
def rule : BoundRule := .sum [.predecessor 0 55079 .coefficient, .predecessor 1 55080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55079 .coefficient)
      LeftBound55076.bound (LeftBound55076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55080 .coefficient)
      LeftBound55053.bound (LeftBound55053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55076.bound, LeftBound55053.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55076.bound, LeftBound55053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55076.actual selector witness, LeftBound55053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55081

namespace LeftBound55085
def owner : Owner := ⟨.program ⟨257⟩, ⟨17450⟩⟩
def transferEvent : Nat := 55085
def frameStart : Nat := 54994
def rule : BoundRule := .product (.predecessor 0 55083 .coefficient) (.predecessor 1 55084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55083 .coefficient)
      LeftBound55081.bound (LeftBound55081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55084 .coefficient)
      LeftAuthority55038.bound (LeftAuthority55038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact55039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55081.bound LeftAuthority55038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55081.bound, LeftAuthority55038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55081.actual selector witness) * (LeftAuthority55038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55085

namespace LeftBound55096
def owner : Owner := ⟨.program ⟨257⟩, ⟨15854⟩⟩
def transferEvent : Nat := 55096
def frameStart : Nat := 54994
def rule : BoundRule := .product (.predecessor 0 55094 .coefficient) (.predecessor 1 55095 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55094 .coefficient)
      LeftAuthority55049.bound (LeftAuthority55049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55095 .coefficient)
      LeftAuthority55092.bound (LeftAuthority55092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55049.bound LeftAuthority55092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55049.bound, LeftAuthority55092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority55049.actual selector witness) * (LeftAuthority55092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55096

namespace LeftBound55104
def owner : Owner := ⟨.program ⟨257⟩, ⟨15855⟩⟩
def transferEvent : Nat := 55104
def frameStart : Nat := 54994
def rule : BoundRule := .sum [.predecessor 0 55102 .coefficient, .predecessor 1 55103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55102 .coefficient)
      LeftAuthority55100.bound (LeftAuthority55100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55100.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55103 .coefficient)
      LeftBound55096.bound (LeftBound55096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority55100.bound, LeftBound55096.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55100.bound, LeftBound55096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority55100.actual selector witness, LeftBound55096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55104

namespace LeftBound55108
def owner : Owner := ⟨.program ⟨257⟩, ⟨17451⟩⟩
def transferEvent : Nat := 55108
def frameStart : Nat := 54994
def rule : BoundRule := .sum [.predecessor 0 55106 .coefficient, .predecessor 1 55107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55106 .coefficient)
      LeftBound55104.bound (LeftBound55104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55107 .coefficient)
      LeftBound55085.bound (LeftBound55085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55104.bound, LeftBound55085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55104.bound, LeftBound55085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55104.actual selector witness, LeftBound55085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55108

namespace LeftBound55121
def owner : Owner := ⟨.program ⟨257⟩, ⟨17449⟩⟩
def transferEvent : Nat := 55121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55119 .coefficient, .predecessor 1 55120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55119 .coefficient)
      LeftBound54942.bound (LeftBound54942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55120 .coefficient)
      LeftBound54925.bound (LeftBound54925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54942.bound, LeftBound54925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54942.bound, LeftBound54925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54942.actual selector witness, LeftBound54925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55121

namespace LeftBound55124
def owner : Owner := ⟨.program ⟨257⟩, ⟨17449⟩⟩
def transferEvent : Nat := 55124
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55118 .summary, .result 54932 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55118 .summary)
      LeftBound54944.bound (LeftBound54944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16372⟩⟩) (rawTerms := some (Proof.Events215.exact55118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54932 .summary)
      LeftBound54927.bound (LeftBound54927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17448⟩⟩) (rawTerms := some (Proof.Events214.exact54932RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54944.bound, LeftBound54927.bound]
def bound : CoeffClass := .finite ⟨2997816280693142192128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54944.bound, LeftBound54927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54944.actual selector witness, LeftBound54927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55124

namespace LeftBound55128
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def transferEvent : Nat := 55128
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55126 .coefficient) (.predecessor 1 55127 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55126 .coefficient)
      LeftBound55121.bound (LeftBound55121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55127 .coefficient)
      LeftAuthority54847.bound (LeftAuthority54847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events214.exact54848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54847.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55121.bound LeftAuthority54847.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55121.bound, LeftAuthority54847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55121.actual selector witness) * (LeftAuthority54847.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55128

namespace LeftBound55129
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def transferEvent : Nat := 55129
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17985⟩⟩]⟩ [⟨.result 54848 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54848 .coefficient)
      LeftAuthority54847.bound (LeftAuthority54847.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17985⟩⟩) (rawTerms := some (Proof.Events214.exact54848RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54847.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54847.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority54847.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55129

namespace LeftBound55130
def owner : Owner := ⟨.program ⟨257⟩, ⟨17987⟩⟩
def transferEvent : Nat := 55130
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55125 .summary) (.transfer 55129) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55125 .summary)
      LeftBound55124.bound (LeftBound55124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17449⟩⟩) (rawTerms := some (Proof.Events215.exact55125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 55129)
      LeftBound55129.bound (LeftBound55129.actual selector witness) := by
  exact .transfer (LeftBound55129.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound55124.bound LeftBound55129.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55124.bound, LeftBound55129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound55124.actual selector witness) * (LeftBound55129.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55130

namespace LeftBound55141
def owner : Owner := ⟨.program ⟨257⟩, ⟨16758⟩⟩
def transferEvent : Nat := 55141
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 55139 .coefficient) (.value (.predecessor 1 55140 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55139 .coefficient)
      LeftAuthority55137.bound (LeftAuthority55137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55140 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55137.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55137.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority55137.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55141

namespace LeftBound55145
def owner : Owner := ⟨.program ⟨257⟩, ⟨16759⟩⟩
def transferEvent : Nat := 55145
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55143 .coefficient) (.predecessor 1 55144 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55143 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55144 .coefficient)
      LeftBound55141.bound (LeftBound55141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound55141.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound55141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound55141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55145

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
