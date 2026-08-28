import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard559

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87923
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def transferEvent : Nat := 87923
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87921 .coefficient) (.predecessor 1 87922 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87921 .coefficient)
      LeftBound75992.bound (LeftBound75992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87922 .coefficient)
      LeftBound87919.bound (LeftBound87919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87919.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75992.bound LeftBound87919.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75992.bound, LeftBound87919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75992.actual selector witness) * (LeftBound87919.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87923

namespace LeftBound87924
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def transferEvent : Nat := 87924
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68193⟩⟩]⟩ [⟨.result 87916 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87916 .coefficient)
      LeftAuthority87915.bound (LeftAuthority87915.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68193⟩⟩) (rawTerms := some (Proof.Events343.exact87916RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87915.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87915.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority87915.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87924

namespace LeftBound87925
def owner : Owner := ⟨.program ⟨257⟩, ⟨68196⟩⟩
def transferEvent : Nat := 87925
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 87924) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 87924)
      LeftBound87924.bound (LeftBound87924.actual selector witness) := by
  exact .transfer (LeftBound87924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound87924.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound87924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound87924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87925

namespace LeftBound88020
def owner : Owner := ⟨.program ⟨257⟩, ⟨65837⟩⟩
def transferEvent : Nat := 88020
def frameStart : Nat := 87981
def rule : BoundRule := .identity (.predecessor 0 88019 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88019 .coefficient)
      LeftAuthority88017.bound (LeftAuthority88017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88017.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88017.derived selector witness)

def rawBound : CoeffClass := LeftAuthority88017.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority88017.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88020

namespace LeftBound88037
def owner : Owner := ⟨.program ⟨257⟩, ⟨69031⟩⟩
def transferEvent : Nat := 88037
def frameStart : Nat := 87981
def rule : BoundRule := .sum [.predecessor 0 88035 .coefficient, .predecessor 1 88036 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88035 .coefficient)
      LeftBound88020.bound (LeftBound88020.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88036 .coefficient)
      LeftAuthority88033.bound (LeftAuthority88033.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority88033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88020.bound, LeftAuthority88033.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88020.bound, LeftAuthority88033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound88020.actual selector witness, LeftAuthority88033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88037

namespace LeftBound88040
def owner : Owner := ⟨.program ⟨257⟩, ⟨69032⟩⟩
def transferEvent : Nat := 88040
def frameStart : Nat := 87981
def rule : BoundRule := .identity (.predecessor 0 88039 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88039 .coefficient)
      LeftBound88037.bound (LeftBound88037.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound88037.derived selector witness)

def rawBound : CoeffClass := LeftBound88037.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound88037.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound88040

namespace LeftBound88046
def owner : Owner := ⟨.program ⟨257⟩, ⟨69033⟩⟩
def transferEvent : Nat := 88046
def frameStart : Nat := 87981
def rule : BoundRule := .product (.predecessor 0 88044 .coefficient) (.predecessor 1 88045 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88044 .coefficient)
      LeftAuthority88042.bound (LeftAuthority88042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88045 .coefficient)
      LeftBound88040.bound (LeftBound88040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority88042.bound LeftBound88040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88042.bound, LeftBound88040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority88042.actual selector witness) * (LeftBound88040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88046

namespace LeftBound88054
def owner : Owner := ⟨.program ⟨257⟩, ⟨69034⟩⟩
def transferEvent : Nat := 88054
def frameStart : Nat := 87981
def rule : BoundRule := .sum [.predecessor 0 88052 .coefficient, .predecessor 1 88053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88052 .coefficient)
      LeftAuthority88050.bound (LeftAuthority88050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88053 .coefficient)
      LeftBound88046.bound (LeftBound88046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88050.bound, LeftBound88046.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88050.bound, LeftBound88046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority88050.actual selector witness, LeftBound88046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88054

namespace LeftBound88058
def owner : Owner := ⟨.program ⟨257⟩, ⟨70637⟩⟩
def transferEvent : Nat := 88058
def frameStart : Nat := 87981
def rule : BoundRule := .product (.predecessor 0 88056 .coefficient) (.predecessor 1 88057 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88056 .coefficient)
      LeftBound88054.bound (LeftBound88054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88057 .coefficient)
      LeftAuthority88031.bound (LeftAuthority88031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88031.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound88054.bound LeftAuthority88031.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88054.bound, LeftAuthority88031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound88054.actual selector witness) * (LeftAuthority88031.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88058

namespace LeftBound88069
def owner : Owner := ⟨.program ⟨257⟩, ⟨67019⟩⟩
def transferEvent : Nat := 88069
def frameStart : Nat := 87981
def rule : BoundRule := .product (.predecessor 0 88067 .coefficient) (.predecessor 1 88068 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88067 .coefficient)
      LeftAuthority88042.bound (LeftAuthority88042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88042.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88068 .coefficient)
      LeftAuthority88065.bound (LeftAuthority88065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority88042.bound LeftAuthority88065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88042.bound, LeftAuthority88065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority88042.actual selector witness) * (LeftAuthority88065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88069

namespace LeftBound88077
def owner : Owner := ⟨.program ⟨257⟩, ⟨67020⟩⟩
def transferEvent : Nat := 88077
def frameStart : Nat := 87981
def rule : BoundRule := .sum [.predecessor 0 88075 .coefficient, .predecessor 1 88076 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88075 .coefficient)
      LeftAuthority88073.bound (LeftAuthority88073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority88073.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority88073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88076 .coefficient)
      LeftBound88069.bound (LeftBound88069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority88073.bound, LeftBound88069.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority88073.bound, LeftBound88069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority88073.actual selector witness, LeftBound88069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88077

namespace LeftBound88081
def owner : Owner := ⟨.program ⟨257⟩, ⟨70650⟩⟩
def transferEvent : Nat := 88081
def frameStart : Nat := 87981
def rule : BoundRule := .sum [.predecessor 0 88079 .coefficient, .predecessor 1 88080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88079 .coefficient)
      LeftBound88077.bound (LeftBound88077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88080 .coefficient)
      LeftBound88058.bound (LeftBound88058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact88063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound88077.bound, LeftBound88058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88077.bound, LeftBound88058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound88077.actual selector witness, LeftBound88058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88081

namespace LeftBound88094
def owner : Owner := ⟨.program ⟨257⟩, ⟨70639⟩⟩
def transferEvent : Nat := 88094
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 88092 .coefficient, .predecessor 1 88093 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88092 .coefficient)
      LeftBound87923.bound (LeftBound87923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88093 .coefficient)
      LeftBound87906.bound (LeftBound87906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87923.bound, LeftBound87906.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87923.bound, LeftBound87906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87923.actual selector witness, LeftBound87906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88094

namespace LeftBound88097
def owner : Owner := ⟨.program ⟨257⟩, ⟨70639⟩⟩
def transferEvent : Nat := 88097
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 88091 .summary, .result 87913 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 88091 .summary)
      LeftBound87925.bound (LeftBound87925.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68196⟩⟩) (rawTerms := some (Proof.Events344.exact88091RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87913 .summary)
      LeftBound87908.bound (LeftBound87908.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70638⟩⟩) (rawTerms := some (Proof.Events343.exact87913RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87925.bound, LeftBound87908.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87925.bound, LeftBound87908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87925.actual selector witness, LeftBound87908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound88097

namespace LeftBound88101
def owner : Owner := ⟨.program ⟨257⟩, ⟨70640⟩⟩
def transferEvent : Nat := 88101
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 88099 .coefficient) (.predecessor 1 88100 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 88099 .coefficient)
      LeftBound88094.bound (LeftBound88094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events344.exact88098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 88100 .coefficient)
      LeftBound15701.bound (LeftBound15701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound88094.bound LeftBound15701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88094.bound, LeftBound15701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound88094.actual selector witness) * (LeftBound15701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound88101

namespace LeftBound88102
def owner : Owner := ⟨.program ⟨257⟩, ⟨70640⟩⟩
def transferEvent : Nat := 88102
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7173⟩⟩]⟩ [⟨.result 15698 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15698 .coefficient)
      LeftAuthority15697.bound (LeftAuthority15697.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7173⟩⟩) (rawTerms := some (Proof.Events061.exact15698RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15697.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15697.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15697.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound88102

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
