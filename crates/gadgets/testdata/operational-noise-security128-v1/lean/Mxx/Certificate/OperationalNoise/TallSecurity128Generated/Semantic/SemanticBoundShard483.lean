import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard482

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76449
def owner : Owner := ⟨.program ⟨257⟩, ⟨14875⟩⟩
def transferEvent : Nat := 76449
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76444 .summary) (.transfer 76448) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76444 .summary)
      LeftBound76442.bound (LeftBound76442.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14874⟩⟩) (rawTerms := some (Proof.Events298.exact76444RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 76448)
      LeftBound76448.bound (LeftBound76448.actual selector witness) := by
  exact .transfer (LeftBound76448.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76442.bound LeftBound76448.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76442.bound, LeftBound76448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76442.actual selector witness) * (LeftBound76448.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76449

namespace LeftBound76457
def owner : Owner := ⟨.program ⟨257⟩, ⟨45305⟩⟩
def transferEvent : Nat := 76457
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76455 .coefficient, .predecessor 1 76456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76455 .coefficient)
      LeftBound76447.bound (LeftBound76447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76456 .coefficient)
      LeftBound76419.bound (LeftBound76419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76419.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76447.bound, LeftBound76419.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76447.bound, LeftBound76419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76447.actual selector witness, LeftBound76419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76457

namespace LeftBound76459
def owner : Owner := ⟨.program ⟨257⟩, ⟨45305⟩⟩
def transferEvent : Nat := 76459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 76454 .summary, .result 76424 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76454 .summary)
      LeftBound76449.bound (LeftBound76449.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14875⟩⟩) (rawTerms := some (Proof.Events298.exact76454RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76449.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76424 .summary)
      LeftBound76421.bound (LeftBound76421.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45304⟩⟩) (rawTerms := some (Proof.Events298.exact76424RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76421.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76449.bound, LeftBound76421.bound]
def bound : CoeffClass := .finite ⟨279222288384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76449.bound, LeftBound76421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76449.actual selector witness, LeftBound76421.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76459

namespace LeftBound76463
def owner : Owner := ⟨.program ⟨257⟩, ⟨47046⟩⟩
def transferEvent : Nat := 76463
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76461 .coefficient) (.predecessor 1 76462 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76461 .coefficient)
      LeftBound76457.bound (LeftBound76457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76462 .coefficient)
      LeftAuthority76395.bound (LeftAuthority76395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76395.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76457.bound LeftAuthority76395.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76457.bound, LeftAuthority76395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76457.actual selector witness) * (LeftAuthority76395.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76463

namespace LeftBound76464
def owner : Owner := ⟨.program ⟨257⟩, ⟨47046⟩⟩
def transferEvent : Nat := 76464
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47045⟩⟩]⟩ [⟨.result 76396 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76396 .coefficient)
      LeftAuthority76395.bound (LeftAuthority76395.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47045⟩⟩) (rawTerms := some (Proof.Events298.exact76396RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76395.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76395.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority76395.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76464

namespace LeftBound76465
def owner : Owner := ⟨.program ⟨257⟩, ⟨47046⟩⟩
def transferEvent : Nat := 76465
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76460 .summary) (.transfer 76464) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76460 .summary)
      LeftBound76459.bound (LeftBound76459.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45305⟩⟩) (rawTerms := some (Proof.Events298.exact76460RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 76464)
      LeftBound76464.bound (LeftBound76464.actual selector witness) := by
  exact .transfer (LeftBound76464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound76459.bound LeftBound76464.bound
def bound : CoeffClass := .finite ⟨2998126492308901724160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76459.bound, LeftBound76464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound76459.actual selector witness) * (LeftBound76464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76465

namespace LeftBound76476
def owner : Owner := ⟨.program ⟨257⟩, ⟨45971⟩⟩
def transferEvent : Nat := 76476
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 76474 .coefficient) (.value (.predecessor 1 76475 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76474 .coefficient)
      LeftAuthority76472.bound (LeftAuthority76472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76475 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76472.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76472.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority76472.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76476

namespace LeftBound76480
def owner : Owner := ⟨.program ⟨257⟩, ⟨45972⟩⟩
def transferEvent : Nat := 76480
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76478 .coefficient) (.predecessor 1 76479 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76478 .coefficient)
      LeftBound75992.bound (LeftBound75992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76479 .coefficient)
      LeftBound76476.bound (LeftBound76476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76476.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75992.bound LeftBound76476.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75992.bound, LeftBound76476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75992.actual selector witness) * (LeftBound76476.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76480

namespace LeftBound76481
def owner : Owner := ⟨.program ⟨257⟩, ⟨45972⟩⟩
def transferEvent : Nat := 76481
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨45969⟩⟩]⟩ [⟨.result 76473 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 76473 .coefficient)
      LeftAuthority76472.bound (LeftAuthority76472.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨45969⟩⟩) (rawTerms := some (Proof.Events298.exact76473RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76472.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76472.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority76472.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76481

namespace LeftBound76482
def owner : Owner := ⟨.program ⟨257⟩, ⟨45972⟩⟩
def transferEvent : Nat := 76482
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 76481) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 76481)
      LeftBound76481.bound (LeftBound76481.actual selector witness) := by
  exact .transfer (LeftBound76481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound76481.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound76481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound76481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76482

namespace LeftBound76561
def owner : Owner := ⟨.program ⟨257⟩, ⟨45299⟩⟩
def transferEvent : Nat := 76561
def frameStart : Nat := 76532
def rule : BoundRule := .product (.predecessor 0 76559 .coefficient) (.predecessor 1 76560 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76559 .coefficient)
      LeftAuthority76557.bound (LeftAuthority76557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76560 .coefficient)
      LeftAuthority76554.bound (LeftAuthority76554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76557.bound LeftAuthority76554.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76557.bound, LeftAuthority76554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority76557.actual selector witness) * (LeftAuthority76554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76561

namespace LeftBound76565
def owner : Owner := ⟨.program ⟨257⟩, ⟨45300⟩⟩
def transferEvent : Nat := 76565
def frameStart : Nat := 76532
def rule : BoundRule := .identity (.predecessor 0 76564 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76564 .coefficient)
      LeftBound76561.bound (LeftBound76561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76561.derived selector witness)

def rawBound : CoeffClass := LeftBound76561.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound76561.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76565

namespace LeftBound76582
def owner : Owner := ⟨.program ⟨257⟩, ⟨46770⟩⟩
def transferEvent : Nat := 76582
def frameStart : Nat := 76532
def rule : BoundRule := .sum [.predecessor 0 76580 .coefficient, .predecessor 1 76581 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76580 .coefficient)
      LeftBound76565.bound (LeftBound76565.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76581 .coefficient)
      LeftAuthority76578.bound (LeftAuthority76578.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76578.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76565.bound, LeftAuthority76578.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76565.bound, LeftAuthority76578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound76565.actual selector witness, LeftAuthority76578.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76582

namespace LeftBound76585
def owner : Owner := ⟨.program ⟨257⟩, ⟨46771⟩⟩
def transferEvent : Nat := 76585
def frameStart : Nat := 76532
def rule : BoundRule := .identity (.predecessor 0 76584 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76584 .coefficient)
      LeftBound76582.bound (LeftBound76582.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76582.derived selector witness)

def rawBound : CoeffClass := LeftBound76582.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound76582.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76585

namespace LeftBound76591
def owner : Owner := ⟨.program ⟨257⟩, ⟨46772⟩⟩
def transferEvent : Nat := 76591
def frameStart : Nat := 76532
def rule : BoundRule := .product (.predecessor 0 76589 .coefficient) (.predecessor 1 76590 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76589 .coefficient)
      LeftAuthority76587.bound (LeftAuthority76587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76590 .coefficient)
      LeftBound76585.bound (LeftBound76585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76585.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority76587.bound LeftBound76585.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76587.bound, LeftBound76585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority76587.actual selector witness) * (LeftBound76585.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76591

namespace LeftBound76607
def owner : Owner := ⟨.program ⟨257⟩, ⟨9563⟩⟩
def transferEvent : Nat := 76607
def frameStart : Nat := 76532
def rule : BoundRule := .scale (.predecessor 0 76605 .coefficient) (.value (.predecessor 1 76606 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 76605 .coefficient)
      LeftAuthority76603.bound (LeftAuthority76603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events299.exact76604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 76606 .coefficient)
      LeftAuthority76594.bound (LeftAuthority76594.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76594.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76603.bound LeftAuthority76594.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76603.bound, LeftAuthority76594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority76603.actual selector witness) * (LeftAuthority76594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76607

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
