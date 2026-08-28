import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard526

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82265
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def transferEvent : Nat := 82265
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51509⟩⟩]⟩ [⟨.result 82257 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 82257 .coefficient)
      LeftAuthority82256.bound (LeftAuthority82256.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51509⟩⟩) (rawTerms := some (Proof.Events321.exact82257RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82256.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82256.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority82256.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82265

namespace LeftBound82266
def owner : Owner := ⟨.program ⟨257⟩, ⟨51512⟩⟩
def transferEvent : Nat := 82266
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 82265) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 82265)
      LeftBound82265.bound (LeftBound82265.actual selector witness) := by
  exact .transfer (LeftBound82265.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound82265.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound82265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound82265.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82266

namespace LeftBound82345
def owner : Owner := ⟨.program ⟨257⟩, ⟨50708⟩⟩
def transferEvent : Nat := 82345
def frameStart : Nat := 82316
def rule : BoundRule := .product (.predecessor 0 82343 .coefficient) (.predecessor 1 82344 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82343 .coefficient)
      LeftAuthority82341.bound (LeftAuthority82341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82341.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82344 .coefficient)
      LeftAuthority82338.bound (LeftAuthority82338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82341.bound LeftAuthority82338.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82341.bound, LeftAuthority82338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority82341.actual selector witness) * (LeftAuthority82338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82345

namespace LeftBound82349
def owner : Owner := ⟨.program ⟨257⟩, ⟨50709⟩⟩
def transferEvent : Nat := 82349
def frameStart : Nat := 82316
def rule : BoundRule := .identity (.predecessor 0 82348 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82348 .coefficient)
      LeftBound82345.bound (LeftBound82345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82345.derived selector witness)

def rawBound : CoeffClass := LeftBound82345.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound82345.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82349

namespace LeftBound82366
def owner : Owner := ⟨.program ⟨257⟩, ⟨52310⟩⟩
def transferEvent : Nat := 82366
def frameStart : Nat := 82316
def rule : BoundRule := .sum [.predecessor 0 82364 .coefficient, .predecessor 1 82365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82364 .coefficient)
      LeftBound82349.bound (LeftBound82349.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82365 .coefficient)
      LeftAuthority82362.bound (LeftAuthority82362.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82349.bound, LeftAuthority82362.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82349.bound, LeftAuthority82362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound82349.actual selector witness, LeftAuthority82362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82366

namespace LeftBound82369
def owner : Owner := ⟨.program ⟨257⟩, ⟨52311⟩⟩
def transferEvent : Nat := 82369
def frameStart : Nat := 82316
def rule : BoundRule := .identity (.predecessor 0 82368 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82368 .coefficient)
      LeftBound82366.bound (LeftBound82366.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82366.derived selector witness)

def rawBound : CoeffClass := LeftBound82366.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound82366.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82369

namespace LeftBound82375
def owner : Owner := ⟨.program ⟨257⟩, ⟨52312⟩⟩
def transferEvent : Nat := 82375
def frameStart : Nat := 82316
def rule : BoundRule := .product (.predecessor 0 82373 .coefficient) (.predecessor 1 82374 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82373 .coefficient)
      LeftAuthority82371.bound (LeftAuthority82371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82374 .coefficient)
      LeftBound82369.bound (LeftBound82369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority82371.bound LeftBound82369.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82371.bound, LeftBound82369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority82371.actual selector witness) * (LeftBound82369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82375

namespace LeftBound82391
def owner : Owner := ⟨.program ⟨257⟩, ⟨9581⟩⟩
def transferEvent : Nat := 82391
def frameStart : Nat := 82316
def rule : BoundRule := .scale (.predecessor 0 82389 .coefficient) (.value (.predecessor 1 82390 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82389 .coefficient)
      LeftAuthority82387.bound (LeftAuthority82387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82390 .coefficient)
      LeftAuthority82378.bound (LeftAuthority82378.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82378.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority82387.bound LeftAuthority82378.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82387.bound, LeftAuthority82378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority82387.actual selector witness) * (LeftAuthority82378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82391

namespace LeftBound82394
def owner : Owner := ⟨.program ⟨257⟩, ⟨7288⟩⟩
def transferEvent : Nat := 82394
def frameStart : Nat := 82316
def rule : BoundRule := .identity (.predecessor 0 82393 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82393 .coefficient)
      LeftAuthority82381.bound (LeftAuthority82381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82381.derived selector witness)

def rawBound : CoeffClass := LeftAuthority82381.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority82381.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82394

namespace LeftBound82398
def owner : Owner := ⟨.program ⟨257⟩, ⟨9582⟩⟩
def transferEvent : Nat := 82398
def frameStart : Nat := 82316
def rule : BoundRule := .product (.predecessor 0 82396 .coefficient) (.predecessor 1 82397 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82396 .coefficient)
      LeftBound82394.bound (LeftBound82394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82397 .coefficient)
      LeftBound82391.bound (LeftBound82391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82391.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound82394.bound LeftBound82391.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82394.bound, LeftBound82391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound82394.actual selector witness) * (LeftBound82391.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82398

namespace LeftBound82403
def owner : Owner := ⟨.program ⟨257⟩, ⟨52313⟩⟩
def transferEvent : Nat := 82403
def frameStart : Nat := 82316
def rule : BoundRule := .sum [.predecessor 0 82401 .coefficient, .predecessor 1 82402 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82401 .coefficient)
      LeftBound82398.bound (LeftBound82398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82402 .coefficient)
      LeftBound82375.bound (LeftBound82375.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82375.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82375.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82398.bound, LeftBound82375.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82398.bound, LeftBound82375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound82398.actual selector witness, LeftBound82375.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82403

namespace LeftBound82407
def owner : Owner := ⟨.program ⟨257⟩, ⟨52588⟩⟩
def transferEvent : Nat := 82407
def frameStart : Nat := 82316
def rule : BoundRule := .product (.predecessor 0 82405 .coefficient) (.predecessor 1 82406 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82405 .coefficient)
      LeftBound82403.bound (LeftBound82403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82406 .coefficient)
      LeftAuthority82360.bound (LeftAuthority82360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound82403.bound LeftAuthority82360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82403.bound, LeftAuthority82360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound82403.actual selector witness) * (LeftAuthority82360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82407

namespace LeftBound82418
def owner : Owner := ⟨.program ⟨257⟩, ⟨50938⟩⟩
def transferEvent : Nat := 82418
def frameStart : Nat := 82316
def rule : BoundRule := .product (.predecessor 0 82416 .coefficient) (.predecessor 1 82417 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82416 .coefficient)
      LeftAuthority82371.bound (LeftAuthority82371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82417 .coefficient)
      LeftAuthority82414.bound (LeftAuthority82414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82371.bound LeftAuthority82414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82371.bound, LeftAuthority82414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority82371.actual selector witness) * (LeftAuthority82414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82418

namespace LeftBound82426
def owner : Owner := ⟨.program ⟨257⟩, ⟨50939⟩⟩
def transferEvent : Nat := 82426
def frameStart : Nat := 82316
def rule : BoundRule := .sum [.predecessor 0 82424 .coefficient, .predecessor 1 82425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82424 .coefficient)
      LeftAuthority82422.bound (LeftAuthority82422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82425 .coefficient)
      LeftBound82418.bound (LeftBound82418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82422.bound, LeftBound82418.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82422.bound, LeftBound82418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority82422.actual selector witness, LeftBound82418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82426

namespace LeftBound82430
def owner : Owner := ⟨.program ⟨257⟩, ⟨52589⟩⟩
def transferEvent : Nat := 82430
def frameStart : Nat := 82316
def rule : BoundRule := .sum [.predecessor 0 82428 .coefficient, .predecessor 1 82429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82428 .coefficient)
      LeftBound82426.bound (LeftBound82426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82429 .coefficient)
      LeftBound82407.bound (LeftBound82407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82426.bound, LeftBound82407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82426.bound, LeftBound82407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound82426.actual selector witness, LeftBound82407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82430

namespace LeftBound82443
def owner : Owner := ⟨.program ⟨257⟩, ⟨52587⟩⟩
def transferEvent : Nat := 82443
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82441 .coefficient, .predecessor 1 82442 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 82441 .coefficient)
      LeftBound82264.bound (LeftBound82264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 82442 .coefficient)
      LeftBound82247.bound (LeftBound82247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82264.bound, LeftBound82247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82264.bound, LeftBound82247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound82264.actual selector witness, LeftBound82247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82443

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
