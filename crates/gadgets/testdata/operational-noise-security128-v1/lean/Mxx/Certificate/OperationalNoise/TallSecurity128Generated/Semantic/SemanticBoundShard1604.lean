import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1603

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound238023
def owner : Owner := ⟨.program ⟨257⟩, ⟨44621⟩⟩
def transferEvent : Nat := 238023
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 238021 .coefficient) (.predecessor 1 238022 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238021 .coefficient)
      LeftBound238016.bound (LeftBound238016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events929.exact238020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238022 .coefficient)
      LeftAuthority237742.bound (LeftAuthority237742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events928.exact237743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237742.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound238016.bound LeftAuthority237742.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238016.bound, LeftAuthority237742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound238016.actual selector witness) * (LeftAuthority237742.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238023

namespace LeftBound238024
def owner : Owner := ⟨.program ⟨257⟩, ⟨44621⟩⟩
def transferEvent : Nat := 238024
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44619⟩⟩]⟩ [⟨.result 237743 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237743 .coefficient)
      LeftAuthority237742.bound (LeftAuthority237742.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44619⟩⟩) (rawTerms := some (Proof.Events928.exact237743RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority237742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority237742.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority237742.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority237742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority237742.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound238024

namespace LeftBound238025
def owner : Owner := ⟨.program ⟨257⟩, ⟨44621⟩⟩
def transferEvent : Nat := 238025
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 238020 .summary) (.transfer 238024) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238020 .summary)
      LeftBound238019.bound (LeftBound238019.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44279⟩⟩) (rawTerms := some (Proof.Events929.exact238020RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 238024)
      LeftBound238024.bound (LeftBound238024.actual selector witness) := by
  exact .transfer (LeftBound238024.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound238019.bound LeftBound238024.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238019.bound, LeftBound238024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound238019.actual selector witness) * (LeftBound238024.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238025

namespace LeftBound238036
def owner : Owner := ⟨.program ⟨257⟩, ⟨43498⟩⟩
def transferEvent : Nat := 238036
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 238034 .coefficient) (.value (.predecessor 1 238035 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238034 .coefficient)
      LeftAuthority238032.bound (LeftAuthority238032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events929.exact238033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238035 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority238032.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238032.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority238032.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound238036

namespace LeftBound238040
def owner : Owner := ⟨.program ⟨257⟩, ⟨43499⟩⟩
def transferEvent : Nat := 238040
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 238038 .coefficient) (.predecessor 1 238039 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238038 .coefficient)
      LeftBound236867.bound (LeftBound236867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238039 .coefficient)
      LeftBound238036.bound (LeftBound238036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events929.exact238037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238036.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236867.bound LeftBound238036.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236867.bound, LeftBound238036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236867.actual selector witness) * (LeftBound238036.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238040

namespace LeftBound238041
def owner : Owner := ⟨.program ⟨257⟩, ⟨43499⟩⟩
def transferEvent : Nat := 238041
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43496⟩⟩]⟩ [⟨.result 238033 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238033 .coefficient)
      LeftAuthority238032.bound (LeftAuthority238032.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43496⟩⟩) (rawTerms := some (Proof.Events929.exact238033RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238032.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority238032.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority238032.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound238041

namespace LeftBound238042
def owner : Owner := ⟨.program ⟨257⟩, ⟨43499⟩⟩
def transferEvent : Nat := 238042
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 238041) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 238041)
      LeftBound238041.bound (LeftBound238041.actual selector witness) := by
  exact .transfer (LeftBound238041.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound238041.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound238041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound238041.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238042

namespace LeftBound238137
def owner : Owner := ⟨.program ⟨257⟩, ⟨42773⟩⟩
def transferEvent : Nat := 238137
def frameStart : Nat := 238098
def rule : BoundRule := .identity (.predecessor 0 238136 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238136 .coefficient)
      LeftAuthority238134.bound (LeftAuthority238134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238134.derived selector witness)

def rawBound : CoeffClass := LeftAuthority238134.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority238134.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound238137

namespace LeftBound238154
def owner : Owner := ⟨.program ⟨257⟩, ⟨44138⟩⟩
def transferEvent : Nat := 238154
def frameStart : Nat := 238098
def rule : BoundRule := .sum [.predecessor 0 238152 .coefficient, .predecessor 1 238153 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238152 .coefficient)
      LeftBound238137.bound (LeftBound238137.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound238137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238153 .coefficient)
      LeftAuthority238150.bound (LeftAuthority238150.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority238150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238137.bound, LeftAuthority238150.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238137.bound, LeftAuthority238150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238137.actual selector witness, LeftAuthority238150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238154

namespace LeftBound238157
def owner : Owner := ⟨.program ⟨257⟩, ⟨44139⟩⟩
def transferEvent : Nat := 238157
def frameStart : Nat := 238098
def rule : BoundRule := .identity (.predecessor 0 238156 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238156 .coefficient)
      LeftBound238154.bound (LeftBound238154.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound238154.derived selector witness)

def rawBound : CoeffClass := LeftBound238154.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound238154.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound238157

namespace LeftBound238163
def owner : Owner := ⟨.program ⟨257⟩, ⟨44140⟩⟩
def transferEvent : Nat := 238163
def frameStart : Nat := 238098
def rule : BoundRule := .product (.predecessor 0 238161 .coefficient) (.predecessor 1 238162 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238161 .coefficient)
      LeftAuthority238159.bound (LeftAuthority238159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238159.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238162 .coefficient)
      LeftBound238157.bound (LeftBound238157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238157.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority238159.bound LeftBound238157.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238159.bound, LeftBound238157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority238159.actual selector witness) * (LeftBound238157.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238163

namespace LeftBound238171
def owner : Owner := ⟨.program ⟨257⟩, ⟨44141⟩⟩
def transferEvent : Nat := 238171
def frameStart : Nat := 238098
def rule : BoundRule := .sum [.predecessor 0 238169 .coefficient, .predecessor 1 238170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238169 .coefficient)
      LeftAuthority238167.bound (LeftAuthority238167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238170 .coefficient)
      LeftBound238163.bound (LeftBound238163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238163.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238163.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority238167.bound, LeftBound238163.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238167.bound, LeftBound238163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority238167.actual selector witness, LeftBound238163.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238171

namespace LeftBound238175
def owner : Owner := ⟨.program ⟨257⟩, ⟨44620⟩⟩
def transferEvent : Nat := 238175
def frameStart : Nat := 238098
def rule : BoundRule := .product (.predecessor 0 238173 .coefficient) (.predecessor 1 238174 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238173 .coefficient)
      LeftBound238171.bound (LeftBound238171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238174 .coefficient)
      LeftAuthority238148.bound (LeftAuthority238148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238148.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238148.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound238171.bound LeftAuthority238148.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238171.bound, LeftAuthority238148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound238171.actual selector witness) * (LeftAuthority238148.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238175

namespace LeftBound238186
def owner : Owner := ⟨.program ⟨257⟩, ⟨42974⟩⟩
def transferEvent : Nat := 238186
def frameStart : Nat := 238098
def rule : BoundRule := .product (.predecessor 0 238184 .coefficient) (.predecessor 1 238185 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238184 .coefficient)
      LeftAuthority238159.bound (LeftAuthority238159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238159.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238185 .coefficient)
      LeftAuthority238182.bound (LeftAuthority238182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority238159.bound LeftAuthority238182.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238159.bound, LeftAuthority238182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority238159.actual selector witness) * (LeftAuthority238182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238186

namespace LeftBound238194
def owner : Owner := ⟨.program ⟨257⟩, ⟨42975⟩⟩
def transferEvent : Nat := 238194
def frameStart : Nat := 238098
def rule : BoundRule := .sum [.predecessor 0 238192 .coefficient, .predecessor 1 238193 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238192 .coefficient)
      LeftAuthority238190.bound (LeftAuthority238190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238193 .coefficient)
      LeftBound238186.bound (LeftBound238186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority238190.bound, LeftBound238186.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238190.bound, LeftBound238186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority238190.actual selector witness, LeftBound238186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238194

namespace LeftBound238198
def owner : Owner := ⟨.program ⟨257⟩, ⟨44623⟩⟩
def transferEvent : Nat := 238198
def frameStart : Nat := 238098
def rule : BoundRule := .sum [.predecessor 0 238196 .coefficient, .predecessor 1 238197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238196 .coefficient)
      LeftBound238194.bound (LeftBound238194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238197 .coefficient)
      LeftBound238175.bound (LeftBound238175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events930.exact238180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238175.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238194.bound, LeftBound238175.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238194.bound, LeftBound238175.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238194.actual selector witness, LeftBound238175.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238198

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
