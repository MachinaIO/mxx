import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1088

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound164226
def owner : Owner := ⟨.program ⟨257⟩, ⟨45951⟩⟩
def transferEvent : Nat := 164226
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 164224 .coefficient) (.value (.predecessor 1 164225 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164224 .coefficient)
      LeftAuthority164222.bound (LeftAuthority164222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164225 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority164222.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164222.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164222.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound164226

namespace LeftBound164230
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def transferEvent : Nat := 164230
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164228 .coefficient) (.predecessor 1 164229 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164228 .coefficient)
      LeftBound163742.bound (LeftBound163742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164229 .coefficient)
      LeftBound164226.bound (LeftBound164226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164226.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163742.bound LeftBound164226.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163742.bound, LeftBound164226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163742.actual selector witness) * (LeftBound164226.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164230

namespace LeftBound164231
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def transferEvent : Nat := 164231
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨45949⟩⟩]⟩ [⟨.result 164223 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164223 .coefficient)
      LeftAuthority164222.bound (LeftAuthority164222.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨45949⟩⟩) (rawTerms := some (Proof.Events641.exact164223RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164222.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority164222.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164222.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164231

namespace LeftBound164232
def owner : Owner := ⟨.program ⟨257⟩, ⟨45952⟩⟩
def transferEvent : Nat := 164232
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 163745 .summary) (.transfer 164231) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 163745 .summary)
      LeftBound163743.bound (LeftBound163743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6466⟩⟩) (rawTerms := some (Proof.Events639.exact163745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound163743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 164231)
      LeftBound164231.bound (LeftBound164231.actual selector witness) := by
  exact .transfer (LeftBound164231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound163743.bound LeftBound164231.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163743.bound, LeftBound164231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound163743.actual selector witness) * (LeftBound164231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164232

namespace LeftBound164311
def owner : Owner := ⟨.program ⟨257⟩, ⟨45251⟩⟩
def transferEvent : Nat := 164311
def frameStart : Nat := 164282
def rule : BoundRule := .product (.predecessor 0 164309 .coefficient) (.predecessor 1 164310 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164309 .coefficient)
      LeftAuthority164307.bound (LeftAuthority164307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164307.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164310 .coefficient)
      LeftAuthority164304.bound (LeftAuthority164304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164305RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164304.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164304.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority164307.bound LeftAuthority164304.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164307.bound, LeftAuthority164304.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority164307.actual selector witness) * (LeftAuthority164304.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164311

namespace LeftBound164315
def owner : Owner := ⟨.program ⟨257⟩, ⟨45252⟩⟩
def transferEvent : Nat := 164315
def frameStart : Nat := 164282
def rule : BoundRule := .identity (.predecessor 0 164314 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164314 .coefficient)
      LeftBound164311.bound (LeftBound164311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164311.derived selector witness)

def rawBound : CoeffClass := LeftBound164311.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound164311.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound164315

namespace LeftBound164332
def owner : Owner := ⟨.program ⟨257⟩, ⟨46762⟩⟩
def transferEvent : Nat := 164332
def frameStart : Nat := 164282
def rule : BoundRule := .sum [.predecessor 0 164330 .coefficient, .predecessor 1 164331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164330 .coefficient)
      LeftBound164315.bound (LeftBound164315.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound164315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164331 .coefficient)
      LeftAuthority164328.bound (LeftAuthority164328.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority164328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164315.bound, LeftAuthority164328.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164315.bound, LeftAuthority164328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164315.actual selector witness, LeftAuthority164328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164332

namespace LeftBound164335
def owner : Owner := ⟨.program ⟨257⟩, ⟨46763⟩⟩
def transferEvent : Nat := 164335
def frameStart : Nat := 164282
def rule : BoundRule := .identity (.predecessor 0 164334 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164334 .coefficient)
      LeftBound164332.bound (LeftBound164332.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound164332.derived selector witness)

def rawBound : CoeffClass := LeftBound164332.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound164332.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound164335

namespace LeftBound164341
def owner : Owner := ⟨.program ⟨257⟩, ⟨46764⟩⟩
def transferEvent : Nat := 164341
def frameStart : Nat := 164282
def rule : BoundRule := .product (.predecessor 0 164339 .coefficient) (.predecessor 1 164340 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164339 .coefficient)
      LeftAuthority164337.bound (LeftAuthority164337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164340 .coefficient)
      LeftBound164335.bound (LeftBound164335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164335.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority164337.bound LeftBound164335.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164337.bound, LeftBound164335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority164337.actual selector witness) * (LeftBound164335.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164341

namespace LeftBound164357
def owner : Owner := ⟨.program ⟨257⟩, ⟨9563⟩⟩
def transferEvent : Nat := 164357
def frameStart : Nat := 164282
def rule : BoundRule := .scale (.predecessor 0 164355 .coefficient) (.value (.predecessor 1 164356 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164355 .coefficient)
      LeftAuthority164353.bound (LeftAuthority164353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164353.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164356 .coefficient)
      LeftAuthority164344.bound (LeftAuthority164344.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority164344.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority164353.bound LeftAuthority164344.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164353.bound, LeftAuthority164344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164353.actual selector witness) * (LeftAuthority164344.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound164357

namespace LeftBound164360
def owner : Owner := ⟨.program ⟨257⟩, ⟨7301⟩⟩
def transferEvent : Nat := 164360
def frameStart : Nat := 164282
def rule : BoundRule := .identity (.predecessor 0 164359 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164359 .coefficient)
      LeftAuthority164347.bound (LeftAuthority164347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164347.derived selector witness)

def rawBound : CoeffClass := LeftAuthority164347.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority164347.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound164360

namespace LeftBound164364
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def transferEvent : Nat := 164364
def frameStart : Nat := 164282
def rule : BoundRule := .product (.predecessor 0 164362 .coefficient) (.predecessor 1 164363 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164362 .coefficient)
      LeftBound164360.bound (LeftBound164360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164363 .coefficient)
      LeftBound164357.bound (LeftBound164357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164357.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164360.bound LeftBound164357.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164360.bound, LeftBound164357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164360.actual selector witness) * (LeftBound164357.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164364

namespace LeftBound164369
def owner : Owner := ⟨.program ⟨257⟩, ⟨46765⟩⟩
def transferEvent : Nat := 164369
def frameStart : Nat := 164282
def rule : BoundRule := .sum [.predecessor 0 164367 .coefficient, .predecessor 1 164368 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164367 .coefficient)
      LeftBound164364.bound (LeftBound164364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164368 .coefficient)
      LeftBound164341.bound (LeftBound164341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164341.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164364.bound, LeftBound164341.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164364.bound, LeftBound164341.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164364.actual selector witness, LeftBound164341.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164369

namespace LeftBound164373
def owner : Owner := ⟨.program ⟨257⟩, ⟨47026⟩⟩
def transferEvent : Nat := 164373
def frameStart : Nat := 164282
def rule : BoundRule := .product (.predecessor 0 164371 .coefficient) (.predecessor 1 164372 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164371 .coefficient)
      LeftBound164369.bound (LeftBound164369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164372 .coefficient)
      LeftAuthority164326.bound (LeftAuthority164326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164369.bound LeftAuthority164326.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164369.bound, LeftAuthority164326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164369.actual selector witness) * (LeftAuthority164326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164373

namespace LeftBound164384
def owner : Owner := ⟨.program ⟨257⟩, ⟨45502⟩⟩
def transferEvent : Nat := 164384
def frameStart : Nat := 164282
def rule : BoundRule := .product (.predecessor 0 164382 .coefficient) (.predecessor 1 164383 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164382 .coefficient)
      LeftAuthority164337.bound (LeftAuthority164337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164383 .coefficient)
      LeftAuthority164380.bound (LeftAuthority164380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority164337.bound LeftAuthority164380.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164337.bound, LeftAuthority164380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority164337.actual selector witness) * (LeftAuthority164380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164384

namespace LeftBound164392
def owner : Owner := ⟨.program ⟨257⟩, ⟨45503⟩⟩
def transferEvent : Nat := 164392
def frameStart : Nat := 164282
def rule : BoundRule := .sum [.predecessor 0 164390 .coefficient, .predecessor 1 164391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164390 .coefficient)
      LeftAuthority164388.bound (LeftAuthority164388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164391 .coefficient)
      LeftBound164384.bound (LeftBound164384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events642.exact164386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority164388.bound, LeftBound164384.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164388.bound, LeftBound164384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority164388.actual selector witness, LeftBound164384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164392

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
