import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1225
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1226

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound183355
def owner : Owner := ⟨.program ⟨257⟩, ⟨59855⟩⟩
def transferEvent : Nat := 183355
def frameStart : Nat := 183245
def rule : BoundRule := .sum [.predecessor 0 183353 .coefficient, .predecessor 1 183354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183353 .coefficient)
      LeftAuthority183351.bound (LeftAuthority183351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183351.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183354 .coefficient)
      LeftBound183347.bound (LeftBound183347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183347.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority183351.bound, LeftBound183347.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183351.bound, LeftBound183347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority183351.actual selector witness, LeftBound183347.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183355

namespace LeftBound183359
def owner : Owner := ⟨.program ⟨257⟩, ⟨61496⟩⟩
def transferEvent : Nat := 183359
def frameStart : Nat := 183245
def rule : BoundRule := .sum [.predecessor 0 183357 .coefficient, .predecessor 1 183358 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183357 .coefficient)
      LeftBound183355.bound (LeftBound183355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183358 .coefficient)
      LeftBound183336.bound (LeftBound183336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183355.bound, LeftBound183336.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183355.bound, LeftBound183336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183355.actual selector witness, LeftBound183336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183359

namespace LeftBound183372
def owner : Owner := ⟨.program ⟨257⟩, ⟨61494⟩⟩
def transferEvent : Nat := 183372
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 183370 .coefficient, .predecessor 1 183371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183370 .coefficient)
      LeftBound183193.bound (LeftBound183193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183371 .coefficient)
      LeftBound183176.bound (LeftBound183176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183193.bound, LeftBound183176.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183193.bound, LeftBound183176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183193.actual selector witness, LeftBound183176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183372

namespace LeftBound183375
def owner : Owner := ⟨.program ⟨257⟩, ⟨61494⟩⟩
def transferEvent : Nat := 183375
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 183369 .summary, .result 183183 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183369 .summary)
      LeftBound183195.bound (LeftBound183195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60422⟩⟩) (rawTerms := some (Proof.Events716.exact183369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183183 .summary)
      LeftBound183178.bound (LeftBound183178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61493⟩⟩) (rawTerms := some (Proof.Events715.exact183183RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183195.bound, LeftBound183178.bound]
def bound : CoeffClass := .finite ⟨2997962647681031733248, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183195.bound, LeftBound183178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183195.actual selector witness, LeftBound183178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183375

namespace LeftBound183379
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def transferEvent : Nat := 183379
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183377 .coefficient) (.predecessor 1 183378 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183377 .coefficient)
      LeftBound183372.bound (LeftBound183372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183378 .coefficient)
      LeftAuthority183098.bound (LeftAuthority183098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183098.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound183372.bound LeftAuthority183098.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183372.bound, LeftAuthority183098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound183372.actual selector witness) * (LeftAuthority183098.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183379

namespace LeftBound183380
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def transferEvent : Nat := 183380
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61985⟩⟩]⟩ [⟨.result 183099 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183099 .coefficient)
      LeftAuthority183098.bound (LeftAuthority183098.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61985⟩⟩) (rawTerms := some (Proof.Events715.exact183099RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183098.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority183098.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority183098.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound183380

namespace LeftBound183381
def owner : Owner := ⟨.program ⟨257⟩, ⟨61987⟩⟩
def transferEvent : Nat := 183381
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 183376 .summary) (.transfer 183380) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183376 .summary)
      LeftBound183375.bound (LeftBound183375.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61494⟩⟩) (rawTerms := some (Proof.Events716.exact183376RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 183380)
      LeftBound183380.bound (LeftBound183380.actual selector witness) := by
  exact .transfer (LeftBound183380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound183375.bound LeftBound183380.bound
def bound : CoeffClass := .finite ⟨32190378816049003834595889643520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183375.bound, LeftBound183380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound183375.actual selector witness) * (LeftBound183380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183381

namespace LeftBound183392
def owner : Owner := ⟨.program ⟨257⟩, ⟨60758⟩⟩
def transferEvent : Nat := 183392
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 183390 .coefficient) (.value (.predecessor 1 183391 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183390 .coefficient)
      LeftAuthority183388.bound (LeftAuthority183388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183391 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority183388.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183388.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority183388.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound183392

namespace LeftBound183396
def owner : Owner := ⟨.program ⟨257⟩, ⟨60759⟩⟩
def transferEvent : Nat := 183396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 183394 .coefficient) (.predecessor 1 183395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183394 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183395 .coefficient)
      LeftBound183392.bound (LeftBound183392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound183392.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound183392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound183392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183396

namespace LeftBound183397
def owner : Owner := ⟨.program ⟨257⟩, ⟨60759⟩⟩
def transferEvent : Nat := 183397
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨60756⟩⟩]⟩ [⟨.result 183389 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183389 .coefficient)
      LeftAuthority183388.bound (LeftAuthority183388.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨60756⟩⟩) (rawTerms := some (Proof.Events716.exact183389RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183388.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183388.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority183388.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority183388.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound183397

namespace LeftBound183398
def owner : Owner := ⟨.program ⟨257⟩, ⟨60759⟩⟩
def transferEvent : Nat := 183398
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 183397) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 183397)
      LeftBound183397.bound (LeftBound183397.actual selector witness) := by
  exact .transfer (LeftBound183397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound183397.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound183397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound183397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183398

namespace LeftBound183493
def owner : Owner := ⟨.program ⟨257⟩, ⟨59853⟩⟩
def transferEvent : Nat := 183493
def frameStart : Nat := 183454
def rule : BoundRule := .identity (.predecessor 0 183492 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183492 .coefficient)
      LeftAuthority183490.bound (LeftAuthority183490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183490.derived selector witness)

def rawBound : CoeffClass := LeftAuthority183490.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority183490.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound183493

namespace LeftBound183510
def owner : Owner := ⟨.program ⟨257⟩, ⟨61318⟩⟩
def transferEvent : Nat := 183510
def frameStart : Nat := 183454
def rule : BoundRule := .sum [.predecessor 0 183508 .coefficient, .predecessor 1 183509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183508 .coefficient)
      LeftBound183493.bound (LeftBound183493.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound183493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183509 .coefficient)
      LeftAuthority183506.bound (LeftAuthority183506.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority183506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound183493.bound, LeftAuthority183506.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183493.bound, LeftAuthority183506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound183493.actual selector witness, LeftAuthority183506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183510

namespace LeftBound183513
def owner : Owner := ⟨.program ⟨257⟩, ⟨61319⟩⟩
def transferEvent : Nat := 183513
def frameStart : Nat := 183454
def rule : BoundRule := .identity (.predecessor 0 183512 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183512 .coefficient)
      LeftBound183510.bound (LeftBound183510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound183510.derived selector witness)

def rawBound : CoeffClass := LeftBound183510.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound183510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound183510.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound183513

namespace LeftBound183519
def owner : Owner := ⟨.program ⟨257⟩, ⟨61320⟩⟩
def transferEvent : Nat := 183519
def frameStart : Nat := 183454
def rule : BoundRule := .product (.predecessor 0 183517 .coefficient) (.predecessor 1 183518 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183517 .coefficient)
      LeftAuthority183515.bound (LeftAuthority183515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183518 .coefficient)
      LeftBound183513.bound (LeftBound183513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority183515.bound LeftBound183513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183515.bound, LeftBound183513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority183515.actual selector witness) * (LeftBound183513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound183519

namespace LeftBound183527
def owner : Owner := ⟨.program ⟨257⟩, ⟨61321⟩⟩
def transferEvent : Nat := 183527
def frameStart : Nat := 183454
def rule : BoundRule := .sum [.predecessor 0 183525 .coefficient, .predecessor 1 183526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 183525 .coefficient)
      LeftAuthority183523.bound (LeftAuthority183523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority183523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority183523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 183526 .coefficient)
      LeftBound183519.bound (LeftBound183519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events716.exact183521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183519.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority183523.bound, LeftBound183519.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority183523.bound, LeftBound183519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority183523.actual selector witness, LeftBound183519.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound183527

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
