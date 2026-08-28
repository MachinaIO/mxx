import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard103
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1017

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound153336
def owner : Owner := ⟨.program ⟨257⟩, ⟨66403⟩⟩
def transferEvent : Nat := 153336
def frameStart : Nat := 153240
def rule : BoundRule := .sum [.predecessor 0 153334 .coefficient, .predecessor 1 153335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153334 .coefficient)
      LeftAuthority153332.bound (LeftAuthority153332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events598.exact153333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority153332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority153332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153335 .coefficient)
      LeftBound153328.bound (LeftBound153328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events598.exact153330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority153332.bound, LeftBound153328.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority153332.bound, LeftBound153328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority153332.actual selector witness, LeftBound153328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153336

namespace LeftBound153340
def owner : Owner := ⟨.program ⟨257⟩, ⟨69953⟩⟩
def transferEvent : Nat := 153340
def frameStart : Nat := 153240
def rule : BoundRule := .sum [.predecessor 0 153338 .coefficient, .predecessor 1 153339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153338 .coefficient)
      LeftBound153336.bound (LeftBound153336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events598.exact153337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153339 .coefficient)
      LeftBound153317.bound (LeftBound153317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events598.exact153322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153336.bound, LeftBound153317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153336.bound, LeftBound153317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153336.actual selector witness, LeftBound153317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153340

namespace LeftBound153353
def owner : Owner := ⟨.program ⟨257⟩, ⟨69943⟩⟩
def transferEvent : Nat := 153353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 153351 .coefficient, .predecessor 1 153352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153351 .coefficient)
      LeftBound153182.bound (LeftBound153182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153352 .coefficient)
      LeftBound153165.bound (LeftBound153165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events598.exact153172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153165.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153182.bound, LeftBound153165.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153182.bound, LeftBound153165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153182.actual selector witness, LeftBound153165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153353

namespace LeftBound153356
def owner : Owner := ⟨.program ⟨257⟩, ⟨69943⟩⟩
def transferEvent : Nat := 153356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 153350 .summary, .result 153172 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 153350 .summary)
      LeftBound153184.bound (LeftBound153184.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68020⟩⟩) (rawTerms := some (Proof.Events599.exact153350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound153184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 153172 .summary)
      LeftBound153167.bound (LeftBound153167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69942⟩⟩) (rawTerms := some (Proof.Events598.exact153172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound153167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153184.bound, LeftBound153167.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153184.bound, LeftBound153167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153184.actual selector witness, LeftBound153167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153356

namespace LeftBound153380
def owner : Owner := ⟨.program ⟨257⟩, ⟨25455⟩⟩
def transferEvent : Nat := 153380
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 153378 .coefficient) (.predecessor 1 153379 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153378 .coefficient)
      LeftAuthority7032.bound (LeftAuthority7032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153379 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority7032.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7032.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority7032.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound153380

namespace LeftBound153385
def owner : Owner := ⟨.program ⟨257⟩, ⟨8239⟩⟩
def transferEvent : Nat := 153385
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 153383 .coefficient) (.predecessor 1 153384 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153383 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153384 .coefficient)
      LeftBound21588.bound (LeftBound21588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound21588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound21588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound21588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound153385

namespace LeftBound153390
def owner : Owner := ⟨.program ⟨257⟩, ⟨25456⟩⟩
def transferEvent : Nat := 153390
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 153388 .coefficient, .predecessor 1 153389 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153388 .coefficient)
      LeftBound153385.bound (LeftBound153385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153389 .coefficient)
      LeftBound153380.bound (LeftBound153380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153385.bound, LeftBound153380.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153385.bound, LeftBound153380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153385.actual selector witness, LeftBound153380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153390

namespace LeftBound153394
def owner : Owner := ⟨.program ⟨257⟩, ⟨25457⟩⟩
def transferEvent : Nat := 153394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 153392 .coefficient, .predecessor 1 153393 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153392 .coefficient)
      LeftBound153390.bound (LeftBound153390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153390.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153393 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153390.bound, LeftBound21580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153390.bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153390.actual selector witness, LeftBound21580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153394

namespace LeftBound153395
def owner : Owner := ⟨.program ⟨257⟩, ⟨25457⟩⟩
def transferEvent : Nat := 153395
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩ [⟨.result 21581 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21581 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21580.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21580.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound153395

namespace LeftBound153400
def owner : Owner := ⟨.program ⟨257⟩, ⟨62387⟩⟩
def transferEvent : Nat := 153400
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 153398 .coefficient) (.predecessor 1 153399 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153398 .coefficient)
      LeftBound153394.bound (LeftBound153394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153399 .coefficient)
      LeftAuthority7035.bound (LeftAuthority7035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound153394.bound LeftAuthority7035.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153394.bound, LeftAuthority7035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound153394.actual selector witness) * (LeftAuthority7035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound153400

namespace LeftBound153401
def owner : Owner := ⟨.program ⟨257⟩, ⟨62387⟩⟩
def transferEvent : Nat := 153401
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨62384⟩⟩], []⟩ [⟨.result 7036 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 7036 .coefficient)
      LeftAuthority7035.bound (LeftAuthority7035.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨62384⟩⟩) (rawTerms := some (Proof.Events027.exact7036RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7035.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7035.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority7035.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound153401

namespace LeftBound153402
def owner : Owner := ⟨.program ⟨257⟩, ⟨62387⟩⟩
def transferEvent : Nat := 153402
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 153397 .summary) (.transfer 153401) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 153397 .summary)
      LeftBound153395.bound (LeftBound153395.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25457⟩⟩) (rawTerms := some (Proof.Events599.exact153397RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound153395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 153401)
      LeftBound153401.bound (LeftBound153401.actual selector witness) := by
  exact .transfer (LeftBound153401.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound153395.bound LeftBound153401.bound
def bound : CoeffClass := .finite ⟨18743296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153395.bound, LeftBound153401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound153395.actual selector witness) * (LeftBound153401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound153402

namespace LeftBound153408
def owner : Owner := ⟨.program ⟨257⟩, ⟨62388⟩⟩
def transferEvent : Nat := 153408
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 153406 .coefficient) (.predecessor 1 153407 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153406 .coefficient)
      LeftAuthority7035.bound (LeftAuthority7035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact7036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153407 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority7035.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7035.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority7035.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound153408

namespace LeftBound153413
def owner : Owner := ⟨.program ⟨257⟩, ⟨8257⟩⟩
def transferEvent : Nat := 153413
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 153411 .coefficient) (.predecessor 1 153412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153411 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153412 .coefficient)
      LeftBound21629.bound (LeftBound21629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound21629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound21629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound21629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound153413

namespace LeftBound153418
def owner : Owner := ⟨.program ⟨257⟩, ⟨62389⟩⟩
def transferEvent : Nat := 153418
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 153416 .coefficient, .predecessor 1 153417 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153416 .coefficient)
      LeftBound153413.bound (LeftBound153413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153417 .coefficient)
      LeftBound153408.bound (LeftBound153408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153413.bound, LeftBound153408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153413.bound, LeftBound153408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153413.actual selector witness, LeftBound153408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153418

namespace LeftBound153422
def owner : Owner := ⟨.program ⟨257⟩, ⟨62390⟩⟩
def transferEvent : Nat := 153422
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 153420 .coefficient, .predecessor 1 153421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 153420 .coefficient)
      LeftBound153418.bound (LeftBound153418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 153421 .coefficient)
      LeftBound21621.bound (LeftBound21621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound153418.bound, LeftBound21621.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound153418.bound, LeftBound21621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound153418.actual selector witness, LeftBound21621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound153422

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
