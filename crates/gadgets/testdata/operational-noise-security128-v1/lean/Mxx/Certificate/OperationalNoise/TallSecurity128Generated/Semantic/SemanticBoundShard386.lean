import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard385

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62442
def owner : Owner := ⟨.program ⟨257⟩, ⟨44095⟩⟩
def transferEvent : Nat := 62442
def frameStart : Nat := 62389
def rule : BoundRule := .identity (.predecessor 0 62441 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62441 .coefficient)
      LeftBound62439.bound (LeftBound62439.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62439.derived selector witness)

def rawBound : CoeffClass := LeftBound62439.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound62439.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62442

namespace LeftBound62448
def owner : Owner := ⟨.program ⟨257⟩, ⟨44096⟩⟩
def transferEvent : Nat := 62448
def frameStart : Nat := 62389
def rule : BoundRule := .product (.predecessor 0 62446 .coefficient) (.predecessor 1 62447 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62446 .coefficient)
      LeftAuthority62444.bound (LeftAuthority62444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62444.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62447 .coefficient)
      LeftBound62442.bound (LeftBound62442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62442.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority62444.bound LeftBound62442.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62444.bound, LeftBound62442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority62444.actual selector witness) * (LeftBound62442.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62448

namespace LeftBound62464
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 62464
def frameStart : Nat := 62389
def rule : BoundRule := .scale (.predecessor 0 62462 .coefficient) (.value (.predecessor 1 62463 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62462 .coefficient)
      LeftAuthority62460.bound (LeftAuthority62460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62463 .coefficient)
      LeftAuthority62451.bound (LeftAuthority62451.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority62451.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62460.bound LeftAuthority62451.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62460.bound, LeftAuthority62451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority62460.actual selector witness) * (LeftAuthority62451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62464

namespace LeftBound62467
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 62467
def frameStart : Nat := 62389
def rule : BoundRule := .identity (.predecessor 0 62466 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62466 .coefficient)
      LeftAuthority62454.bound (LeftAuthority62454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62454.derived selector witness)

def rawBound : CoeffClass := LeftAuthority62454.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority62454.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62467

namespace LeftBound62471
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 62471
def frameStart : Nat := 62389
def rule : BoundRule := .product (.predecessor 0 62469 .coefficient) (.predecessor 1 62470 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62469 .coefficient)
      LeftBound62467.bound (LeftBound62467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62470 .coefficient)
      LeftBound62464.bound (LeftBound62464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62464.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound62467.bound LeftBound62464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62467.bound, LeftBound62464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound62467.actual selector witness) * (LeftBound62464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62471

namespace LeftBound62476
def owner : Owner := ⟨.program ⟨257⟩, ⟨44097⟩⟩
def transferEvent : Nat := 62476
def frameStart : Nat := 62389
def rule : BoundRule := .sum [.predecessor 0 62474 .coefficient, .predecessor 1 62475 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62474 .coefficient)
      LeftBound62471.bound (LeftBound62471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62475 .coefficient)
      LeftBound62448.bound (LeftBound62448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62448.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62471.bound, LeftBound62448.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62471.bound, LeftBound62448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound62471.actual selector witness, LeftBound62448.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62476

namespace LeftBound62480
def owner : Owner := ⟨.program ⟨257⟩, ⟨44379⟩⟩
def transferEvent : Nat := 62480
def frameStart : Nat := 62389
def rule : BoundRule := .product (.predecessor 0 62478 .coefficient) (.predecessor 1 62479 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62478 .coefficient)
      LeftBound62476.bound (LeftBound62476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62479 .coefficient)
      LeftAuthority62433.bound (LeftAuthority62433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62433.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound62476.bound LeftAuthority62433.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62476.bound, LeftAuthority62433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound62476.actual selector witness) * (LeftAuthority62433.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62480

namespace LeftBound62491
def owner : Owner := ⟨.program ⟨257⟩, ⟨42846⟩⟩
def transferEvent : Nat := 62491
def frameStart : Nat := 62389
def rule : BoundRule := .product (.predecessor 0 62489 .coefficient) (.predecessor 1 62490 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62489 .coefficient)
      LeftAuthority62444.bound (LeftAuthority62444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62444.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62490 .coefficient)
      LeftAuthority62487.bound (LeftAuthority62487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62487.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority62444.bound LeftAuthority62487.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62444.bound, LeftAuthority62487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority62444.actual selector witness) * (LeftAuthority62487.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62491

namespace LeftBound62499
def owner : Owner := ⟨.program ⟨257⟩, ⟨42847⟩⟩
def transferEvent : Nat := 62499
def frameStart : Nat := 62389
def rule : BoundRule := .sum [.predecessor 0 62497 .coefficient, .predecessor 1 62498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62497 .coefficient)
      LeftAuthority62495.bound (LeftAuthority62495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62498 .coefficient)
      LeftBound62491.bound (LeftBound62491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62491.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62495.bound, LeftBound62491.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62495.bound, LeftBound62491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority62495.actual selector witness, LeftBound62491.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62499

namespace LeftBound62503
def owner : Owner := ⟨.program ⟨257⟩, ⟨44380⟩⟩
def transferEvent : Nat := 62503
def frameStart : Nat := 62389
def rule : BoundRule := .sum [.predecessor 0 62501 .coefficient, .predecessor 1 62502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62501 .coefficient)
      LeftBound62499.bound (LeftBound62499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62502 .coefficient)
      LeftBound62480.bound (LeftBound62480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62499.bound, LeftBound62480.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62499.bound, LeftBound62480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound62499.actual selector witness, LeftBound62480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62503

namespace LeftBound62516
def owner : Owner := ⟨.program ⟨257⟩, ⟨44378⟩⟩
def transferEvent : Nat := 62516
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62514 .coefficient, .predecessor 1 62515 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62514 .coefficient)
      LeftBound62337.bound (LeftBound62337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62515 .coefficient)
      LeftBound62320.bound (LeftBound62320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62337.bound, LeftBound62320.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62337.bound, LeftBound62320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound62337.actual selector witness, LeftBound62320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62516

namespace LeftBound62519
def owner : Owner := ⟨.program ⟨257⟩, ⟨44378⟩⟩
def transferEvent : Nat := 62519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62513 .summary, .result 62327 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 62513 .summary)
      LeftBound62339.bound (LeftBound62339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43302⟩⟩) (rawTerms := some (Proof.Events244.exact62513RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 62327 .summary)
      LeftBound62322.bound (LeftBound62322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44377⟩⟩) (rawTerms := some (Proof.Events243.exact62327RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62339.bound, LeftBound62322.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62339.bound, LeftBound62322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound62339.actual selector witness, LeftBound62322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62519

namespace LeftBound62523
def owner : Owner := ⟨.program ⟨257⟩, ⟨44846⟩⟩
def transferEvent : Nat := 62523
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62521 .coefficient) (.predecessor 1 62522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62521 .coefficient)
      LeftBound62516.bound (LeftBound62516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62522 .coefficient)
      LeftAuthority62242.bound (LeftAuthority62242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62242.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound62516.bound LeftAuthority62242.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62516.bound, LeftAuthority62242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound62516.actual selector witness) * (LeftAuthority62242.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62523

namespace LeftBound62524
def owner : Owner := ⟨.program ⟨257⟩, ⟨44846⟩⟩
def transferEvent : Nat := 62524
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44844⟩⟩]⟩ [⟨.result 62243 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 62243 .coefficient)
      LeftAuthority62242.bound (LeftAuthority62242.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44844⟩⟩) (rawTerms := some (Proof.Events243.exact62243RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62242.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62242.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority62242.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62524

namespace LeftBound62525
def owner : Owner := ⟨.program ⟨257⟩, ⟨44846⟩⟩
def transferEvent : Nat := 62525
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 62520 .summary) (.transfer 62524) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 62520 .summary)
      LeftBound62519.bound (LeftBound62519.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44378⟩⟩) (rawTerms := some (Proof.Events244.exact62520RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 62524)
      LeftBound62524.bound (LeftBound62524.actual selector witness) := by
  exact .transfer (LeftBound62524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound62519.bound LeftBound62524.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62519.bound, LeftBound62524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound62519.actual selector witness) * (LeftBound62524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62525

namespace LeftBound62536
def owner : Owner := ⟨.program ⟨257⟩, ⟨43678⟩⟩
def transferEvent : Nat := 62536
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 62534 .coefficient) (.value (.predecessor 1 62535 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 62534 .coefficient)
      LeftAuthority62532.bound (LeftAuthority62532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 62535 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62532.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62532.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority62532.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62536

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
