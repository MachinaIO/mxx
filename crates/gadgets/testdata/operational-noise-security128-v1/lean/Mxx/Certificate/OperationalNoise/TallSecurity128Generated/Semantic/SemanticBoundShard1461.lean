import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1458
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1459
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1460

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound217599
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 217599
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217597 .coefficient, .predecessor 1 217598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217597 .coefficient)
      LeftBound217595.bound (LeftBound217595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217598 .coefficient)
      LeftAuthority217536.bound (LeftAuthority217536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217595.bound, LeftAuthority217536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217595.bound, LeftAuthority217536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217595.actual selector witness, LeftAuthority217536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217599

namespace LeftBound217603
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 217603
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217601 .coefficient, .predecessor 1 217602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217601 .coefficient)
      LeftBound217599.bound (LeftBound217599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217602 .coefficient)
      LeftAuthority217533.bound (LeftAuthority217533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217599.bound, LeftAuthority217533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217599.bound, LeftAuthority217533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217599.actual selector witness, LeftAuthority217533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217603

namespace LeftBound217607
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 217607
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217605 .coefficient, .predecessor 1 217606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217605 .coefficient)
      LeftBound217603.bound (LeftBound217603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217606 .coefficient)
      LeftAuthority217530.bound (LeftAuthority217530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217603.bound, LeftAuthority217530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217603.bound, LeftAuthority217530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217603.actual selector witness, LeftAuthority217530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217607

namespace LeftBound217611
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 217611
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217609 .coefficient, .predecessor 1 217610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217609 .coefficient)
      LeftBound217607.bound (LeftBound217607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217610 .coefficient)
      LeftAuthority217527.bound (LeftAuthority217527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217607.bound, LeftAuthority217527.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217607.bound, LeftAuthority217527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217607.actual selector witness, LeftAuthority217527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217611

namespace LeftBound217615
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 217615
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217613 .coefficient, .predecessor 1 217614 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217613 .coefficient)
      LeftBound217611.bound (LeftBound217611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217614 .coefficient)
      LeftAuthority217524.bound (LeftAuthority217524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217611.bound, LeftAuthority217524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217611.bound, LeftAuthority217524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217611.actual selector witness, LeftAuthority217524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217615

namespace LeftBound217619
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 217619
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217617 .coefficient, .predecessor 1 217618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217617 .coefficient)
      LeftBound217615.bound (LeftBound217615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217618 .coefficient)
      LeftAuthority217521.bound (LeftAuthority217521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217615.bound, LeftAuthority217521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217615.bound, LeftAuthority217521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217615.actual selector witness, LeftAuthority217521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217619

namespace LeftBound217623
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 217623
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217621 .coefficient, .predecessor 1 217622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217621 .coefficient)
      LeftBound217619.bound (LeftBound217619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217622 .coefficient)
      LeftAuthority217518.bound (LeftAuthority217518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217518.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217619.bound, LeftAuthority217518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217619.bound, LeftAuthority217518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217619.actual selector witness, LeftAuthority217518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217623

namespace LeftBound217627
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 217627
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217625 .coefficient, .predecessor 1 217626 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217625 .coefficient)
      LeftBound217623.bound (LeftBound217623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217626 .coefficient)
      LeftAuthority217515.bound (LeftAuthority217515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217623.bound, LeftAuthority217515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217623.bound, LeftAuthority217515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217623.actual selector witness, LeftAuthority217515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217627

namespace LeftBound217631
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 217631
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217629 .coefficient, .predecessor 1 217630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217629 .coefficient)
      LeftBound217627.bound (LeftBound217627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217630 .coefficient)
      LeftAuthority217512.bound (LeftAuthority217512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217512.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217627.bound, LeftAuthority217512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217627.bound, LeftAuthority217512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217627.actual selector witness, LeftAuthority217512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217631

namespace LeftBound217635
def owner : Owner := ⟨.program ⟨257⟩, ⟨69090⟩⟩
def transferEvent : Nat := 217635
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217633 .coefficient, .predecessor 1 217634 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217633 .coefficient)
      LeftBound217631.bound (LeftBound217631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217634 .coefficient)
      LeftBound217491.bound (LeftBound217491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217491.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217631.bound, LeftBound217491.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217631.bound, LeftBound217491.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217631.actual selector witness, LeftBound217491.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217635

namespace LeftBound217639
def owner : Owner := ⟨.program ⟨257⟩, ⟨71237⟩⟩
def transferEvent : Nat := 217639
def frameStart : Nat := 216961
def rule : BoundRule := .product (.predecessor 0 217637 .coefficient) (.predecessor 1 217638 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217637 .coefficient)
      LeftBound217635.bound (LeftBound217635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217635.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217638 .coefficient)
      LeftAuthority217476.bound (LeftAuthority217476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217476.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217476.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound217635.bound LeftAuthority217476.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217635.bound, LeftAuthority217476.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound217635.actual selector witness) * (LeftAuthority217476.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound217639

namespace LeftBound217718
def owner : Owner := ⟨.program ⟨257⟩, ⟨67459⟩⟩
def transferEvent : Nat := 217718
def frameStart : Nat := 216961
def rule : BoundRule := .product (.predecessor 0 217716 .coefficient) (.predecessor 1 217717 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217716 .coefficient)
      LeftAuthority217487.bound (LeftAuthority217487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217717 .coefficient)
      LeftAuthority217714.bound (LeftAuthority217714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217714.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority217487.bound LeftAuthority217714.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority217487.bound, LeftAuthority217714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority217487.actual selector witness) * (LeftAuthority217714.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound217718

namespace LeftBound217726
def owner : Owner := ⟨.program ⟨257⟩, ⟨67464⟩⟩
def transferEvent : Nat := 217726
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217724 .coefficient, .predecessor 1 217725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217724 .coefficient)
      LeftAuthority217722.bound (LeftAuthority217722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217725 .coefficient)
      LeftBound217718.bound (LeftBound217718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority217722.bound, LeftBound217718.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority217722.bound, LeftBound217718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority217722.actual selector witness, LeftBound217718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217726

namespace LeftBound217730
def owner : Owner := ⟨.program ⟨257⟩, ⟨71241⟩⟩
def transferEvent : Nat := 217730
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217728 .coefficient, .predecessor 1 217729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217728 .coefficient)
      LeftBound217726.bound (LeftBound217726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217729 .coefficient)
      LeftBound217639.bound (LeftBound217639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217726.bound, LeftBound217639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217726.bound, LeftBound217639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217726.actual selector witness, LeftBound217639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217730

namespace LeftBound217777
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def transferEvent : Nat := 217777
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 217775 .coefficient, .predecessor 1 217776 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217775 .coefficient)
      LeftBound216368.bound (LeftBound216368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events850.exact217774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217776 .coefficient)
      LeftBound216283.bound (LeftBound216283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events845.exact216358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216368.bound, LeftBound216283.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216368.bound, LeftBound216283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216368.actual selector witness, LeftBound216283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217777

namespace LeftBound217814
def owner : Owner := ⟨.program ⟨257⟩, ⟨71239⟩⟩
def transferEvent : Nat := 217814
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 217774 .summary, .result 216358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 217774 .summary)
      LeftBound216370.bound (LeftBound216370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68373⟩⟩) (rawTerms := some (Proof.Events850.exact217774RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216358 .summary)
      LeftBound216285.bound (LeftBound216285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71238⟩⟩) (rawTerms := some (Proof.Events845.exact216358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216370.bound, LeftBound216285.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469506489977540968448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216370.bound, LeftBound216285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216370.actual selector witness, LeftBound216285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217814

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
