import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard141

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27455
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 27455
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27453 .coefficient, .predecessor 1 27454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27453 .coefficient)
      LeftBound27451.bound (LeftBound27451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27454 .coefficient)
      LeftAuthority27420.bound (LeftAuthority27420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27420.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27451.bound, LeftAuthority27420.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27451.bound, LeftAuthority27420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27451.actual selector witness, LeftAuthority27420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27455

namespace LeftBound27459
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 27459
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27457 .coefficient, .predecessor 1 27458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27457 .coefficient)
      LeftBound27455.bound (LeftBound27455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27458 .coefficient)
      LeftAuthority27417.bound (LeftAuthority27417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27455.bound, LeftAuthority27417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27455.bound, LeftAuthority27417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27455.actual selector witness, LeftAuthority27417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27459

namespace LeftBound27463
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 27463
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27461 .coefficient, .predecessor 1 27462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27461 .coefficient)
      LeftBound27459.bound (LeftBound27459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27462 .coefficient)
      LeftAuthority27414.bound (LeftAuthority27414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27459.bound, LeftAuthority27414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27459.bound, LeftAuthority27414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27459.actual selector witness, LeftAuthority27414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27463

namespace LeftBound27467
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 27467
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27465 .coefficient, .predecessor 1 27466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27465 .coefficient)
      LeftBound27463.bound (LeftBound27463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27466 .coefficient)
      LeftAuthority27411.bound (LeftAuthority27411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27463.bound, LeftAuthority27411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27463.bound, LeftAuthority27411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27463.actual selector witness, LeftAuthority27411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27467

namespace LeftBound27471
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 27471
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27469 .coefficient, .predecessor 1 27470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27469 .coefficient)
      LeftBound27467.bound (LeftBound27467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27470 .coefficient)
      LeftAuthority27408.bound (LeftAuthority27408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27467.bound, LeftAuthority27408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27467.bound, LeftAuthority27408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27467.actual selector witness, LeftAuthority27408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27471

namespace LeftBound27475
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 27475
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27473 .coefficient, .predecessor 1 27474 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27473 .coefficient)
      LeftBound27471.bound (LeftBound27471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27474 .coefficient)
      LeftAuthority27405.bound (LeftAuthority27405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27471.bound, LeftAuthority27405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27471.bound, LeftAuthority27405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27471.actual selector witness, LeftAuthority27405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27475

namespace LeftBound27479
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 27479
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27477 .coefficient, .predecessor 1 27478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27477 .coefficient)
      LeftBound27475.bound (LeftBound27475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27478 .coefficient)
      LeftAuthority27402.bound (LeftAuthority27402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27402.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27475.bound, LeftAuthority27402.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27475.bound, LeftAuthority27402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27475.actual selector witness, LeftAuthority27402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27479

namespace LeftBound27483
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 27483
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27481 .coefficient, .predecessor 1 27482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27481 .coefficient)
      LeftBound27479.bound (LeftBound27479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27482 .coefficient)
      LeftAuthority27399.bound (LeftAuthority27399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27399.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27479.bound, LeftAuthority27399.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27479.bound, LeftAuthority27399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27479.actual selector witness, LeftAuthority27399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27483

namespace LeftBound27487
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 27487
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27485 .coefficient, .predecessor 1 27486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27485 .coefficient)
      LeftBound27483.bound (LeftBound27483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27486 .coefficient)
      LeftAuthority27396.bound (LeftAuthority27396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27483.bound, LeftAuthority27396.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27483.bound, LeftAuthority27396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27483.actual selector witness, LeftAuthority27396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27487

namespace LeftBound27491
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 27491
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27489 .coefficient, .predecessor 1 27490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27489 .coefficient)
      LeftBound27487.bound (LeftBound27487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27490 .coefficient)
      LeftAuthority27393.bound (LeftAuthority27393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27393.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27487.bound, LeftAuthority27393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27487.bound, LeftAuthority27393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27487.actual selector witness, LeftAuthority27393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27491

namespace LeftBound27495
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 27495
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27493 .coefficient, .predecessor 1 27494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27493 .coefficient)
      LeftBound27491.bound (LeftBound27491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27494 .coefficient)
      LeftAuthority27390.bound (LeftAuthority27390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27390.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27491.bound, LeftAuthority27390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27491.bound, LeftAuthority27390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27491.actual selector witness, LeftAuthority27390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27495

namespace LeftBound27499
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 27499
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27497 .coefficient, .predecessor 1 27498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27497 .coefficient)
      LeftBound27495.bound (LeftBound27495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27498 .coefficient)
      LeftAuthority27387.bound (LeftAuthority27387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27495.bound, LeftAuthority27387.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27495.bound, LeftAuthority27387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27495.actual selector witness, LeftAuthority27387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27499

namespace LeftBound27503
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 27503
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27501 .coefficient, .predecessor 1 27502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27501 .coefficient)
      LeftBound27499.bound (LeftBound27499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27502 .coefficient)
      LeftAuthority27384.bound (LeftAuthority27384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27499.bound, LeftAuthority27384.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27499.bound, LeftAuthority27384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27499.actual selector witness, LeftAuthority27384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27503

namespace LeftBound27507
def owner : Owner := ⟨.program ⟨257⟩, ⟨69054⟩⟩
def transferEvent : Nat := 27507
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27505 .coefficient, .predecessor 1 27506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27505 .coefficient)
      LeftBound27503.bound (LeftBound27503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27506 .coefficient)
      LeftBound27363.bound (LeftBound27363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27503.bound, LeftBound27363.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27503.bound, LeftBound27363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27503.actual selector witness, LeftBound27363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27507

namespace LeftBound27511
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def transferEvent : Nat := 27511
def frameStart : Nat := 26833
def rule : BoundRule := .product (.predecessor 0 27509 .coefficient) (.predecessor 1 27510 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27509 .coefficient)
      LeftBound27507.bound (LeftBound27507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27510 .coefficient)
      LeftAuthority27348.bound (LeftAuthority27348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27348.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27507.bound LeftAuthority27348.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27507.bound, LeftAuthority27348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27507.actual selector witness) * (LeftAuthority27348.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27511

namespace LeftBound27590
def owner : Owner := ⟨.program ⟨257⟩, ⟨67295⟩⟩
def transferEvent : Nat := 27590
def frameStart : Nat := 26833
def rule : BoundRule := .product (.predecessor 0 27588 .coefficient) (.predecessor 1 27589 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27588 .coefficient)
      LeftAuthority27359.bound (LeftAuthority27359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27589 .coefficient)
      LeftAuthority27586.bound (LeftAuthority27586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events107.exact27587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27359.bound LeftAuthority27586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27359.bound, LeftAuthority27586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority27359.actual selector witness) * (LeftAuthority27586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27590

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
