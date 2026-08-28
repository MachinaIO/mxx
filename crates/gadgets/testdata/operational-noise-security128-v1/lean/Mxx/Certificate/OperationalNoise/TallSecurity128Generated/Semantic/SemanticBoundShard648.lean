import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard647

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100418
def owner : Owner := ⟨.program ⟨257⟩, ⟨57217⟩⟩
def transferEvent : Nat := 100418
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100416 .coefficient, .predecessor 1 100417 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100416 .coefficient)
      LeftBound100414.bound (LeftBound100414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100414.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100417 .coefficient)
      LeftAuthority100256.bound (LeftAuthority100256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100414.bound, LeftAuthority100256.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100414.bound, LeftAuthority100256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100414.actual selector witness, LeftAuthority100256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100418

namespace LeftBound100422
def owner : Owner := ⟨.program ⟨257⟩, ⟨60197⟩⟩
def transferEvent : Nat := 100422
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100420 .coefficient, .predecessor 1 100421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100420 .coefficient)
      LeftBound100418.bound (LeftBound100418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100421 .coefficient)
      LeftAuthority100233.bound (LeftAuthority100233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100418.bound, LeftAuthority100233.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100418.bound, LeftAuthority100233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100418.actual selector witness, LeftAuthority100233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100422

namespace LeftBound100426
def owner : Owner := ⟨.program ⟨257⟩, ⟨63177⟩⟩
def transferEvent : Nat := 100426
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100424 .coefficient, .predecessor 1 100425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100424 .coefficient)
      LeftBound100422.bound (LeftBound100422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100425 .coefficient)
      LeftAuthority100210.bound (LeftAuthority100210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100422.bound, LeftAuthority100210.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100422.bound, LeftAuthority100210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100422.actual selector witness, LeftAuthority100210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100426

namespace LeftBound100430
def owner : Owner := ⟨.program ⟨257⟩, ⟨66952⟩⟩
def transferEvent : Nat := 100430
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100428 .coefficient, .predecessor 1 100429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100428 .coefficient)
      LeftBound100426.bound (LeftBound100426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100429 .coefficient)
      LeftAuthority100187.bound (LeftAuthority100187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100187.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100426.bound, LeftAuthority100187.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100426.bound, LeftAuthority100187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100426.actual selector witness, LeftAuthority100187.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100430

namespace LeftBound100434
def owner : Owner := ⟨.program ⟨257⟩, ⟨66953⟩⟩
def transferEvent : Nat := 100434
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100432 .coefficient, .predecessor 1 100433 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100432 .coefficient)
      LeftBound100430.bound (LeftBound100430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100433 .coefficient)
      LeftAuthority100164.bound (LeftAuthority100164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100430.bound, LeftAuthority100164.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100430.bound, LeftAuthority100164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100430.actual selector witness, LeftAuthority100164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100434

namespace LeftBound100438
def owner : Owner := ⟨.program ⟨257⟩, ⟨66954⟩⟩
def transferEvent : Nat := 100438
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100436 .coefficient, .predecessor 1 100437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100436 .coefficient)
      LeftBound100434.bound (LeftBound100434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100437 .coefficient)
      LeftAuthority100141.bound (LeftAuthority100141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100434.bound, LeftAuthority100141.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100434.bound, LeftAuthority100141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100434.actual selector witness, LeftAuthority100141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100438

namespace LeftBound100442
def owner : Owner := ⟨.program ⟨257⟩, ⟨66955⟩⟩
def transferEvent : Nat := 100442
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100440 .coefficient, .predecessor 1 100441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100440 .coefficient)
      LeftBound100438.bound (LeftBound100438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100441 .coefficient)
      LeftAuthority100118.bound (LeftAuthority100118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100438.bound, LeftAuthority100118.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100438.bound, LeftAuthority100118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100438.actual selector witness, LeftAuthority100118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100442

namespace LeftBound100446
def owner : Owner := ⟨.program ⟨257⟩, ⟨66956⟩⟩
def transferEvent : Nat := 100446
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100444 .coefficient, .predecessor 1 100445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100444 .coefficient)
      LeftBound100442.bound (LeftBound100442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100445 .coefficient)
      LeftAuthority100095.bound (LeftAuthority100095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events391.exact100096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100442.bound, LeftAuthority100095.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100442.bound, LeftAuthority100095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100442.actual selector witness, LeftAuthority100095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100446

namespace LeftBound100450
def owner : Owner := ⟨.program ⟨257⟩, ⟨66957⟩⟩
def transferEvent : Nat := 100450
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100448 .coefficient, .predecessor 1 100449 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100448 .coefficient)
      LeftBound100446.bound (LeftBound100446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100449 .coefficient)
      LeftAuthority100072.bound (LeftAuthority100072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100446.bound, LeftAuthority100072.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100446.bound, LeftAuthority100072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100446.actual selector witness, LeftAuthority100072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100450

namespace LeftBound100454
def owner : Owner := ⟨.program ⟨257⟩, ⟨66958⟩⟩
def transferEvent : Nat := 100454
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100452 .coefficient, .predecessor 1 100453 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100452 .coefficient)
      LeftBound100450.bound (LeftBound100450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100453 .coefficient)
      LeftAuthority100049.bound (LeftAuthority100049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100450.bound, LeftAuthority100049.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100450.bound, LeftAuthority100049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100450.actual selector witness, LeftAuthority100049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100454

namespace LeftBound100458
def owner : Owner := ⟨.program ⟨257⟩, ⟨66959⟩⟩
def transferEvent : Nat := 100458
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100456 .coefficient, .predecessor 1 100457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100456 .coefficient)
      LeftBound100454.bound (LeftBound100454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100457 .coefficient)
      LeftAuthority100026.bound (LeftAuthority100026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100454.bound, LeftAuthority100026.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100454.bound, LeftAuthority100026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100454.actual selector witness, LeftAuthority100026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100458

namespace LeftBound100462
def owner : Owner := ⟨.program ⟨257⟩, ⟨66960⟩⟩
def transferEvent : Nat := 100462
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100460 .coefficient, .predecessor 1 100461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100460 .coefficient)
      LeftBound100458.bound (LeftBound100458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100461 .coefficient)
      LeftAuthority100003.bound (LeftAuthority100003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100458.bound, LeftAuthority100003.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100458.bound, LeftAuthority100003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100458.actual selector witness, LeftAuthority100003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100462

namespace LeftBound100465
def owner : Owner := ⟨.program ⟨257⟩, ⟨66961⟩⟩
def transferEvent : Nat := 100465
def frameStart : Nat := 99961
def rule : BoundRule := .identity (.predecessor 0 100464 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100464 .coefficient)
      LeftBound100462.bound (LeftBound100462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100462.derived selector witness)

def rawBound : CoeffClass := LeftBound100462.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound100462.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100465

namespace LeftBound100482
def owner : Owner := ⟨.program ⟨257⟩, ⟨69107⟩⟩
def transferEvent : Nat := 100482
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100480 .coefficient, .predecessor 1 100481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100480 .coefficient)
      LeftBound100465.bound (LeftBound100465.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100481 .coefficient)
      LeftAuthority100478.bound (LeftAuthority100478.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority100478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100465.bound, LeftAuthority100478.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100465.bound, LeftAuthority100478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100465.actual selector witness, LeftAuthority100478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100482

namespace LeftBound100485
def owner : Owner := ⟨.program ⟨257⟩, ⟨69108⟩⟩
def transferEvent : Nat := 100485
def frameStart : Nat := 99961
def rule : BoundRule := .identity (.predecessor 0 100484 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100484 .coefficient)
      LeftBound100482.bound (LeftBound100482.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100482.derived selector witness)

def rawBound : CoeffClass := LeftBound100482.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound100482.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100485

namespace LeftBound100491
def owner : Owner := ⟨.program ⟨257⟩, ⟨69109⟩⟩
def transferEvent : Nat := 100491
def frameStart : Nat := 99961
def rule : BoundRule := .product (.predecessor 0 100489 .coefficient) (.predecessor 1 100490 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100489 .coefficient)
      LeftAuthority100487.bound (LeftAuthority100487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100490 .coefficient)
      LeftBound100485.bound (LeftBound100485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority100487.bound LeftBound100485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100487.bound, LeftBound100485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority100487.actual selector witness) * (LeftBound100485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100491

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
