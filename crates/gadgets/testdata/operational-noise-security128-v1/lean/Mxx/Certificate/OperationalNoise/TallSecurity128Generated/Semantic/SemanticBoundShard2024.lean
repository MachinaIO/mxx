import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2023

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound298293
def owner : Owner := ⟨.program ⟨257⟩, ⟨25855⟩⟩
def transferEvent : Nat := 298293
def frameStart : Nat := 298276
def rule : BoundRule := .product (.predecessor 0 298291 .coefficient) (.predecessor 1 298292 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298291 .coefficient)
      LeftAuthority298289.bound (LeftAuthority298289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298292 .coefficient)
      LeftAuthority298286.bound (LeftAuthority298286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority298289.bound LeftAuthority298286.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298289.bound, LeftAuthority298286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority298289.actual selector witness) * (LeftAuthority298286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298293

namespace LeftBound298297
def owner : Owner := ⟨.program ⟨257⟩, ⟨25856⟩⟩
def transferEvent : Nat := 298297
def frameStart : Nat := 298276
def rule : BoundRule := .identity (.predecessor 0 298296 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298296 .coefficient)
      LeftBound298293.bound (LeftBound298293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298293.derived selector witness)

def rawBound : CoeffClass := LeftBound298293.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound298293.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound298297

namespace LeftBound298314
def owner : Owner := ⟨.program ⟨257⟩, ⟨27646⟩⟩
def transferEvent : Nat := 298314
def frameStart : Nat := 298276
def rule : BoundRule := .sum [.predecessor 0 298312 .coefficient, .predecessor 1 298313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298312 .coefficient)
      LeftBound298297.bound (LeftBound298297.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound298297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298313 .coefficient)
      LeftAuthority298310.bound (LeftAuthority298310.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority298310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298297.bound, LeftAuthority298310.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298297.bound, LeftAuthority298310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298297.actual selector witness, LeftAuthority298310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298314

namespace LeftBound298317
def owner : Owner := ⟨.program ⟨257⟩, ⟨27647⟩⟩
def transferEvent : Nat := 298317
def frameStart : Nat := 298276
def rule : BoundRule := .identity (.predecessor 0 298316 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298316 .coefficient)
      LeftBound298314.bound (LeftBound298314.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound298314.derived selector witness)

def rawBound : CoeffClass := LeftBound298314.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound298314.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound298317

namespace LeftBound298323
def owner : Owner := ⟨.program ⟨257⟩, ⟨27648⟩⟩
def transferEvent : Nat := 298323
def frameStart : Nat := 298276
def rule : BoundRule := .product (.predecessor 0 298321 .coefficient) (.predecessor 1 298322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298321 .coefficient)
      LeftAuthority298319.bound (LeftAuthority298319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298322 .coefficient)
      LeftBound298317.bound (LeftBound298317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298317.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority298319.bound LeftBound298317.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298319.bound, LeftBound298317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority298319.actual selector witness) * (LeftBound298317.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298323

namespace LeftBound298339
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 298339
def frameStart : Nat := 298276
def rule : BoundRule := .scale (.predecessor 0 298337 .coefficient) (.value (.predecessor 1 298338 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298337 .coefficient)
      LeftAuthority298335.bound (LeftAuthority298335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298338 .coefficient)
      LeftAuthority298326.bound (LeftAuthority298326.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority298326.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority298335.bound LeftAuthority298326.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298335.bound, LeftAuthority298326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority298335.actual selector witness) * (LeftAuthority298326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound298339

namespace LeftBound298342
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 298342
def frameStart : Nat := 298276
def rule : BoundRule := .identity (.predecessor 0 298341 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298341 .coefficient)
      LeftAuthority298329.bound (LeftAuthority298329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298329.derived selector witness)

def rawBound : CoeffClass := LeftAuthority298329.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority298329.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound298342

namespace LeftBound298346
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 298346
def frameStart : Nat := 298276
def rule : BoundRule := .product (.predecessor 0 298344 .coefficient) (.predecessor 1 298345 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298344 .coefficient)
      LeftBound298342.bound (LeftBound298342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298345 .coefficient)
      LeftBound298339.bound (LeftBound298339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298342.bound LeftBound298339.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298342.bound, LeftBound298339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298342.actual selector witness) * (LeftBound298339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298346

namespace LeftBound298351
def owner : Owner := ⟨.program ⟨257⟩, ⟨27649⟩⟩
def transferEvent : Nat := 298351
def frameStart : Nat := 298276
def rule : BoundRule := .sum [.predecessor 0 298349 .coefficient, .predecessor 1 298350 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298349 .coefficient)
      LeftBound298346.bound (LeftBound298346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298350 .coefficient)
      LeftBound298323.bound (LeftBound298323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298323.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298346.bound, LeftBound298323.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298346.bound, LeftBound298323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298346.actual selector witness, LeftBound298323.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298351

namespace LeftBound298355
def owner : Owner := ⟨.program ⟨257⟩, ⟨27812⟩⟩
def transferEvent : Nat := 298355
def frameStart : Nat := 298276
def rule : BoundRule := .product (.predecessor 0 298353 .coefficient) (.predecessor 1 298354 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298353 .coefficient)
      LeftBound298351.bound (LeftBound298351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298354 .coefficient)
      LeftAuthority298308.bound (LeftAuthority298308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298308.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298351.bound LeftAuthority298308.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298351.bound, LeftAuthority298308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298351.actual selector witness) * (LeftAuthority298308.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298355

namespace LeftBound298366
def owner : Owner := ⟨.program ⟨257⟩, ⟨26330⟩⟩
def transferEvent : Nat := 298366
def frameStart : Nat := 298276
def rule : BoundRule := .product (.predecessor 0 298364 .coefficient) (.predecessor 1 298365 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298364 .coefficient)
      LeftAuthority298319.bound (LeftAuthority298319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298365 .coefficient)
      LeftAuthority298362.bound (LeftAuthority298362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298362.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority298319.bound LeftAuthority298362.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298319.bound, LeftAuthority298362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority298319.actual selector witness) * (LeftAuthority298362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298366

namespace LeftBound298374
def owner : Owner := ⟨.program ⟨257⟩, ⟨26331⟩⟩
def transferEvent : Nat := 298374
def frameStart : Nat := 298276
def rule : BoundRule := .sum [.predecessor 0 298372 .coefficient, .predecessor 1 298373 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298372 .coefficient)
      LeftAuthority298370.bound (LeftAuthority298370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298373 .coefficient)
      LeftBound298366.bound (LeftBound298366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298366.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority298370.bound, LeftBound298366.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298370.bound, LeftBound298366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority298370.actual selector witness, LeftBound298366.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298374

namespace LeftBound298378
def owner : Owner := ⟨.program ⟨257⟩, ⟨27813⟩⟩
def transferEvent : Nat := 298378
def frameStart : Nat := 298276
def rule : BoundRule := .sum [.predecessor 0 298376 .coefficient, .predecessor 1 298377 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298376 .coefficient)
      LeftBound298374.bound (LeftBound298374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298377 .coefficient)
      LeftBound298355.bound (LeftBound298355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298374.bound, LeftBound298355.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298374.bound, LeftBound298355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298374.actual selector witness, LeftBound298355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298378

namespace LeftBound298391
def owner : Owner := ⟨.program ⟨257⟩, ⟨27811⟩⟩
def transferEvent : Nat := 298391
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 298389 .coefficient, .predecessor 1 298390 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298389 .coefficient)
      LeftBound298236.bound (LeftBound298236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298390 .coefficient)
      LeftBound298219.bound (LeftBound298219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1164.exact298226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298236.bound, LeftBound298219.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298236.bound, LeftBound298219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298236.actual selector witness, LeftBound298219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298391

namespace LeftBound298394
def owner : Owner := ⟨.program ⟨257⟩, ⟨27811⟩⟩
def transferEvent : Nat := 298394
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 298388 .summary, .result 298226 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298388 .summary)
      LeftBound298238.bound (LeftBound298238.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26752⟩⟩) (rawTerms := some (Proof.Events1165.exact298388RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298226 .summary)
      LeftBound298221.bound (LeftBound298221.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27810⟩⟩) (rawTerms := some (Proof.Events1164.exact298226RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298238.bound, LeftBound298221.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298238.bound, LeftBound298221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298238.actual selector witness, LeftBound298221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298394

namespace LeftBound298398
def owner : Owner := ⟨.program ⟨257⟩, ⟨28041⟩⟩
def transferEvent : Nat := 298398
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 298396 .coefficient) (.predecessor 1 298397 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298396 .coefficient)
      LeftBound298391.bound (LeftBound298391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1165.exact298395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298397 .coefficient)
      LeftAuthority298141.bound (LeftAuthority298141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1164.exact298142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298391.bound LeftAuthority298141.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298391.bound, LeftAuthority298141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298391.actual selector witness) * (LeftAuthority298141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298398

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
