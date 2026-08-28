import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard123

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24267
def owner : Owner := ⟨.program ⟨257⟩, ⟨31252⟩⟩
def transferEvent : Nat := 24267
def frameStart : Nat := 24238
def rule : BoundRule := .product (.predecessor 0 24265 .coefficient) (.predecessor 1 24266 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24265 .coefficient)
      LeftAuthority24263.bound (LeftAuthority24263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24266 .coefficient)
      LeftAuthority24260.bound (LeftAuthority24260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24260.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24263.bound LeftAuthority24260.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24263.bound, LeftAuthority24260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority24263.actual selector witness) * (LeftAuthority24260.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24267

namespace LeftBound24271
def owner : Owner := ⟨.program ⟨257⟩, ⟨31253⟩⟩
def transferEvent : Nat := 24271
def frameStart : Nat := 24238
def rule : BoundRule := .identity (.predecessor 0 24270 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24270 .coefficient)
      LeftBound24267.bound (LeftBound24267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24267.derived selector witness)

def rawBound : CoeffClass := LeftBound24267.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound24267.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24271

namespace LeftBound24288
def owner : Owner := ⟨.program ⟨257⟩, ⟨33190⟩⟩
def transferEvent : Nat := 24288
def frameStart : Nat := 24238
def rule : BoundRule := .sum [.predecessor 0 24286 .coefficient, .predecessor 1 24287 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24286 .coefficient)
      LeftBound24271.bound (LeftBound24271.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24287 .coefficient)
      LeftAuthority24284.bound (LeftAuthority24284.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24271.bound, LeftAuthority24284.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24271.bound, LeftAuthority24284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24271.actual selector witness, LeftAuthority24284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24288

namespace LeftBound24291
def owner : Owner := ⟨.program ⟨257⟩, ⟨33191⟩⟩
def transferEvent : Nat := 24291
def frameStart : Nat := 24238
def rule : BoundRule := .identity (.predecessor 0 24290 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24290 .coefficient)
      LeftBound24288.bound (LeftBound24288.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24288.derived selector witness)

def rawBound : CoeffClass := LeftBound24288.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound24288.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24291

namespace LeftBound24297
def owner : Owner := ⟨.program ⟨257⟩, ⟨33192⟩⟩
def transferEvent : Nat := 24297
def frameStart : Nat := 24238
def rule : BoundRule := .product (.predecessor 0 24295 .coefficient) (.predecessor 1 24296 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24295 .coefficient)
      LeftAuthority24293.bound (LeftAuthority24293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24296 .coefficient)
      LeftBound24291.bound (LeftBound24291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24291.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority24293.bound LeftBound24291.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24293.bound, LeftBound24291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority24293.actual selector witness) * (LeftBound24291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24297

namespace LeftBound24313
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 24313
def frameStart : Nat := 24238
def rule : BoundRule := .scale (.predecessor 0 24311 .coefficient) (.value (.predecessor 1 24312 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24311 .coefficient)
      LeftAuthority24309.bound (LeftAuthority24309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24309.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24312 .coefficient)
      LeftAuthority24300.bound (LeftAuthority24300.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24300.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24309.bound LeftAuthority24300.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24309.bound, LeftAuthority24300.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority24309.actual selector witness) * (LeftAuthority24300.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24313

namespace LeftBound24316
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 24316
def frameStart : Nat := 24238
def rule : BoundRule := .identity (.predecessor 0 24315 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24315 .coefficient)
      LeftAuthority24303.bound (LeftAuthority24303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24303.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24303.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority24303.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24316

namespace LeftBound24320
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 24320
def frameStart : Nat := 24238
def rule : BoundRule := .product (.predecessor 0 24318 .coefficient) (.predecessor 1 24319 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24318 .coefficient)
      LeftBound24316.bound (LeftBound24316.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24317RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24316.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24319 .coefficient)
      LeftBound24313.bound (LeftBound24313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24313.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24316.bound LeftBound24313.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24316.bound, LeftBound24313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24316.actual selector witness) * (LeftBound24313.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24320

namespace LeftBound24325
def owner : Owner := ⟨.program ⟨257⟩, ⟨33193⟩⟩
def transferEvent : Nat := 24325
def frameStart : Nat := 24238
def rule : BoundRule := .sum [.predecessor 0 24323 .coefficient, .predecessor 1 24324 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24323 .coefficient)
      LeftBound24320.bound (LeftBound24320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24320.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24324 .coefficient)
      LeftBound24297.bound (LeftBound24297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24320.bound, LeftBound24297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24320.bound, LeftBound24297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24320.actual selector witness, LeftBound24297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24325

namespace LeftBound24329
def owner : Owner := ⟨.program ⟨257⟩, ⟨33366⟩⟩
def transferEvent : Nat := 24329
def frameStart : Nat := 24238
def rule : BoundRule := .product (.predecessor 0 24327 .coefficient) (.predecessor 1 24328 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24327 .coefficient)
      LeftBound24325.bound (LeftBound24325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24328 .coefficient)
      LeftAuthority24282.bound (LeftAuthority24282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24282.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24325.bound LeftAuthority24282.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24325.bound, LeftAuthority24282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24325.actual selector witness) * (LeftAuthority24282.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24329

namespace LeftBound24340
def owner : Owner := ⟨.program ⟨257⟩, ⟨31760⟩⟩
def transferEvent : Nat := 24340
def frameStart : Nat := 24238
def rule : BoundRule := .product (.predecessor 0 24338 .coefficient) (.predecessor 1 24339 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24338 .coefficient)
      LeftAuthority24293.bound (LeftAuthority24293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24339 .coefficient)
      LeftAuthority24336.bound (LeftAuthority24336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24336.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24293.bound LeftAuthority24336.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24293.bound, LeftAuthority24336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority24293.actual selector witness) * (LeftAuthority24336.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24340

namespace LeftBound24348
def owner : Owner := ⟨.program ⟨257⟩, ⟨31761⟩⟩
def transferEvent : Nat := 24348
def frameStart : Nat := 24238
def rule : BoundRule := .sum [.predecessor 0 24346 .coefficient, .predecessor 1 24347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24346 .coefficient)
      LeftAuthority24344.bound (LeftAuthority24344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24347 .coefficient)
      LeftBound24340.bound (LeftBound24340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24344.bound, LeftBound24340.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24344.bound, LeftBound24340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority24344.actual selector witness, LeftBound24340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24348

namespace LeftBound24352
def owner : Owner := ⟨.program ⟨257⟩, ⟨33367⟩⟩
def transferEvent : Nat := 24352
def frameStart : Nat := 24238
def rule : BoundRule := .sum [.predecessor 0 24350 .coefficient, .predecessor 1 24351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24350 .coefficient)
      LeftBound24348.bound (LeftBound24348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24351 .coefficient)
      LeftBound24329.bound (LeftBound24329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24348.bound, LeftBound24329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24348.bound, LeftBound24329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24348.actual selector witness, LeftBound24329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24352

namespace LeftBound24365
def owner : Owner := ⟨.program ⟨257⟩, ⟨33365⟩⟩
def transferEvent : Nat := 24365
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24363 .coefficient, .predecessor 1 24364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24363 .coefficient)
      LeftBound24186.bound (LeftBound24186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24364 .coefficient)
      LeftBound24169.bound (LeftBound24169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24186.bound, LeftBound24169.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24186.bound, LeftBound24169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24186.actual selector witness, LeftBound24169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24365

namespace LeftBound24368
def owner : Owner := ⟨.program ⟨257⟩, ⟨33365⟩⟩
def transferEvent : Nat := 24368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24362 .summary, .result 24176 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24362 .summary)
      LeftBound24188.bound (LeftBound24188.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32305⟩⟩) (rawTerms := some (Proof.Events095.exact24362RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24176 .summary)
      LeftBound24171.bound (LeftBound24171.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33364⟩⟩) (rawTerms := some (Proof.Events094.exact24176RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24188.bound, LeftBound24171.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24188.bound, LeftBound24171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound24188.actual selector witness, LeftBound24171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24368

namespace LeftBound24372
def owner : Owner := ⟨.program ⟨257⟩, ⟨33624⟩⟩
def transferEvent : Nat := 24372
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24370 .coefficient) (.predecessor 1 24371 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 24370 .coefficient)
      LeftBound24365.bound (LeftBound24365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 24371 .coefficient)
      LeftAuthority24072.bound (LeftAuthority24072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound24365.bound LeftAuthority24072.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24365.bound, LeftAuthority24072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound24365.actual selector witness) * (LeftAuthority24072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24372

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
