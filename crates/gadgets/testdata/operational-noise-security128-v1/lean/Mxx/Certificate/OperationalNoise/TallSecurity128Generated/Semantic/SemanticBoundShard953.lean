import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard952

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound144337
def owner : Owner := ⟨.program ⟨257⟩, ⟨66120⟩⟩
def transferEvent : Nat := 144337
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144335 .coefficient, .predecessor 1 144336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144335 .coefficient)
      LeftBound144333.bound (LeftBound144333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144336 .coefficient)
      LeftAuthority143878.bound (LeftAuthority143878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144333.bound, LeftAuthority143878.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144333.bound, LeftAuthority143878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144333.actual selector witness, LeftAuthority143878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144337

namespace LeftBound144340
def owner : Owner := ⟨.program ⟨257⟩, ⟨66121⟩⟩
def transferEvent : Nat := 144340
def frameStart : Nat := 143836
def rule : BoundRule := .identity (.predecessor 0 144339 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144339 .coefficient)
      LeftBound144337.bound (LeftBound144337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144337.derived selector witness)

def rawBound : CoeffClass := LeftBound144337.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound144337.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound144340

namespace LeftBound144357
def owner : Owner := ⟨.program ⟨257⟩, ⟨69059⟩⟩
def transferEvent : Nat := 144357
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144355 .coefficient, .predecessor 1 144356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144355 .coefficient)
      LeftBound144340.bound (LeftBound144340.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound144340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144356 .coefficient)
      LeftAuthority144353.bound (LeftAuthority144353.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority144353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144340.bound, LeftAuthority144353.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144340.bound, LeftAuthority144353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144340.actual selector witness, LeftAuthority144353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144357

namespace LeftBound144360
def owner : Owner := ⟨.program ⟨257⟩, ⟨69060⟩⟩
def transferEvent : Nat := 144360
def frameStart : Nat := 143836
def rule : BoundRule := .identity (.predecessor 0 144359 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144359 .coefficient)
      LeftBound144357.bound (LeftBound144357.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound144357.derived selector witness)

def rawBound : CoeffClass := LeftBound144357.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound144357.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound144360

namespace LeftBound144366
def owner : Owner := ⟨.program ⟨257⟩, ⟨69061⟩⟩
def transferEvent : Nat := 144366
def frameStart : Nat := 143836
def rule : BoundRule := .product (.predecessor 0 144364 .coefficient) (.predecessor 1 144365 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144364 .coefficient)
      LeftAuthority144362.bound (LeftAuthority144362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144365 .coefficient)
      LeftBound144360.bound (LeftBound144360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority144362.bound LeftBound144360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority144362.bound, LeftBound144360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority144362.actual selector witness) * (LeftBound144360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound144366

namespace LeftBound144442
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 144442
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144440 .coefficient, .predecessor 1 144441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144440 .coefficient)
      LeftAuthority144438.bound (LeftAuthority144438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144441 .coefficient)
      LeftAuthority144435.bound (LeftAuthority144435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority144438.bound, LeftAuthority144435.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority144438.bound, LeftAuthority144435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority144438.actual selector witness, LeftAuthority144435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144442

namespace LeftBound144446
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 144446
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144444 .coefficient, .predecessor 1 144445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144444 .coefficient)
      LeftBound144442.bound (LeftBound144442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144445 .coefficient)
      LeftAuthority144432.bound (LeftAuthority144432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144442.bound, LeftAuthority144432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144442.bound, LeftAuthority144432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144442.actual selector witness, LeftAuthority144432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144446

namespace LeftBound144450
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 144450
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144448 .coefficient, .predecessor 1 144449 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144448 .coefficient)
      LeftBound144446.bound (LeftBound144446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144449 .coefficient)
      LeftAuthority144429.bound (LeftAuthority144429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144446.bound, LeftAuthority144429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144446.bound, LeftAuthority144429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144446.actual selector witness, LeftAuthority144429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144450

namespace LeftBound144454
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 144454
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144452 .coefficient, .predecessor 1 144453 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144452 .coefficient)
      LeftBound144450.bound (LeftBound144450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144453 .coefficient)
      LeftAuthority144426.bound (LeftAuthority144426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144450.bound, LeftAuthority144426.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144450.bound, LeftAuthority144426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144450.actual selector witness, LeftAuthority144426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144454

namespace LeftBound144458
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 144458
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144456 .coefficient, .predecessor 1 144457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144456 .coefficient)
      LeftBound144454.bound (LeftBound144454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144457 .coefficient)
      LeftAuthority144423.bound (LeftAuthority144423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144454.bound, LeftAuthority144423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144454.bound, LeftAuthority144423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144454.actual selector witness, LeftAuthority144423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144458

namespace LeftBound144462
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 144462
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144460 .coefficient, .predecessor 1 144461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144460 .coefficient)
      LeftBound144458.bound (LeftBound144458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144461 .coefficient)
      LeftAuthority144420.bound (LeftAuthority144420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144420.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144420.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144458.bound, LeftAuthority144420.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144458.bound, LeftAuthority144420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144458.actual selector witness, LeftAuthority144420.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144462

namespace LeftBound144466
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 144466
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144464 .coefficient, .predecessor 1 144465 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144464 .coefficient)
      LeftBound144462.bound (LeftBound144462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144465 .coefficient)
      LeftAuthority144417.bound (LeftAuthority144417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144462.bound, LeftAuthority144417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144462.bound, LeftAuthority144417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144462.actual selector witness, LeftAuthority144417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144466

namespace LeftBound144470
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 144470
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144468 .coefficient, .predecessor 1 144469 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144468 .coefficient)
      LeftBound144466.bound (LeftBound144466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144469 .coefficient)
      LeftAuthority144414.bound (LeftAuthority144414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144466.bound, LeftAuthority144414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144466.bound, LeftAuthority144414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144466.actual selector witness, LeftAuthority144414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144470

namespace LeftBound144474
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 144474
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144472 .coefficient, .predecessor 1 144473 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144472 .coefficient)
      LeftBound144470.bound (LeftBound144470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144470.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144473 .coefficient)
      LeftAuthority144411.bound (LeftAuthority144411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144411.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144470.bound, LeftAuthority144411.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144470.bound, LeftAuthority144411.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144470.actual selector witness, LeftAuthority144411.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144474

namespace LeftBound144478
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 144478
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144476 .coefficient, .predecessor 1 144477 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144476 .coefficient)
      LeftBound144474.bound (LeftBound144474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144474.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144477 .coefficient)
      LeftAuthority144408.bound (LeftAuthority144408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144474.bound, LeftAuthority144408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144474.bound, LeftAuthority144408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144474.actual selector witness, LeftAuthority144408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144478

namespace LeftBound144482
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 144482
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144480 .coefficient, .predecessor 1 144481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144480 .coefficient)
      LeftBound144478.bound (LeftBound144478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144481 .coefficient)
      LeftAuthority144405.bound (LeftAuthority144405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events564.exact144406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144478.bound, LeftAuthority144405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144478.bound, LeftAuthority144405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144478.actual selector witness, LeftAuthority144405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144482

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
