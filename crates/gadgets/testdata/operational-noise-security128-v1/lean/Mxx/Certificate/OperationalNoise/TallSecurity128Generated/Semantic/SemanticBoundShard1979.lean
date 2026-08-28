import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1929
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1978

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound292522
def owner : Owner := ⟨.program ⟨257⟩, ⟨26361⟩⟩
def transferEvent : Nat := 292522
def frameStart : Nat := 292483
def rule : BoundRule := .identity (.predecessor 0 292521 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292521 .coefficient)
      LeftAuthority292519.bound (LeftAuthority292519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292519.derived selector witness)

def rawBound : CoeffClass := LeftAuthority292519.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority292519.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound292522

namespace LeftBound292539
def owner : Owner := ⟨.program ⟨257⟩, ⟨27742⟩⟩
def transferEvent : Nat := 292539
def frameStart : Nat := 292483
def rule : BoundRule := .sum [.predecessor 0 292537 .coefficient, .predecessor 1 292538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292537 .coefficient)
      LeftBound292522.bound (LeftBound292522.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound292522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292538 .coefficient)
      LeftAuthority292535.bound (LeftAuthority292535.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority292535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292522.bound, LeftAuthority292535.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292522.bound, LeftAuthority292535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292522.actual selector witness, LeftAuthority292535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292539

namespace LeftBound292542
def owner : Owner := ⟨.program ⟨257⟩, ⟨27743⟩⟩
def transferEvent : Nat := 292542
def frameStart : Nat := 292483
def rule : BoundRule := .identity (.predecessor 0 292541 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292541 .coefficient)
      LeftBound292539.bound (LeftBound292539.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound292539.derived selector witness)

def rawBound : CoeffClass := LeftBound292539.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound292539.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound292542

namespace LeftBound292548
def owner : Owner := ⟨.program ⟨257⟩, ⟨27744⟩⟩
def transferEvent : Nat := 292548
def frameStart : Nat := 292483
def rule : BoundRule := .product (.predecessor 0 292546 .coefficient) (.predecessor 1 292547 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292546 .coefficient)
      LeftAuthority292544.bound (LeftAuthority292544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292547 .coefficient)
      LeftBound292542.bound (LeftBound292542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292542.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority292544.bound LeftBound292542.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292544.bound, LeftBound292542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority292544.actual selector witness) * (LeftBound292542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292548

namespace LeftBound292556
def owner : Owner := ⟨.program ⟨257⟩, ⟨27745⟩⟩
def transferEvent : Nat := 292556
def frameStart : Nat := 292483
def rule : BoundRule := .sum [.predecessor 0 292554 .coefficient, .predecessor 1 292555 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292554 .coefficient)
      LeftAuthority292552.bound (LeftAuthority292552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292555 .coefficient)
      LeftBound292548.bound (LeftBound292548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority292552.bound, LeftBound292548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292552.bound, LeftBound292548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority292552.actual selector witness, LeftBound292548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292556

namespace LeftBound292560
def owner : Owner := ⟨.program ⟨257⟩, ⟨28134⟩⟩
def transferEvent : Nat := 292560
def frameStart : Nat := 292483
def rule : BoundRule := .product (.predecessor 0 292558 .coefficient) (.predecessor 1 292559 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292558 .coefficient)
      LeftBound292556.bound (LeftBound292556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292559 .coefficient)
      LeftAuthority292533.bound (LeftAuthority292533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292556.bound LeftAuthority292533.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292556.bound, LeftAuthority292533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292556.actual selector witness) * (LeftAuthority292533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292560

namespace LeftBound292571
def owner : Owner := ⟨.program ⟨257⟩, ⟨26546⟩⟩
def transferEvent : Nat := 292571
def frameStart : Nat := 292483
def rule : BoundRule := .product (.predecessor 0 292569 .coefficient) (.predecessor 1 292570 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292569 .coefficient)
      LeftAuthority292544.bound (LeftAuthority292544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292544.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292570 .coefficient)
      LeftAuthority292567.bound (LeftAuthority292567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority292544.bound LeftAuthority292567.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292544.bound, LeftAuthority292567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority292544.actual selector witness) * (LeftAuthority292567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292571

namespace LeftBound292579
def owner : Owner := ⟨.program ⟨257⟩, ⟨26547⟩⟩
def transferEvent : Nat := 292579
def frameStart : Nat := 292483
def rule : BoundRule := .sum [.predecessor 0 292577 .coefficient, .predecessor 1 292578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292577 .coefficient)
      LeftAuthority292575.bound (LeftAuthority292575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292578 .coefficient)
      LeftBound292571.bound (LeftBound292571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292571.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority292575.bound, LeftBound292571.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292575.bound, LeftBound292571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority292575.actual selector witness, LeftBound292571.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292579

namespace LeftBound292583
def owner : Owner := ⟨.program ⟨257⟩, ⟨28138⟩⟩
def transferEvent : Nat := 292583
def frameStart : Nat := 292483
def rule : BoundRule := .sum [.predecessor 0 292581 .coefficient, .predecessor 1 292582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292581 .coefficient)
      LeftBound292579.bound (LeftBound292579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292582 .coefficient)
      LeftBound292560.bound (LeftBound292560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292579.bound, LeftBound292560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292579.bound, LeftBound292560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292579.actual selector witness, LeftBound292560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292583

namespace LeftBound292596
def owner : Owner := ⟨.program ⟨257⟩, ⟨28136⟩⟩
def transferEvent : Nat := 292596
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 292594 .coefficient, .predecessor 1 292595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292594 .coefficient)
      LeftBound292425.bound (LeftBound292425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292595 .coefficient)
      LeftBound292408.bound (LeftBound292408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292408.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292425.bound, LeftBound292408.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292425.bound, LeftBound292408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292425.actual selector witness, LeftBound292408.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292596

namespace LeftBound292599
def owner : Owner := ⟨.program ⟨257⟩, ⟨28136⟩⟩
def transferEvent : Nat := 292599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 292593 .summary, .result 292415 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292593 .summary)
      LeftBound292427.bound (LeftBound292427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27035⟩⟩) (rawTerms := some (Proof.Events1142.exact292593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292415 .summary)
      LeftBound292410.bound (LeftBound292410.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28135⟩⟩) (rawTerms := some (Proof.Events1142.exact292415RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound292427.bound, LeftBound292410.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292427.bound, LeftBound292410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound292427.actual selector witness, LeftBound292410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound292599

namespace LeftBound292603
def owner : Owner := ⟨.program ⟨257⟩, ⟨28137⟩⟩
def transferEvent : Nat := 292603
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 292601 .coefficient) (.predecessor 1 292602 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292601 .coefficient)
      LeftBound292596.bound (LeftBound292596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1142.exact292600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound292596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound292596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292602 .coefficient)
      LeftBound15681.bound (LeftBound15681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292596.bound LeftBound15681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292596.bound, LeftBound15681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292596.actual selector witness) * (LeftBound15681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292603

namespace LeftBound292604
def owner : Owner := ⟨.program ⟨257⟩, ⟨28137⟩⟩
def transferEvent : Nat := 292604
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩ [⟨.result 15678 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15678 .coefficient)
      LeftAuthority15677.bound (LeftAuthority15677.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7169⟩⟩) (rawTerms := some (Proof.Events061.exact15678RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15677.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15677.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15677.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound292604

namespace LeftBound292605
def owner : Owner := ⟨.program ⟨257⟩, ⟨28137⟩⟩
def transferEvent : Nat := 292605
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 292600 .summary) (.transfer 292604) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292600 .summary)
      LeftBound292599.bound (LeftBound292599.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28136⟩⟩) (rawTerms := some (Proof.Events1142.exact292600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 292604)
      LeftBound292604.bound (LeftBound292604.actual selector witness) := by
  exact .transfer (LeftBound292604.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound292599.bound LeftBound292604.bound
def bound : CoeffClass := .finite ⟨345654216875549026890382321864211871825920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound292599.bound, LeftBound292604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound292599.actual selector witness) * (LeftBound292604.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292605

namespace LeftBound292620
def owner : Owner := ⟨.program ⟨257⟩, ⟨69690⟩⟩
def transferEvent : Nat := 292620
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 292618 .coefficient) (.predecessor 1 292619 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 292618 .coefficient)
      LeftBound284765.bound (LeftBound284765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1112.exact284769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound284765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound284765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 292619 .coefficient)
      LeftAuthority292616.bound (LeftAuthority292616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1143.exact292617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292616.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound284765.bound LeftAuthority292616.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound284765.bound, LeftAuthority292616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound284765.actual selector witness) * (LeftAuthority292616.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound292620

namespace LeftBound292621
def owner : Owner := ⟨.program ⟨257⟩, ⟨69690⟩⟩
def transferEvent : Nat := 292621
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨69688⟩⟩]⟩ [⟨.result 292617 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292617 .coefficient)
      LeftAuthority292616.bound (LeftAuthority292616.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨69688⟩⟩) (rawTerms := some (Proof.Events1143.exact292617RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority292616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority292616.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority292616.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority292616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority292616.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound292621

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
