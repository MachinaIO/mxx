import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1459

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound217450
def owner : Owner := ⟨.program ⟨257⟩, ⟨66607⟩⟩
def transferEvent : Nat := 217450
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217448 .coefficient, .predecessor 1 217449 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217448 .coefficient)
      LeftBound217446.bound (LeftBound217446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217449 .coefficient)
      LeftAuthority217072.bound (LeftAuthority217072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events847.exact217073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217446.bound, LeftAuthority217072.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217446.bound, LeftAuthority217072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217446.actual selector witness, LeftAuthority217072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217450

namespace LeftBound217454
def owner : Owner := ⟨.program ⟨257⟩, ⟨66608⟩⟩
def transferEvent : Nat := 217454
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217452 .coefficient, .predecessor 1 217453 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217452 .coefficient)
      LeftBound217450.bound (LeftBound217450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217450.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217453 .coefficient)
      LeftAuthority217049.bound (LeftAuthority217049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events847.exact217050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217049.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217450.bound, LeftAuthority217049.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217450.bound, LeftAuthority217049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217450.actual selector witness, LeftAuthority217049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217454

namespace LeftBound217458
def owner : Owner := ⟨.program ⟨257⟩, ⟨66609⟩⟩
def transferEvent : Nat := 217458
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217456 .coefficient, .predecessor 1 217457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217456 .coefficient)
      LeftBound217454.bound (LeftBound217454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217457 .coefficient)
      LeftAuthority217026.bound (LeftAuthority217026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events847.exact217027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217454.bound, LeftAuthority217026.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217454.bound, LeftAuthority217026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217454.actual selector witness, LeftAuthority217026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217458

namespace LeftBound217462
def owner : Owner := ⟨.program ⟨257⟩, ⟨66610⟩⟩
def transferEvent : Nat := 217462
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217460 .coefficient, .predecessor 1 217461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217460 .coefficient)
      LeftBound217458.bound (LeftBound217458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217461 .coefficient)
      LeftAuthority217003.bound (LeftAuthority217003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events847.exact217004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217458.bound, LeftAuthority217003.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217458.bound, LeftAuthority217003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217458.actual selector witness, LeftAuthority217003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217462

namespace LeftBound217465
def owner : Owner := ⟨.program ⟨257⟩, ⟨66611⟩⟩
def transferEvent : Nat := 217465
def frameStart : Nat := 216961
def rule : BoundRule := .identity (.predecessor 0 217464 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217464 .coefficient)
      LeftBound217462.bound (LeftBound217462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217462.derived selector witness)

def rawBound : CoeffClass := LeftBound217462.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound217462.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound217465

namespace LeftBound217482
def owner : Owner := ⟨.program ⟨257⟩, ⟨69087⟩⟩
def transferEvent : Nat := 217482
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217480 .coefficient, .predecessor 1 217481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217480 .coefficient)
      LeftBound217465.bound (LeftBound217465.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound217465.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217481 .coefficient)
      LeftAuthority217478.bound (LeftAuthority217478.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority217478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217465.bound, LeftAuthority217478.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217465.bound, LeftAuthority217478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217465.actual selector witness, LeftAuthority217478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217482

namespace LeftBound217485
def owner : Owner := ⟨.program ⟨257⟩, ⟨69088⟩⟩
def transferEvent : Nat := 217485
def frameStart : Nat := 216961
def rule : BoundRule := .identity (.predecessor 0 217484 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217484 .coefficient)
      LeftBound217482.bound (LeftBound217482.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound217482.derived selector witness)

def rawBound : CoeffClass := LeftBound217482.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound217482.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound217485

namespace LeftBound217491
def owner : Owner := ⟨.program ⟨257⟩, ⟨69089⟩⟩
def transferEvent : Nat := 217491
def frameStart : Nat := 216961
def rule : BoundRule := .product (.predecessor 0 217489 .coefficient) (.predecessor 1 217490 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217489 .coefficient)
      LeftAuthority217487.bound (LeftAuthority217487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217490 .coefficient)
      LeftBound217485.bound (LeftBound217485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority217487.bound LeftBound217485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority217487.bound, LeftBound217485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority217487.actual selector witness) * (LeftBound217485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound217491

namespace LeftBound217567
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 217567
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217565 .coefficient, .predecessor 1 217566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217565 .coefficient)
      LeftAuthority217563.bound (LeftAuthority217563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217566 .coefficient)
      LeftAuthority217560.bound (LeftAuthority217560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority217563.bound, LeftAuthority217560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority217563.bound, LeftAuthority217560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority217563.actual selector witness, LeftAuthority217560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217567

namespace LeftBound217571
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 217571
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217569 .coefficient, .predecessor 1 217570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217569 .coefficient)
      LeftBound217567.bound (LeftBound217567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217570 .coefficient)
      LeftAuthority217557.bound (LeftAuthority217557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217567.bound, LeftAuthority217557.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217567.bound, LeftAuthority217557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217567.actual selector witness, LeftAuthority217557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217571

namespace LeftBound217575
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 217575
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217573 .coefficient, .predecessor 1 217574 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217573 .coefficient)
      LeftBound217571.bound (LeftBound217571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217574 .coefficient)
      LeftAuthority217554.bound (LeftAuthority217554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217571.bound, LeftAuthority217554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217571.bound, LeftAuthority217554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217571.actual selector witness, LeftAuthority217554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217575

namespace LeftBound217579
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 217579
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217577 .coefficient, .predecessor 1 217578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217577 .coefficient)
      LeftBound217575.bound (LeftBound217575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217578 .coefficient)
      LeftAuthority217551.bound (LeftAuthority217551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217575.bound, LeftAuthority217551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217575.bound, LeftAuthority217551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217575.actual selector witness, LeftAuthority217551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217579

namespace LeftBound217583
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 217583
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217581 .coefficient, .predecessor 1 217582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217581 .coefficient)
      LeftBound217579.bound (LeftBound217579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217582 .coefficient)
      LeftAuthority217548.bound (LeftAuthority217548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217579.bound, LeftAuthority217548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217579.bound, LeftAuthority217548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217579.actual selector witness, LeftAuthority217548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217583

namespace LeftBound217587
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 217587
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217585 .coefficient, .predecessor 1 217586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217585 .coefficient)
      LeftBound217583.bound (LeftBound217583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217586 .coefficient)
      LeftAuthority217545.bound (LeftAuthority217545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217545.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217545.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217583.bound, LeftAuthority217545.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217583.bound, LeftAuthority217545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217583.actual selector witness, LeftAuthority217545.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217587

namespace LeftBound217591
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 217591
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217589 .coefficient, .predecessor 1 217590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217589 .coefficient)
      LeftBound217587.bound (LeftBound217587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217590 .coefficient)
      LeftAuthority217542.bound (LeftAuthority217542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217587.bound, LeftAuthority217542.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217587.bound, LeftAuthority217542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217587.actual selector witness, LeftAuthority217542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217591

namespace LeftBound217595
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 217595
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217593 .coefficient, .predecessor 1 217594 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217593 .coefficient)
      LeftBound217591.bound (LeftBound217591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217594 .coefficient)
      LeftAuthority217539.bound (LeftAuthority217539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217591.bound, LeftAuthority217539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217591.bound, LeftAuthority217539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217591.actual selector witness, LeftAuthority217539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217595

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
