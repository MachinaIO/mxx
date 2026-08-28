import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard012
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard013

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound4499
def owner : Owner := ⟨.program ⟨257⟩, ⟨60202⟩⟩
def transferEvent : Nat := 4499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4497 .coefficient, .predecessor 1 4498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4497 .coefficient)
      LeftBound4495.bound (LeftBound4495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4498 .coefficient)
      LeftBound4410.bound (LeftBound4410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4495.bound, LeftBound4410.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4495.bound, LeftBound4410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4495.actual selector witness, LeftBound4410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4499

namespace LeftBound4503
def owner : Owner := ⟨.program ⟨257⟩, ⟨63182⟩⟩
def transferEvent : Nat := 4503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4501 .coefficient, .predecessor 1 4502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4501 .coefficient)
      LeftBound4499.bound (LeftBound4499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4502 .coefficient)
      LeftBound4402.bound (LeftBound4402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4499.bound, LeftBound4402.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4499.bound, LeftBound4402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4499.actual selector witness, LeftBound4402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4503

namespace LeftBound4507
def owner : Owner := ⟨.program ⟨257⟩, ⟨66940⟩⟩
def transferEvent : Nat := 4507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4505 .coefficient, .predecessor 1 4506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4505 .coefficient)
      LeftBound4503.bound (LeftBound4503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4506 .coefficient)
      LeftBound4394.bound (LeftBound4394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4503.bound, LeftBound4394.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4503.bound, LeftBound4394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4503.actual selector witness, LeftBound4394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4507

namespace LeftBound4511
def owner : Owner := ⟨.program ⟨257⟩, ⟨66941⟩⟩
def transferEvent : Nat := 4511
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4509 .coefficient, .predecessor 1 4510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4509 .coefficient)
      LeftBound4507.bound (LeftBound4507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4510 .coefficient)
      LeftBound4386.bound (LeftBound4386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4507.bound, LeftBound4386.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4507.bound, LeftBound4386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4507.actual selector witness, LeftBound4386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4511

namespace LeftBound4515
def owner : Owner := ⟨.program ⟨257⟩, ⟨66942⟩⟩
def transferEvent : Nat := 4515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4513 .coefficient, .predecessor 1 4514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4513 .coefficient)
      LeftBound4511.bound (LeftBound4511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4514 .coefficient)
      LeftBound4378.bound (LeftBound4378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4511.bound, LeftBound4378.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4511.bound, LeftBound4378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4511.actual selector witness, LeftBound4378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4515

namespace LeftBound4519
def owner : Owner := ⟨.program ⟨257⟩, ⟨66943⟩⟩
def transferEvent : Nat := 4519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4517 .coefficient, .predecessor 1 4518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4517 .coefficient)
      LeftBound4515.bound (LeftBound4515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4518 .coefficient)
      LeftBound4370.bound (LeftBound4370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4515.bound, LeftBound4370.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4515.bound, LeftBound4370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4515.actual selector witness, LeftBound4370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4519

namespace LeftBound4523
def owner : Owner := ⟨.program ⟨257⟩, ⟨66944⟩⟩
def transferEvent : Nat := 4523
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4521 .coefficient, .predecessor 1 4522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4521 .coefficient)
      LeftBound4519.bound (LeftBound4519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4522 .coefficient)
      LeftBound4362.bound (LeftBound4362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4519.bound, LeftBound4362.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4519.bound, LeftBound4362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4519.actual selector witness, LeftBound4362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4523

namespace LeftBound4527
def owner : Owner := ⟨.program ⟨257⟩, ⟨66945⟩⟩
def transferEvent : Nat := 4527
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4525 .coefficient, .predecessor 1 4526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4525 .coefficient)
      LeftBound4523.bound (LeftBound4523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4526 .coefficient)
      LeftBound4354.bound (LeftBound4354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4523.bound, LeftBound4354.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4523.bound, LeftBound4354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4523.actual selector witness, LeftBound4354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4527

namespace LeftBound4531
def owner : Owner := ⟨.program ⟨257⟩, ⟨66946⟩⟩
def transferEvent : Nat := 4531
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4529 .coefficient, .predecessor 1 4530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4529 .coefficient)
      LeftBound4527.bound (LeftBound4527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4530 .coefficient)
      LeftBound4346.bound (LeftBound4346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4527.bound, LeftBound4346.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4527.bound, LeftBound4346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4527.actual selector witness, LeftBound4346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4531

namespace LeftBound4535
def owner : Owner := ⟨.program ⟨257⟩, ⟨66947⟩⟩
def transferEvent : Nat := 4535
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4533 .coefficient, .predecessor 1 4534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4533 .coefficient)
      LeftBound4531.bound (LeftBound4531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4534 .coefficient)
      LeftBound4338.bound (LeftBound4338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4531.bound, LeftBound4338.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4531.bound, LeftBound4338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4531.actual selector witness, LeftBound4338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4535

namespace LeftBound4539
def owner : Owner := ⟨.program ⟨257⟩, ⟨66948⟩⟩
def transferEvent : Nat := 4539
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4537 .coefficient, .predecessor 1 4538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4537 .coefficient)
      LeftBound4535.bound (LeftBound4535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4538 .coefficient)
      LeftBound4330.bound (LeftBound4330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4535.bound, LeftBound4330.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4535.bound, LeftBound4330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4535.actual selector witness, LeftBound4330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4539

namespace LeftBound4543
def owner : Owner := ⟨.program ⟨257⟩, ⟨67569⟩⟩
def transferEvent : Nat := 4543
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 4541 .coefficient, .predecessor 1 4542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4541 .coefficient)
      LeftBound4539.bound (LeftBound4539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4542 .coefficient)
      LeftBound4322.bound (LeftBound4322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound4539.bound, LeftBound4322.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4539.bound, LeftBound4322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound4539.actual selector witness, LeftBound4322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound4543

namespace LeftBound4547
def owner : Owner := ⟨.program ⟨257⟩, ⟨67570⟩⟩
def transferEvent : Nat := 4547
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4545 .coefficient) (.predecessor 1 4546 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 4545 .coefficient)
      LeftBound4543.bound (LeftBound4543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound4543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound4543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 4546 .coefficient)
      LeftAuthority3820.bound (LeftAuthority3820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound4543.bound LeftAuthority3820.bound
def bound : CoeffClass := .finite ⟨223529891348418298797505805253447602201678612727609896473620943264911270816916914851763519744766275047759070230113084014740802988133059038385191832687590800455445274867304318927489096632644648239104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound4543.bound, LeftAuthority3820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound4543.actual selector witness) * (LeftAuthority3820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4547

namespace LeftBound5070
def owner : Owner := ⟨.program ⟨257⟩, ⟨67477⟩⟩
def transferEvent : Nat := 5070
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5068 .coefficient) (.predecessor 1 5069 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 5068 .coefficient)
      LeftAuthority5066.bound (LeftAuthority5066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 5069 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5066.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5066.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority5066.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5070

namespace LeftBound5078
def owner : Owner := ⟨.program ⟨257⟩, ⟨48373⟩⟩
def transferEvent : Nat := 5078
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5076 .coefficient) (.predecessor 1 5077 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 5076 .coefficient)
      LeftAuthority5074.bound (LeftAuthority5074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 5077 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5074.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5074.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority5074.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5078

namespace LeftBound5086
def owner : Owner := ⟨.program ⟨257⟩, ⟨45693⟩⟩
def transferEvent : Nat := 5086
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5084 .coefficient) (.predecessor 1 5085 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 5084 .coefficient)
      LeftAuthority5082.bound (LeftAuthority5082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact5083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5082.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 5085 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority5082.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5082.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority5082.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5086

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
