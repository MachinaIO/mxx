import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100567
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 100567
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100565 .coefficient, .predecessor 1 100566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100565 .coefficient)
      LeftAuthority100563.bound (LeftAuthority100563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100566 .coefficient)
      LeftAuthority100560.bound (LeftAuthority100560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100563.bound, LeftAuthority100560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100563.bound, LeftAuthority100560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority100563.actual selector witness, LeftAuthority100560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100567

namespace LeftBound100571
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 100571
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100569 .coefficient, .predecessor 1 100570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100569 .coefficient)
      LeftBound100567.bound (LeftBound100567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100570 .coefficient)
      LeftAuthority100557.bound (LeftAuthority100557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100567.bound, LeftAuthority100557.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100567.bound, LeftAuthority100557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100567.actual selector witness, LeftAuthority100557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100571

namespace LeftBound100575
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 100575
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100573 .coefficient, .predecessor 1 100574 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100573 .coefficient)
      LeftBound100571.bound (LeftBound100571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100574 .coefficient)
      LeftAuthority100554.bound (LeftAuthority100554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100571.bound, LeftAuthority100554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100571.bound, LeftAuthority100554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100571.actual selector witness, LeftAuthority100554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100575

namespace LeftBound100579
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 100579
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100577 .coefficient, .predecessor 1 100578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100577 .coefficient)
      LeftBound100575.bound (LeftBound100575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100578 .coefficient)
      LeftAuthority100551.bound (LeftAuthority100551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100575.bound, LeftAuthority100551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100575.bound, LeftAuthority100551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100575.actual selector witness, LeftAuthority100551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100579

namespace LeftBound100583
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 100583
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100581 .coefficient, .predecessor 1 100582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100581 .coefficient)
      LeftBound100579.bound (LeftBound100579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100582 .coefficient)
      LeftAuthority100548.bound (LeftAuthority100548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100579.bound, LeftAuthority100548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100579.bound, LeftAuthority100548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100579.actual selector witness, LeftAuthority100548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100583

namespace LeftBound100587
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 100587
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100585 .coefficient, .predecessor 1 100586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100585 .coefficient)
      LeftBound100583.bound (LeftBound100583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100586 .coefficient)
      LeftAuthority100545.bound (LeftAuthority100545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100545.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100545.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100583.bound, LeftAuthority100545.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100583.bound, LeftAuthority100545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100583.actual selector witness, LeftAuthority100545.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100587

namespace LeftBound100591
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 100591
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100589 .coefficient, .predecessor 1 100590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100589 .coefficient)
      LeftBound100587.bound (LeftBound100587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100590 .coefficient)
      LeftAuthority100542.bound (LeftAuthority100542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100587.bound, LeftAuthority100542.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100587.bound, LeftAuthority100542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100587.actual selector witness, LeftAuthority100542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100591

namespace LeftBound100595
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 100595
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100593 .coefficient, .predecessor 1 100594 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100593 .coefficient)
      LeftBound100591.bound (LeftBound100591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100594 .coefficient)
      LeftAuthority100539.bound (LeftAuthority100539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100591.bound, LeftAuthority100539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100591.bound, LeftAuthority100539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100591.actual selector witness, LeftAuthority100539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100595

namespace LeftBound100599
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 100599
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100597 .coefficient, .predecessor 1 100598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100597 .coefficient)
      LeftBound100595.bound (LeftBound100595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100598 .coefficient)
      LeftAuthority100536.bound (LeftAuthority100536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100595.bound, LeftAuthority100536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100595.bound, LeftAuthority100536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100595.actual selector witness, LeftAuthority100536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100599

namespace LeftBound100603
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 100603
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100601 .coefficient, .predecessor 1 100602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100601 .coefficient)
      LeftBound100599.bound (LeftBound100599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100602 .coefficient)
      LeftAuthority100533.bound (LeftAuthority100533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100533.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100599.bound, LeftAuthority100533.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100599.bound, LeftAuthority100533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100599.actual selector witness, LeftAuthority100533.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100603

namespace LeftBound100607
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 100607
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100605 .coefficient, .predecessor 1 100606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100605 .coefficient)
      LeftBound100603.bound (LeftBound100603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100603.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100606 .coefficient)
      LeftAuthority100530.bound (LeftAuthority100530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100603.bound, LeftAuthority100530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100603.bound, LeftAuthority100530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100603.actual selector witness, LeftAuthority100530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100607

namespace LeftBound100611
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 100611
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100609 .coefficient, .predecessor 1 100610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100609 .coefficient)
      LeftBound100607.bound (LeftBound100607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100610 .coefficient)
      LeftAuthority100527.bound (LeftAuthority100527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100527.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100607.bound, LeftAuthority100527.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100607.bound, LeftAuthority100527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100607.actual selector witness, LeftAuthority100527.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100611

namespace LeftBound100615
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 100615
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100613 .coefficient, .predecessor 1 100614 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100613 .coefficient)
      LeftBound100611.bound (LeftBound100611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100614 .coefficient)
      LeftAuthority100524.bound (LeftAuthority100524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100611.bound, LeftAuthority100524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100611.bound, LeftAuthority100524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100611.actual selector witness, LeftAuthority100524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100615

namespace LeftBound100619
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 100619
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100617 .coefficient, .predecessor 1 100618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100617 .coefficient)
      LeftBound100615.bound (LeftBound100615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100618 .coefficient)
      LeftAuthority100521.bound (LeftAuthority100521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100615.bound, LeftAuthority100521.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100615.bound, LeftAuthority100521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100615.actual selector witness, LeftAuthority100521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100619

namespace LeftBound100623
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 100623
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100621 .coefficient, .predecessor 1 100622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100621 .coefficient)
      LeftBound100619.bound (LeftBound100619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100622 .coefficient)
      LeftAuthority100518.bound (LeftAuthority100518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100518.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100619.bound, LeftAuthority100518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100619.bound, LeftAuthority100518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100619.actual selector witness, LeftAuthority100518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100623

namespace LeftBound100627
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 100627
def frameStart : Nat := 99961
def rule : BoundRule := .sum [.predecessor 0 100625 .coefficient, .predecessor 1 100626 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 100625 .coefficient)
      LeftBound100623.bound (LeftBound100623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 100626 .coefficient)
      LeftAuthority100515.bound (LeftAuthority100515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100623.bound, LeftAuthority100515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100623.bound, LeftAuthority100515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound100623.actual selector witness, LeftAuthority100515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100627

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
