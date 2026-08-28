import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard198
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard251

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound43526
def owner : Owner := ⟨.program ⟨257⟩, ⟨36142⟩⟩
def transferEvent : Nat := 43526
def frameStart : Nat := 43470
def rule : BoundRule := .sum [.predecessor 0 43524 .coefficient, .predecessor 1 43525 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43524 .coefficient)
      LeftBound43509.bound (LeftBound43509.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43525 .coefficient)
      LeftAuthority43522.bound (LeftAuthority43522.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority43522.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43509.bound, LeftAuthority43522.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43509.bound, LeftAuthority43522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound43509.actual selector witness, LeftAuthority43522.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43526

namespace LeftBound43529
def owner : Owner := ⟨.program ⟨257⟩, ⟨36143⟩⟩
def transferEvent : Nat := 43529
def frameStart : Nat := 43470
def rule : BoundRule := .identity (.predecessor 0 43528 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43528 .coefficient)
      LeftBound43526.bound (LeftBound43526.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound43526.derived selector witness)

def rawBound : CoeffClass := LeftBound43526.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound43526.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound43529

namespace LeftBound43535
def owner : Owner := ⟨.program ⟨257⟩, ⟨36144⟩⟩
def transferEvent : Nat := 43535
def frameStart : Nat := 43470
def rule : BoundRule := .product (.predecessor 0 43533 .coefficient) (.predecessor 1 43534 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43533 .coefficient)
      LeftAuthority43531.bound (LeftAuthority43531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43531.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43534 .coefficient)
      LeftBound43529.bound (LeftBound43529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43529.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority43531.bound LeftBound43529.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43531.bound, LeftBound43529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority43531.actual selector witness) * (LeftBound43529.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43535

namespace LeftBound43543
def owner : Owner := ⟨.program ⟨257⟩, ⟨36145⟩⟩
def transferEvent : Nat := 43543
def frameStart : Nat := 43470
def rule : BoundRule := .sum [.predecessor 0 43541 .coefficient, .predecessor 1 43542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43541 .coefficient)
      LeftAuthority43539.bound (LeftAuthority43539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43542 .coefficient)
      LeftBound43535.bound (LeftBound43535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43535.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43539.bound, LeftBound43535.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43539.bound, LeftBound43535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority43539.actual selector witness, LeftBound43535.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43543

namespace LeftBound43547
def owner : Owner := ⟨.program ⟨257⟩, ⟨36849⟩⟩
def transferEvent : Nat := 43547
def frameStart : Nat := 43470
def rule : BoundRule := .product (.predecessor 0 43545 .coefficient) (.predecessor 1 43546 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43545 .coefficient)
      LeftBound43543.bound (LeftBound43543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43546 .coefficient)
      LeftAuthority43520.bound (LeftAuthority43520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound43543.bound LeftAuthority43520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43543.bound, LeftAuthority43520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound43543.actual selector witness) * (LeftAuthority43520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43547

namespace LeftBound43558
def owner : Owner := ⟨.program ⟨257⟩, ⟨35078⟩⟩
def transferEvent : Nat := 43558
def frameStart : Nat := 43470
def rule : BoundRule := .product (.predecessor 0 43556 .coefficient) (.predecessor 1 43557 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43556 .coefficient)
      LeftAuthority43531.bound (LeftAuthority43531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43531.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43557 .coefficient)
      LeftAuthority43554.bound (LeftAuthority43554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43554.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority43531.bound LeftAuthority43554.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43531.bound, LeftAuthority43554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority43531.actual selector witness) * (LeftAuthority43554.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43558

namespace LeftBound43566
def owner : Owner := ⟨.program ⟨257⟩, ⟨35079⟩⟩
def transferEvent : Nat := 43566
def frameStart : Nat := 43470
def rule : BoundRule := .sum [.predecessor 0 43564 .coefficient, .predecessor 1 43565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43564 .coefficient)
      LeftAuthority43562.bound (LeftAuthority43562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43565 .coefficient)
      LeftBound43558.bound (LeftBound43558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority43562.bound, LeftBound43558.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43562.bound, LeftBound43558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority43562.actual selector witness, LeftBound43558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43566

namespace LeftBound43570
def owner : Owner := ⟨.program ⟨257⟩, ⟨36853⟩⟩
def transferEvent : Nat := 43570
def frameStart : Nat := 43470
def rule : BoundRule := .sum [.predecessor 0 43568 .coefficient, .predecessor 1 43569 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43568 .coefficient)
      LeftBound43566.bound (LeftBound43566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43567RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43569 .coefficient)
      LeftBound43547.bound (LeftBound43547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43566.bound, LeftBound43547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43566.bound, LeftBound43547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound43566.actual selector witness, LeftBound43547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43570

namespace LeftBound43583
def owner : Owner := ⟨.program ⟨257⟩, ⟨36851⟩⟩
def transferEvent : Nat := 43583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 43581 .coefficient, .predecessor 1 43582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43581 .coefficient)
      LeftBound43412.bound (LeftBound43412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43582 .coefficient)
      LeftBound43395.bound (LeftBound43395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43395.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43412.bound, LeftBound43395.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43412.bound, LeftBound43395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound43412.actual selector witness, LeftBound43395.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43583

namespace LeftBound43586
def owner : Owner := ⟨.program ⟨257⟩, ⟨36851⟩⟩
def transferEvent : Nat := 43586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 43580 .summary, .result 43402 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43580 .summary)
      LeftBound43414.bound (LeftBound43414.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35675⟩⟩) (rawTerms := some (Proof.Events170.exact43580RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43414.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43402 .summary)
      LeftBound43397.bound (LeftBound43397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36850⟩⟩) (rawTerms := some (Proof.Events169.exact43402RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound43414.bound, LeftBound43397.bound]
def bound : CoeffClass := .finite ⟨32192539770951767057087530795008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43414.bound, LeftBound43397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound43414.actual selector witness, LeftBound43397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound43586

namespace LeftBound43590
def owner : Owner := ⟨.program ⟨257⟩, ⟨36852⟩⟩
def transferEvent : Nat := 43590
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43588 .coefficient) (.predecessor 1 43589 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43588 .coefficient)
      LeftBound43583.bound (LeftBound43583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43589 .coefficient)
      LeftBound15641.bound (LeftBound15641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound43583.bound LeftBound15641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43583.bound, LeftBound15641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound43583.actual selector witness) * (LeftBound15641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43590

namespace LeftBound43591
def owner : Owner := ⟨.program ⟨257⟩, ⟨36852⟩⟩
def transferEvent : Nat := 43591
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩ [⟨.result 15638 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15638 .coefficient)
      LeftAuthority15637.bound (LeftAuthority15637.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7163⟩⟩) (rawTerms := some (Proof.Events061.exact15638RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15637.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15637.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15637.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43591

namespace LeftBound43592
def owner : Owner := ⟨.program ⟨257⟩, ⟨36852⟩⟩
def transferEvent : Nat := 43592
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 43587 .summary) (.transfer 43591) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43587 .summary)
      LeftBound43586.bound (LeftBound43586.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36851⟩⟩) (rawTerms := some (Proof.Events170.exact43587RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 43591)
      LeftBound43591.bound (LeftBound43591.actual selector witness) := by
  exact .transfer (LeftBound43591.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound43586.bound LeftBound43591.bound
def bound : CoeffClass := .finite ⟨345664763728542925759002774434880600145920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound43586.bound, LeftBound43591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound43586.actual selector witness) * (LeftBound43591.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43592

namespace LeftBound43607
def owner : Owner := ⟨.program ⟨257⟩, ⟨31190⟩⟩
def transferEvent : Nat := 43607
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 43605 .coefficient) (.predecessor 1 43606 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 43605 .coefficient)
      LeftBound35194.bound (LeftBound35194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events137.exact35198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 43606 .coefficient)
      LeftAuthority43603.bound (LeftAuthority43603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43603.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound35194.bound LeftAuthority43603.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35194.bound, LeftAuthority43603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound35194.actual selector witness) * (LeftAuthority43603.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43607

namespace LeftBound43608
def owner : Owner := ⟨.program ⟨257⟩, ⟨31190⟩⟩
def transferEvent : Nat := 43608
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨31188⟩⟩]⟩ [⟨.result 43604 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43604 .coefficient)
      LeftAuthority43603.bound (LeftAuthority43603.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨31188⟩⟩) (rawTerms := some (Proof.Events170.exact43604RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority43603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority43603.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority43603.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority43603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority43603.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound43608

namespace LeftBound43609
def owner : Owner := ⟨.program ⟨257⟩, ⟨31190⟩⟩
def transferEvent : Nat := 43609
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35198 .summary) (.transfer 43608) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 35198 .summary)
      LeftBound35197.bound (LeftBound35197.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30700⟩⟩) (rawTerms := some (Proof.Events137.exact35198RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 43608)
      LeftBound43608.bound (LeftBound43608.actual selector witness) := by
  exact .transfer (LeftBound43608.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound35197.bound LeftBound43608.bound
def bound : CoeffClass := .finite ⟨32192146870060190229763897425920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35197.bound, LeftBound43608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound35197.actual selector witness) * (LeftBound43608.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound43609

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
