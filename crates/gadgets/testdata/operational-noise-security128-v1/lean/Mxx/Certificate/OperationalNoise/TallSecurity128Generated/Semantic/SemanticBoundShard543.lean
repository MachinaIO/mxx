import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard532
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard536
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard539
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard540
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard542

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84492
def owner : Owner := ⟨.program ⟨257⟩, ⟨15837⟩⟩
def transferEvent : Nat := 84492
def frameStart : Nat := 84453
def rule : BoundRule := .identity (.predecessor 0 84491 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84491 .coefficient)
      LeftAuthority84489.bound (LeftAuthority84489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84489.derived selector witness)

def rawBound : CoeffClass := LeftAuthority84489.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority84489.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84492

namespace LeftBound84509
def owner : Owner := ⟨.program ⟨257⟩, ⟨17230⟩⟩
def transferEvent : Nat := 84509
def frameStart : Nat := 84453
def rule : BoundRule := .sum [.predecessor 0 84507 .coefficient, .predecessor 1 84508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84507 .coefficient)
      LeftBound84492.bound (LeftBound84492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84508 .coefficient)
      LeftAuthority84505.bound (LeftAuthority84505.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84492.bound, LeftAuthority84505.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84492.bound, LeftAuthority84505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84492.actual selector witness, LeftAuthority84505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84509

namespace LeftBound84512
def owner : Owner := ⟨.program ⟨257⟩, ⟨17231⟩⟩
def transferEvent : Nat := 84512
def frameStart : Nat := 84453
def rule : BoundRule := .identity (.predecessor 0 84511 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84511 .coefficient)
      LeftBound84509.bound (LeftBound84509.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84509.derived selector witness)

def rawBound : CoeffClass := LeftBound84509.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound84509.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84512

namespace LeftBound84518
def owner : Owner := ⟨.program ⟨257⟩, ⟨17232⟩⟩
def transferEvent : Nat := 84518
def frameStart : Nat := 84453
def rule : BoundRule := .product (.predecessor 0 84516 .coefficient) (.predecessor 1 84517 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84516 .coefficient)
      LeftAuthority84514.bound (LeftAuthority84514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84517 .coefficient)
      LeftBound84512.bound (LeftBound84512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84512.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority84514.bound LeftBound84512.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84514.bound, LeftBound84512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority84514.actual selector witness) * (LeftBound84512.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84518

namespace LeftBound84526
def owner : Owner := ⟨.program ⟨257⟩, ⟨17233⟩⟩
def transferEvent : Nat := 84526
def frameStart : Nat := 84453
def rule : BoundRule := .sum [.predecessor 0 84524 .coefficient, .predecessor 1 84525 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84524 .coefficient)
      LeftAuthority84522.bound (LeftAuthority84522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84525 .coefficient)
      LeftBound84518.bound (LeftBound84518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84522.bound, LeftBound84518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84522.bound, LeftBound84518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority84522.actual selector witness, LeftBound84518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84526

namespace LeftBound84530
def owner : Owner := ⟨.program ⟨257⟩, ⟨17930⟩⟩
def transferEvent : Nat := 84530
def frameStart : Nat := 84453
def rule : BoundRule := .product (.predecessor 0 84528 .coefficient) (.predecessor 1 84529 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84528 .coefficient)
      LeftBound84526.bound (LeftBound84526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84529 .coefficient)
      LeftAuthority84503.bound (LeftAuthority84503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound84526.bound LeftAuthority84503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84526.bound, LeftAuthority84503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound84526.actual selector witness) * (LeftAuthority84503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84530

namespace LeftBound84541
def owner : Owner := ⟨.program ⟨257⟩, ⟨16132⟩⟩
def transferEvent : Nat := 84541
def frameStart : Nat := 84453
def rule : BoundRule := .product (.predecessor 0 84539 .coefficient) (.predecessor 1 84540 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84539 .coefficient)
      LeftAuthority84514.bound (LeftAuthority84514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84540 .coefficient)
      LeftAuthority84537.bound (LeftAuthority84537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84514.bound LeftAuthority84537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84514.bound, LeftAuthority84537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority84514.actual selector witness) * (LeftAuthority84537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84541

namespace LeftBound84549
def owner : Owner := ⟨.program ⟨257⟩, ⟨16133⟩⟩
def transferEvent : Nat := 84549
def frameStart : Nat := 84453
def rule : BoundRule := .sum [.predecessor 0 84547 .coefficient, .predecessor 1 84548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84547 .coefficient)
      LeftAuthority84545.bound (LeftAuthority84545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84545.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84548 .coefficient)
      LeftBound84541.bound (LeftBound84541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84545.bound, LeftBound84541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84545.bound, LeftBound84541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority84545.actual selector witness, LeftBound84541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84549

namespace LeftBound84553
def owner : Owner := ⟨.program ⟨257⟩, ⟨17933⟩⟩
def transferEvent : Nat := 84553
def frameStart : Nat := 84453
def rule : BoundRule := .sum [.predecessor 0 84551 .coefficient, .predecessor 1 84552 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84551 .coefficient)
      LeftBound84549.bound (LeftBound84549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84552 .coefficient)
      LeftBound84530.bound (LeftBound84530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84549.bound, LeftBound84530.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84549.bound, LeftBound84530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84549.actual selector witness, LeftBound84530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84553

namespace LeftBound84566
def owner : Owner := ⟨.program ⟨257⟩, ⟨17932⟩⟩
def transferEvent : Nat := 84566
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84564 .coefficient, .predecessor 1 84565 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84564 .coefficient)
      LeftBound84395.bound (LeftBound84395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84565 .coefficient)
      LeftBound84378.bound (LeftBound84378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events329.exact84385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84395.bound, LeftBound84378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84395.bound, LeftBound84378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84395.actual selector witness, LeftBound84378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84566

namespace LeftBound84569
def owner : Owner := ⟨.program ⟨257⟩, ⟨17932⟩⟩
def transferEvent : Nat := 84569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84563 .summary, .result 84385 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84563 .summary)
      LeftBound84397.bound (LeftBound84397.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16719⟩⟩) (rawTerms := some (Proof.Events330.exact84563RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84385 .summary)
      LeftBound84380.bound (LeftBound84380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17931⟩⟩) (rawTerms := some (Proof.Events329.exact84385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84397.bound, LeftBound84380.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84397.bound, LeftBound84380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84397.actual selector witness, LeftBound84380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84569

namespace LeftBound84573
def owner : Owner := ⟨.program ⟨257⟩, ⟨20842⟩⟩
def transferEvent : Nat := 84573
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84571 .coefficient, .predecessor 1 84572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84571 .coefficient)
      LeftBound84566.bound (LeftBound84566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84566.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84572 .coefficient)
      LeftBound84084.bound (LeftBound84084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events328.exact84088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84084.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84566.bound, LeftBound84084.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84566.bound, LeftBound84084.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84566.actual selector witness, LeftBound84084.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84573

namespace LeftBound84574
def owner : Owner := ⟨.program ⟨257⟩, ⟨20842⟩⟩
def transferEvent : Nat := 84574
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84570 .summary, .result 84088 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84570 .summary)
      LeftBound84569.bound (LeftBound84569.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17932⟩⟩) (rawTerms := some (Proof.Events330.exact84570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84088 .summary)
      LeftBound84087.bound (LeftBound84087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20841⟩⟩) (rawTerms := some (Proof.Events328.exact84088RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84569.bound, LeftBound84087.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84569.bound, LeftBound84087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84569.actual selector witness, LeftBound84087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84574

namespace LeftBound84578
def owner : Owner := ⟨.program ⟨257⟩, ⟨24062⟩⟩
def transferEvent : Nat := 84578
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84576 .coefficient, .predecessor 1 84577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84576 .coefficient)
      LeftBound84573.bound (LeftBound84573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84577 .coefficient)
      LeftBound83602.bound (LeftBound83602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events326.exact83606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83602.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83602.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84573.bound, LeftBound83602.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84573.bound, LeftBound83602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84573.actual selector witness, LeftBound83602.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84578

namespace LeftBound84579
def owner : Owner := ⟨.program ⟨257⟩, ⟨24062⟩⟩
def transferEvent : Nat := 84579
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84575 .summary, .result 83606 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 84575 .summary)
      LeftBound84574.bound (LeftBound84574.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20842⟩⟩) (rawTerms := some (Proof.Events330.exact84575RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 83606 .summary)
      LeftBound83605.bound (LeftBound83605.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24061⟩⟩) (rawTerms := some (Proof.Events326.exact83606RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound83605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84574.bound, LeftBound83605.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84574.bound, LeftBound83605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84574.actual selector witness, LeftBound83605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84579

namespace LeftBound84583
def owner : Owner := ⟨.program ⟨257⟩, ⟨34082⟩⟩
def transferEvent : Nat := 84583
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84581 .coefficient, .predecessor 1 84582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 84581 .coefficient)
      LeftBound84578.bound (LeftBound84578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 84582 .coefficient)
      LeftBound83120.bound (LeftBound83120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events324.exact83124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound83120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound83120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84578.bound, LeftBound83120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84578.bound, LeftBound83120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84578.actual selector witness, LeftBound83120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84583

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
