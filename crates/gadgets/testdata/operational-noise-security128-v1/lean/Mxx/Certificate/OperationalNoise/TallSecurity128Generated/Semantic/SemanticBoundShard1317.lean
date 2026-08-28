import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1316

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound196474
def owner : Owner := ⟨.program ⟨257⟩, ⟨27694⟩⟩
def transferEvent : Nat := 196474
def frameStart : Nat := 196424
def rule : BoundRule := .sum [.predecessor 0 196472 .coefficient, .predecessor 1 196473 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196472 .coefficient)
      LeftBound196457.bound (LeftBound196457.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound196457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196473 .coefficient)
      LeftAuthority196470.bound (LeftAuthority196470.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority196470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound196457.bound, LeftAuthority196470.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196457.bound, LeftAuthority196470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound196457.actual selector witness, LeftAuthority196470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196474

namespace LeftBound196477
def owner : Owner := ⟨.program ⟨257⟩, ⟨27695⟩⟩
def transferEvent : Nat := 196477
def frameStart : Nat := 196424
def rule : BoundRule := .identity (.predecessor 0 196476 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196476 .coefficient)
      LeftBound196474.bound (LeftBound196474.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound196474.derived selector witness)

def rawBound : CoeffClass := LeftBound196474.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound196474.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound196477

namespace LeftBound196483
def owner : Owner := ⟨.program ⟨257⟩, ⟨27696⟩⟩
def transferEvent : Nat := 196483
def frameStart : Nat := 196424
def rule : BoundRule := .product (.predecessor 0 196481 .coefficient) (.predecessor 1 196482 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196481 .coefficient)
      LeftAuthority196479.bound (LeftAuthority196479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196482 .coefficient)
      LeftBound196477.bound (LeftBound196477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority196479.bound LeftBound196477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196479.bound, LeftBound196477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority196479.actual selector witness) * (LeftBound196477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196483

namespace LeftBound196499
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 196499
def frameStart : Nat := 196424
def rule : BoundRule := .scale (.predecessor 0 196497 .coefficient) (.value (.predecessor 1 196498 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196497 .coefficient)
      LeftAuthority196495.bound (LeftAuthority196495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196495.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196498 .coefficient)
      LeftAuthority196486.bound (LeftAuthority196486.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority196486.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority196495.bound LeftAuthority196486.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196495.bound, LeftAuthority196486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority196495.actual selector witness) * (LeftAuthority196486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound196499

namespace LeftBound196502
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 196502
def frameStart : Nat := 196424
def rule : BoundRule := .identity (.predecessor 0 196501 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196501 .coefficient)
      LeftAuthority196489.bound (LeftAuthority196489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196489.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196489.derived selector witness)

def rawBound : CoeffClass := LeftAuthority196489.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority196489.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound196502

namespace LeftBound196506
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 196506
def frameStart : Nat := 196424
def rule : BoundRule := .product (.predecessor 0 196504 .coefficient) (.predecessor 1 196505 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196504 .coefficient)
      LeftBound196502.bound (LeftBound196502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196502.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196505 .coefficient)
      LeftBound196499.bound (LeftBound196499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196499.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound196502.bound LeftBound196499.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196502.bound, LeftBound196499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound196502.actual selector witness) * (LeftBound196499.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196506

namespace LeftBound196511
def owner : Owner := ⟨.program ⟨257⟩, ⟨27697⟩⟩
def transferEvent : Nat := 196511
def frameStart : Nat := 196424
def rule : BoundRule := .sum [.predecessor 0 196509 .coefficient, .predecessor 1 196510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196509 .coefficient)
      LeftBound196506.bound (LeftBound196506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196506.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196510 .coefficient)
      LeftBound196483.bound (LeftBound196483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound196506.bound, LeftBound196483.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196506.bound, LeftBound196483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound196506.actual selector witness, LeftBound196483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196511

namespace LeftBound196515
def owner : Owner := ⟨.program ⟨257⟩, ⟨27944⟩⟩
def transferEvent : Nat := 196515
def frameStart : Nat := 196424
def rule : BoundRule := .product (.predecessor 0 196513 .coefficient) (.predecessor 1 196514 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196513 .coefficient)
      LeftBound196511.bound (LeftBound196511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196514 .coefficient)
      LeftAuthority196468.bound (LeftAuthority196468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound196511.bound LeftAuthority196468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196511.bound, LeftAuthority196468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound196511.actual selector witness) * (LeftAuthority196468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196515

namespace LeftBound196526
def owner : Owner := ⟨.program ⟨257⟩, ⟨26426⟩⟩
def transferEvent : Nat := 196526
def frameStart : Nat := 196424
def rule : BoundRule := .product (.predecessor 0 196524 .coefficient) (.predecessor 1 196525 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196524 .coefficient)
      LeftAuthority196479.bound (LeftAuthority196479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196479.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196525 .coefficient)
      LeftAuthority196522.bound (LeftAuthority196522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196522.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196522.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority196479.bound LeftAuthority196522.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196479.bound, LeftAuthority196522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority196479.actual selector witness) * (LeftAuthority196522.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196526

namespace LeftBound196534
def owner : Owner := ⟨.program ⟨257⟩, ⟨26427⟩⟩
def transferEvent : Nat := 196534
def frameStart : Nat := 196424
def rule : BoundRule := .sum [.predecessor 0 196532 .coefficient, .predecessor 1 196533 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196532 .coefficient)
      LeftAuthority196530.bound (LeftAuthority196530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196533 .coefficient)
      LeftBound196526.bound (LeftBound196526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority196530.bound, LeftBound196526.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196530.bound, LeftBound196526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority196530.actual selector witness, LeftBound196526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196534

namespace LeftBound196538
def owner : Owner := ⟨.program ⟨257⟩, ⟨27945⟩⟩
def transferEvent : Nat := 196538
def frameStart : Nat := 196424
def rule : BoundRule := .sum [.predecessor 0 196536 .coefficient, .predecessor 1 196537 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196536 .coefficient)
      LeftBound196534.bound (LeftBound196534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196537 .coefficient)
      LeftBound196515.bound (LeftBound196515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound196534.bound, LeftBound196515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196534.bound, LeftBound196515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound196534.actual selector witness, LeftBound196515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196538

namespace LeftBound196551
def owner : Owner := ⟨.program ⟨257⟩, ⟨27943⟩⟩
def transferEvent : Nat := 196551
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 196549 .coefficient, .predecessor 1 196550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196549 .coefficient)
      LeftBound196372.bound (LeftBound196372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196550 .coefficient)
      LeftBound196355.bound (LeftBound196355.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196355.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound196372.bound, LeftBound196355.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196372.bound, LeftBound196355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound196372.actual selector witness, LeftBound196355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196551

namespace LeftBound196554
def owner : Owner := ⟨.program ⟨257⟩, ⟨27943⟩⟩
def transferEvent : Nat := 196554
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 196548 .summary, .result 196362 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196548 .summary)
      LeftBound196374.bound (LeftBound196374.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26872⟩⟩) (rawTerms := some (Proof.Events767.exact196548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound196374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196362 .summary)
      LeftBound196357.bound (LeftBound196357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27942⟩⟩) (rawTerms := some (Proof.Events767.exact196362RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound196357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound196374.bound, LeftBound196357.bound]
def bound : CoeffClass := .finite ⟨2998072422921948889088, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196374.bound, LeftBound196357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound196374.actual selector witness, LeftBound196357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound196554

namespace LeftBound196558
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def transferEvent : Nat := 196558
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 196556 .coefficient) (.predecessor 1 196557 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 196556 .coefficient)
      LeftBound196551.bound (LeftBound196551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events767.exact196555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound196551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound196551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 196557 .coefficient)
      LeftAuthority196277.bound (LeftAuthority196277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events766.exact196278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196277.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound196551.bound LeftAuthority196277.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196551.bound, LeftAuthority196277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound196551.actual selector witness) * (LeftAuthority196277.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196558

namespace LeftBound196559
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def transferEvent : Nat := 196559
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28339⟩⟩]⟩ [⟨.result 196278 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196278 .coefficient)
      LeftAuthority196277.bound (LeftAuthority196277.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28339⟩⟩) (rawTerms := some (Proof.Events766.exact196278RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority196277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority196277.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority196277.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority196277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority196277.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound196559

namespace LeftBound196560
def owner : Owner := ⟨.program ⟨257⟩, ⟨28341⟩⟩
def transferEvent : Nat := 196560
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 196555 .summary) (.transfer 196559) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 196555 .summary)
      LeftBound196554.bound (LeftBound196554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27943⟩⟩) (rawTerms := some (Proof.Events767.exact196555RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound196554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 196559)
      LeftBound196559.bound (LeftBound196559.actual selector witness) := by
  exact .transfer (LeftBound196559.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound196554.bound LeftBound196559.bound
def bound : CoeffClass := .finite ⟨32191557518723128098041228165120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound196554.bound, LeftBound196559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound196554.actual selector witness) * (LeftBound196559.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound196560

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
