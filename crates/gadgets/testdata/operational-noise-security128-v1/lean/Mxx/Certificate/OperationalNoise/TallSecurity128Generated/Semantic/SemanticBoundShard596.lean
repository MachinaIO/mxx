import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard595

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92636
def owner : Owner := ⟨.program ⟨257⟩, ⟨37236⟩⟩
def transferEvent : Nat := 92636
def frameStart : Nat := 92603
def rule : BoundRule := .identity (.predecessor 0 92635 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92635 .coefficient)
      LeftBound92632.bound (LeftBound92632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92632.derived selector witness)

def rawBound : CoeffClass := LeftBound92632.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound92632.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92636

namespace LeftBound92653
def owner : Owner := ⟨.program ⟨257⟩, ⟨38726⟩⟩
def transferEvent : Nat := 92653
def frameStart : Nat := 92603
def rule : BoundRule := .sum [.predecessor 0 92651 .coefficient, .predecessor 1 92652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92651 .coefficient)
      LeftBound92636.bound (LeftBound92636.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92636.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92652 .coefficient)
      LeftAuthority92649.bound (LeftAuthority92649.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92636.bound, LeftAuthority92649.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92636.bound, LeftAuthority92649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92636.actual selector witness, LeftAuthority92649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92653

namespace LeftBound92656
def owner : Owner := ⟨.program ⟨257⟩, ⟨38727⟩⟩
def transferEvent : Nat := 92656
def frameStart : Nat := 92603
def rule : BoundRule := .identity (.predecessor 0 92655 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92655 .coefficient)
      LeftBound92653.bound (LeftBound92653.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92653.derived selector witness)

def rawBound : CoeffClass := LeftBound92653.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound92653.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92656

namespace LeftBound92662
def owner : Owner := ⟨.program ⟨257⟩, ⟨38728⟩⟩
def transferEvent : Nat := 92662
def frameStart : Nat := 92603
def rule : BoundRule := .product (.predecessor 0 92660 .coefficient) (.predecessor 1 92661 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92660 .coefficient)
      LeftAuthority92658.bound (LeftAuthority92658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92661 .coefficient)
      LeftBound92656.bound (LeftBound92656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92656.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority92658.bound LeftBound92656.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92658.bound, LeftBound92656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority92658.actual selector witness) * (LeftBound92656.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92662

namespace LeftBound92678
def owner : Owner := ⟨.program ⟨257⟩, ⟨9554⟩⟩
def transferEvent : Nat := 92678
def frameStart : Nat := 92603
def rule : BoundRule := .scale (.predecessor 0 92676 .coefficient) (.value (.predecessor 1 92677 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92676 .coefficient)
      LeftAuthority92674.bound (LeftAuthority92674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92677 .coefficient)
      LeftAuthority92665.bound (LeftAuthority92665.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92665.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92674.bound LeftAuthority92665.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92674.bound, LeftAuthority92665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority92674.actual selector witness) * (LeftAuthority92665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92678

namespace LeftBound92681
def owner : Owner := ⟨.program ⟨257⟩, ⟨7298⟩⟩
def transferEvent : Nat := 92681
def frameStart : Nat := 92603
def rule : BoundRule := .identity (.predecessor 0 92680 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92680 .coefficient)
      LeftAuthority92668.bound (LeftAuthority92668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92668.derived selector witness)

def rawBound : CoeffClass := LeftAuthority92668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority92668.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92681

namespace LeftBound92685
def owner : Owner := ⟨.program ⟨257⟩, ⟨9555⟩⟩
def transferEvent : Nat := 92685
def frameStart : Nat := 92603
def rule : BoundRule := .product (.predecessor 0 92683 .coefficient) (.predecessor 1 92684 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92683 .coefficient)
      LeftBound92681.bound (LeftBound92681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92684 .coefficient)
      LeftBound92678.bound (LeftBound92678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92681.bound LeftBound92678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92681.bound, LeftBound92678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92681.actual selector witness) * (LeftBound92678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92685

namespace LeftBound92690
def owner : Owner := ⟨.program ⟨257⟩, ⟨38729⟩⟩
def transferEvent : Nat := 92690
def frameStart : Nat := 92603
def rule : BoundRule := .sum [.predecessor 0 92688 .coefficient, .predecessor 1 92689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92688 .coefficient)
      LeftBound92685.bound (LeftBound92685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92685.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92689 .coefficient)
      LeftBound92662.bound (LeftBound92662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92685.bound, LeftBound92662.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92685.bound, LeftBound92662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92685.actual selector witness, LeftBound92662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92690

namespace LeftBound92694
def owner : Owner := ⟨.program ⟨257⟩, ⟨38997⟩⟩
def transferEvent : Nat := 92694
def frameStart : Nat := 92603
def rule : BoundRule := .product (.predecessor 0 92692 .coefficient) (.predecessor 1 92693 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92692 .coefficient)
      LeftBound92690.bound (LeftBound92690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92693 .coefficient)
      LeftAuthority92647.bound (LeftAuthority92647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92647.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92647.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92690.bound LeftAuthority92647.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92690.bound, LeftAuthority92647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92690.actual selector witness) * (LeftAuthority92647.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92694

namespace LeftBound92705
def owner : Owner := ⟨.program ⟨257⟩, ⟨37470⟩⟩
def transferEvent : Nat := 92705
def frameStart : Nat := 92603
def rule : BoundRule := .product (.predecessor 0 92703 .coefficient) (.predecessor 1 92704 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92703 .coefficient)
      LeftAuthority92658.bound (LeftAuthority92658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92704 .coefficient)
      LeftAuthority92701.bound (LeftAuthority92701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92658.bound LeftAuthority92701.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92658.bound, LeftAuthority92701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority92658.actual selector witness) * (LeftAuthority92701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92705

namespace LeftBound92713
def owner : Owner := ⟨.program ⟨257⟩, ⟨37471⟩⟩
def transferEvent : Nat := 92713
def frameStart : Nat := 92603
def rule : BoundRule := .sum [.predecessor 0 92711 .coefficient, .predecessor 1 92712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92711 .coefficient)
      LeftAuthority92709.bound (LeftAuthority92709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92712 .coefficient)
      LeftBound92705.bound (LeftBound92705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92709.bound, LeftBound92705.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92709.bound, LeftBound92705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority92709.actual selector witness, LeftBound92705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92713

namespace LeftBound92717
def owner : Owner := ⟨.program ⟨257⟩, ⟨38998⟩⟩
def transferEvent : Nat := 92717
def frameStart : Nat := 92603
def rule : BoundRule := .sum [.predecessor 0 92715 .coefficient, .predecessor 1 92716 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92715 .coefficient)
      LeftBound92713.bound (LeftBound92713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92716 .coefficient)
      LeftBound92694.bound (LeftBound92694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92713.bound, LeftBound92694.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92713.bound, LeftBound92694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92713.actual selector witness, LeftBound92694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92717

namespace LeftBound92730
def owner : Owner := ⟨.program ⟨257⟩, ⟨38996⟩⟩
def transferEvent : Nat := 92730
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92728 .coefficient, .predecessor 1 92729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92728 .coefficient)
      LeftBound92551.bound (LeftBound92551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92729 .coefficient)
      LeftBound92534.bound (LeftBound92534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92551.bound, LeftBound92534.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92551.bound, LeftBound92534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92551.actual selector witness, LeftBound92534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92730

namespace LeftBound92733
def owner : Owner := ⟨.program ⟨257⟩, ⟨38996⟩⟩
def transferEvent : Nat := 92733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92727 .summary, .result 92541 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92727 .summary)
      LeftBound92553.bound (LeftBound92553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨37922⟩⟩) (rawTerms := some (Proof.Events362.exact92727RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92541 .summary)
      LeftBound92536.bound (LeftBound92536.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38995⟩⟩) (rawTerms := some (Proof.Events361.exact92541RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92553.bound, LeftBound92536.bound]
def bound : CoeffClass := .finite ⟨2998182198162866044928, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92553.bound, LeftBound92536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound92553.actual selector witness, LeftBound92536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92733

namespace LeftBound92737
def owner : Owner := ⟨.program ⟨257⟩, ⟨39436⟩⟩
def transferEvent : Nat := 92737
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92735 .coefficient) (.predecessor 1 92736 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 92735 .coefficient)
      LeftBound92730.bound (LeftBound92730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 92736 .coefficient)
      LeftAuthority92456.bound (LeftAuthority92456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound92730.bound LeftAuthority92456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92730.bound, LeftAuthority92456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound92730.actual selector witness) * (LeftAuthority92456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92737

namespace LeftBound92738
def owner : Owner := ⟨.program ⟨257⟩, ⟨39436⟩⟩
def transferEvent : Nat := 92738
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39434⟩⟩]⟩ [⟨.result 92457 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 92457 .coefficient)
      LeftAuthority92456.bound (LeftAuthority92456.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39434⟩⟩) (rawTerms := some (Proof.Events361.exact92457RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92456.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92456.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority92456.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92738

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
