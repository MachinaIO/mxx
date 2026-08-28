import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard490

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound77546
def owner : Owner := ⟨.program ⟨257⟩, ⟨41410⟩⟩
def transferEvent : Nat := 77546
def frameStart : Nat := 77496
def rule : BoundRule := .sum [.predecessor 0 77544 .coefficient, .predecessor 1 77545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77544 .coefficient)
      LeftBound77529.bound (LeftBound77529.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77545 .coefficient)
      LeftAuthority77542.bound (LeftAuthority77542.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77542.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77529.bound, LeftAuthority77542.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77529.bound, LeftAuthority77542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77529.actual selector witness, LeftAuthority77542.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77546

namespace LeftBound77549
def owner : Owner := ⟨.program ⟨257⟩, ⟨41411⟩⟩
def transferEvent : Nat := 77549
def frameStart : Nat := 77496
def rule : BoundRule := .identity (.predecessor 0 77548 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77548 .coefficient)
      LeftBound77546.bound (LeftBound77546.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound77546.derived selector witness)

def rawBound : CoeffClass := LeftBound77546.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound77546.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77549

namespace LeftBound77555
def owner : Owner := ⟨.program ⟨257⟩, ⟨41412⟩⟩
def transferEvent : Nat := 77555
def frameStart : Nat := 77496
def rule : BoundRule := .product (.predecessor 0 77553 .coefficient) (.predecessor 1 77554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77553 .coefficient)
      LeftAuthority77551.bound (LeftAuthority77551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77554 .coefficient)
      LeftBound77549.bound (LeftBound77549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority77551.bound LeftBound77549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77551.bound, LeftBound77549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority77551.actual selector witness) * (LeftBound77549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77555

namespace LeftBound77571
def owner : Owner := ⟨.program ⟨257⟩, ⟨9557⟩⟩
def transferEvent : Nat := 77571
def frameStart : Nat := 77496
def rule : BoundRule := .scale (.predecessor 0 77569 .coefficient) (.value (.predecessor 1 77570 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77569 .coefficient)
      LeftAuthority77567.bound (LeftAuthority77567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77570 .coefficient)
      LeftAuthority77558.bound (LeftAuthority77558.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority77558.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority77567.bound LeftAuthority77558.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77567.bound, LeftAuthority77558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority77567.actual selector witness) * (LeftAuthority77558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound77571

namespace LeftBound77574
def owner : Owner := ⟨.program ⟨257⟩, ⟨7299⟩⟩
def transferEvent : Nat := 77574
def frameStart : Nat := 77496
def rule : BoundRule := .identity (.predecessor 0 77573 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77573 .coefficient)
      LeftAuthority77561.bound (LeftAuthority77561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77561.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77561.derived selector witness)

def rawBound : CoeffClass := LeftAuthority77561.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority77561.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound77574

namespace LeftBound77578
def owner : Owner := ⟨.program ⟨257⟩, ⟨9558⟩⟩
def transferEvent : Nat := 77578
def frameStart : Nat := 77496
def rule : BoundRule := .product (.predecessor 0 77576 .coefficient) (.predecessor 1 77577 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77576 .coefficient)
      LeftBound77574.bound (LeftBound77574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77577 .coefficient)
      LeftBound77571.bound (LeftBound77571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77571.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77574.bound LeftBound77571.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77574.bound, LeftBound77571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77574.actual selector witness) * (LeftBound77571.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77578

namespace LeftBound77583
def owner : Owner := ⟨.program ⟨257⟩, ⟨41413⟩⟩
def transferEvent : Nat := 77583
def frameStart : Nat := 77496
def rule : BoundRule := .sum [.predecessor 0 77581 .coefficient, .predecessor 1 77582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77581 .coefficient)
      LeftBound77578.bound (LeftBound77578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77582 .coefficient)
      LeftBound77555.bound (LeftBound77555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77578.bound, LeftBound77555.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77578.bound, LeftBound77555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77578.actual selector witness, LeftBound77555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77583

namespace LeftBound77587
def owner : Owner := ⟨.program ⟨257⟩, ⟨41688⟩⟩
def transferEvent : Nat := 77587
def frameStart : Nat := 77496
def rule : BoundRule := .product (.predecessor 0 77585 .coefficient) (.predecessor 1 77586 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77585 .coefficient)
      LeftBound77583.bound (LeftBound77583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77586 .coefficient)
      LeftAuthority77540.bound (LeftAuthority77540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77540.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77583.bound LeftAuthority77540.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77583.bound, LeftAuthority77540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77583.actual selector witness) * (LeftAuthority77540.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77587

namespace LeftBound77598
def owner : Owner := ⟨.program ⟨257⟩, ⟨40158⟩⟩
def transferEvent : Nat := 77598
def frameStart : Nat := 77496
def rule : BoundRule := .product (.predecessor 0 77596 .coefficient) (.predecessor 1 77597 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77596 .coefficient)
      LeftAuthority77551.bound (LeftAuthority77551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77597 .coefficient)
      LeftAuthority77594.bound (LeftAuthority77594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority77551.bound LeftAuthority77594.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77551.bound, LeftAuthority77594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority77551.actual selector witness) * (LeftAuthority77594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77598

namespace LeftBound77606
def owner : Owner := ⟨.program ⟨257⟩, ⟨40159⟩⟩
def transferEvent : Nat := 77606
def frameStart : Nat := 77496
def rule : BoundRule := .sum [.predecessor 0 77604 .coefficient, .predecessor 1 77605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77604 .coefficient)
      LeftAuthority77602.bound (LeftAuthority77602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77605 .coefficient)
      LeftBound77598.bound (LeftBound77598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority77602.bound, LeftBound77598.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77602.bound, LeftBound77598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority77602.actual selector witness, LeftBound77598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77606

namespace LeftBound77610
def owner : Owner := ⟨.program ⟨257⟩, ⟨41689⟩⟩
def transferEvent : Nat := 77610
def frameStart : Nat := 77496
def rule : BoundRule := .sum [.predecessor 0 77608 .coefficient, .predecessor 1 77609 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77608 .coefficient)
      LeftBound77606.bound (LeftBound77606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77609 .coefficient)
      LeftBound77587.bound (LeftBound77587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77606.bound, LeftBound77587.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77606.bound, LeftBound77587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77606.actual selector witness, LeftBound77587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77610

namespace LeftBound77623
def owner : Owner := ⟨.program ⟨257⟩, ⟨41687⟩⟩
def transferEvent : Nat := 77623
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 77621 .coefficient, .predecessor 1 77622 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77621 .coefficient)
      LeftBound77444.bound (LeftBound77444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77622 .coefficient)
      LeftBound77427.bound (LeftBound77427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77444.bound, LeftBound77427.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77444.bound, LeftBound77427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77444.actual selector witness, LeftBound77427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77623

namespace LeftBound77626
def owner : Owner := ⟨.program ⟨257⟩, ⟨41687⟩⟩
def transferEvent : Nat := 77626
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 77620 .summary, .result 77434 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77620 .summary)
      LeftBound77446.bound (LeftBound77446.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40612⟩⟩) (rawTerms := some (Proof.Events303.exact77620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77434 .summary)
      LeftBound77429.bound (LeftBound77429.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41686⟩⟩) (rawTerms := some (Proof.Events302.exact77434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound77446.bound, LeftBound77429.bound]
def bound : CoeffClass := .finite ⟨2998218789909838430208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77446.bound, LeftBound77429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound77446.actual selector witness, LeftBound77429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound77626

namespace LeftBound77630
def owner : Owner := ⟨.program ⟨257⟩, ⟨42141⟩⟩
def transferEvent : Nat := 77630
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 77628 .coefficient) (.predecessor 1 77629 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 77628 .coefficient)
      LeftBound77623.bound (LeftBound77623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events303.exact77627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound77623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound77623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 77629 .coefficient)
      LeftAuthority77349.bound (LeftAuthority77349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events302.exact77350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77349.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77623.bound LeftAuthority77349.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77623.bound, LeftAuthority77349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77623.actual selector witness) * (LeftAuthority77349.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77630

namespace LeftBound77631
def owner : Owner := ⟨.program ⟨257⟩, ⟨42141⟩⟩
def transferEvent : Nat := 77631
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨42139⟩⟩]⟩ [⟨.result 77350 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77350 .coefficient)
      LeftAuthority77349.bound (LeftAuthority77349.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42139⟩⟩) (rawTerms := some (Proof.Events302.exact77350RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority77349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority77349.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority77349.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority77349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority77349.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound77631

namespace LeftBound77632
def owner : Owner := ⟨.program ⟨257⟩, ⟨42141⟩⟩
def transferEvent : Nat := 77632
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 77627 .summary) (.transfer 77631) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 77627 .summary)
      LeftBound77626.bound (LeftBound77626.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41687⟩⟩) (rawTerms := some (Proof.Events303.exact77627RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound77626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 77631)
      LeftBound77631.bound (LeftBound77631.actual selector witness) := by
  exact .transfer (LeftBound77631.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound77626.bound LeftBound77631.bound
def bound : CoeffClass := .finite ⟨32193129122288627115968346193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound77626.bound, LeftBound77631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound77626.actual selector witness) * (LeftBound77631.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound77632

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
