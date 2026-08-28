import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1100
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1160

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound174515
def owner : Owner := ⟨.program ⟨257⟩, ⟨44162⟩⟩
def transferEvent : Nat := 174515
def frameStart : Nat := 174459
def rule : BoundRule := .sum [.predecessor 0 174513 .coefficient, .predecessor 1 174514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174513 .coefficient)
      LeftBound174498.bound (LeftBound174498.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound174498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174514 .coefficient)
      LeftAuthority174511.bound (LeftAuthority174511.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority174511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound174498.bound, LeftAuthority174511.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174498.bound, LeftAuthority174511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound174498.actual selector witness, LeftAuthority174511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174515

namespace LeftBound174518
def owner : Owner := ⟨.program ⟨257⟩, ⟨44163⟩⟩
def transferEvent : Nat := 174518
def frameStart : Nat := 174459
def rule : BoundRule := .identity (.predecessor 0 174517 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174517 .coefficient)
      LeftBound174515.bound (LeftBound174515.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound174515.derived selector witness)

def rawBound : CoeffClass := LeftBound174515.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound174515.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound174518

namespace LeftBound174524
def owner : Owner := ⟨.program ⟨257⟩, ⟨44164⟩⟩
def transferEvent : Nat := 174524
def frameStart : Nat := 174459
def rule : BoundRule := .product (.predecessor 0 174522 .coefficient) (.predecessor 1 174523 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174522 .coefficient)
      LeftAuthority174520.bound (LeftAuthority174520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174523 .coefficient)
      LeftBound174518.bound (LeftBound174518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority174520.bound LeftBound174518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority174520.bound, LeftBound174518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority174520.actual selector witness) * (LeftBound174518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174524

namespace LeftBound174532
def owner : Owner := ⟨.program ⟨257⟩, ⟨44165⟩⟩
def transferEvent : Nat := 174532
def frameStart : Nat := 174459
def rule : BoundRule := .sum [.predecessor 0 174530 .coefficient, .predecessor 1 174531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174530 .coefficient)
      LeftAuthority174528.bound (LeftAuthority174528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174531 .coefficient)
      LeftBound174524.bound (LeftBound174524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority174528.bound, LeftBound174524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority174528.bound, LeftBound174524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority174528.actual selector witness, LeftBound174524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174532

namespace LeftBound174536
def owner : Owner := ⟨.program ⟨257⟩, ⟨44764⟩⟩
def transferEvent : Nat := 174536
def frameStart : Nat := 174459
def rule : BoundRule := .product (.predecessor 0 174534 .coefficient) (.predecessor 1 174535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174534 .coefficient)
      LeftBound174532.bound (LeftBound174532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174535 .coefficient)
      LeftAuthority174509.bound (LeftAuthority174509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound174532.bound LeftAuthority174509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174532.bound, LeftAuthority174509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound174532.actual selector witness) * (LeftAuthority174509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174536

namespace LeftBound174547
def owner : Owner := ⟨.program ⟨257⟩, ⟨43056⟩⟩
def transferEvent : Nat := 174547
def frameStart : Nat := 174459
def rule : BoundRule := .product (.predecessor 0 174545 .coefficient) (.predecessor 1 174546 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174545 .coefficient)
      LeftAuthority174520.bound (LeftAuthority174520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174520.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174546 .coefficient)
      LeftAuthority174543.bound (LeftAuthority174543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174543.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority174520.bound LeftAuthority174543.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority174520.bound, LeftAuthority174543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority174520.actual selector witness) * (LeftAuthority174543.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174547

namespace LeftBound174555
def owner : Owner := ⟨.program ⟨257⟩, ⟨43057⟩⟩
def transferEvent : Nat := 174555
def frameStart : Nat := 174459
def rule : BoundRule := .sum [.predecessor 0 174553 .coefficient, .predecessor 1 174554 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174553 .coefficient)
      LeftAuthority174551.bound (LeftAuthority174551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174554 .coefficient)
      LeftBound174547.bound (LeftBound174547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority174551.bound, LeftBound174547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority174551.bound, LeftBound174547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority174551.actual selector witness, LeftBound174547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174555

namespace LeftBound174559
def owner : Owner := ⟨.program ⟨257⟩, ⟨44768⟩⟩
def transferEvent : Nat := 174559
def frameStart : Nat := 174459
def rule : BoundRule := .sum [.predecessor 0 174557 .coefficient, .predecessor 1 174558 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174557 .coefficient)
      LeftBound174555.bound (LeftBound174555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174558 .coefficient)
      LeftBound174536.bound (LeftBound174536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound174555.bound, LeftBound174536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174555.bound, LeftBound174536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound174555.actual selector witness, LeftBound174536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174559

namespace LeftBound174572
def owner : Owner := ⟨.program ⟨257⟩, ⟨44766⟩⟩
def transferEvent : Nat := 174572
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 174570 .coefficient, .predecessor 1 174571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174570 .coefficient)
      LeftBound174401.bound (LeftBound174401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174571 .coefficient)
      LeftBound174384.bound (LeftBound174384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174384.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound174401.bound, LeftBound174384.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174401.bound, LeftBound174384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound174401.actual selector witness, LeftBound174384.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174572

namespace LeftBound174575
def owner : Owner := ⟨.program ⟨257⟩, ⟨44766⟩⟩
def transferEvent : Nat := 174575
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 174569 .summary, .result 174391 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174569 .summary)
      LeftBound174403.bound (LeftBound174403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43615⟩⟩) (rawTerms := some (Proof.Events681.exact174569RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound174403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174391 .summary)
      LeftBound174386.bound (LeftBound174386.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44765⟩⟩) (rawTerms := some (Proof.Events681.exact174391RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound174386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound174403.bound, LeftBound174386.bound]
def bound : CoeffClass := .finite ⟨32193718473625891320532869316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174403.bound, LeftBound174386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound174403.actual selector witness, LeftBound174386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound174575

namespace LeftBound174579
def owner : Owner := ⟨.program ⟨257⟩, ⟨44767⟩⟩
def transferEvent : Nat := 174579
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 174577 .coefficient) (.predecessor 1 174578 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174577 .coefficient)
      LeftBound174572.bound (LeftBound174572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events681.exact174576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound174572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound174572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174578 .coefficient)
      LeftBound15581.bound (LeftBound15581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15581.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound174572.bound LeftBound15581.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174572.bound, LeftBound15581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound174572.actual selector witness) * (LeftBound15581.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174579

namespace LeftBound174580
def owner : Owner := ⟨.program ⟨257⟩, ⟨44767⟩⟩
def transferEvent : Nat := 174580
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7153⟩⟩]⟩ [⟨.result 15578 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15578 .coefficient)
      LeftAuthority15577.bound (LeftAuthority15577.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7153⟩⟩) (rawTerms := some (Proof.Events060.exact15578RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15577.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15577.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15577.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound174580

namespace LeftBound174581
def owner : Owner := ⟨.program ⟨257⟩, ⟨44767⟩⟩
def transferEvent : Nat := 174581
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 174576 .summary) (.transfer 174580) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174576 .summary)
      LeftBound174575.bound (LeftBound174575.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44766⟩⟩) (rawTerms := some (Proof.Events681.exact174576RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound174575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 174580)
      LeftBound174580.bound (LeftBound174580.actual selector witness) := by
  exact .transfer (LeftBound174580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound174575.bound LeftBound174580.bound
def bound : CoeffClass := .finite ⟨345677419952135604401347317519683074129920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound174575.bound, LeftBound174580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound174575.actual selector witness) * (LeftBound174580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174581

namespace LeftBound174596
def owner : Owner := ⟨.program ⟨257⟩, ⟨42085⟩⟩
def transferEvent : Nat := 174596
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 174594 .coefficient) (.predecessor 1 174595 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 174594 .coefficient)
      LeftBound165373.bound (LeftBound165373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events646.exact165377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound165373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound165373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 174595 .coefficient)
      LeftAuthority174592.bound (LeftAuthority174592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events682.exact174593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound165373.bound LeftAuthority174592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound165373.bound, LeftAuthority174592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound165373.actual selector witness) * (LeftAuthority174592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174596

namespace LeftBound174597
def owner : Owner := ⟨.program ⟨257⟩, ⟨42085⟩⟩
def transferEvent : Nat := 174597
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨42083⟩⟩]⟩ [⟨.result 174593 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 174593 .coefficient)
      LeftAuthority174592.bound (LeftAuthority174592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨42083⟩⟩) (rawTerms := some (Proof.Events682.exact174593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority174592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority174592.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority174592.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority174592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority174592.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound174597

namespace LeftBound174598
def owner : Owner := ⟨.program ⟨257⟩, ⟨42085⟩⟩
def transferEvent : Nat := 174598
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 165377 .summary) (.transfer 174597) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 165377 .summary)
      LeftBound165376.bound (LeftBound165376.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41665⟩⟩) (rawTerms := some (Proof.Events646.exact165377RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound165376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 174597)
      LeftBound174597.bound (LeftBound174597.actual selector witness) := by
  exact .transfer (LeftBound174597.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound165376.bound LeftBound174597.bound
def bound : CoeffClass := .finite ⟨32193129122288627115968346193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound165376.bound, LeftBound174597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound165376.actual selector witness) * (LeftBound174597.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound174598

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
