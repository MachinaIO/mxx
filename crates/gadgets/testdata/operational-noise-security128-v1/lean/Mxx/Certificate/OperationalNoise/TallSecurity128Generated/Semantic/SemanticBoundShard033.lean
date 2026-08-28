import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard032

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10450
def owner : Owner := ⟨.program ⟨257⟩, ⟨16031⟩⟩
def transferEvent : Nat := 10450
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10448 .coefficient) (.predecessor 1 10449 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10448 .coefficient)
      LeftAuthority10446.bound (LeftAuthority10446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10449 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority10446.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10446.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority10446.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10450

namespace LeftBound10455
def owner : Owner := ⟨.program ⟨257⟩, ⟨16032⟩⟩
def transferEvent : Nat := 10455
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10453 .coefficient, .predecessor 1 10454 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10453 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10454 .coefficient)
      LeftBound10450.bound (LeftBound10450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10450.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound10450.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound10450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound10450.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10455

namespace LeftBound10459
def owner : Owner := ⟨.program ⟨257⟩, ⟨18863⟩⟩
def transferEvent : Nat := 10459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10457 .coefficient, .predecessor 1 10458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10457 .coefficient)
      LeftBound10455.bound (LeftBound10455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10458 .coefficient)
      LeftBound10442.bound (LeftBound10442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10455.bound, LeftBound10442.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10455.bound, LeftBound10442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10455.actual selector witness, LeftBound10442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10459

namespace LeftBound10463
def owner : Owner := ⟨.program ⟨257⟩, ⟨22083⟩⟩
def transferEvent : Nat := 10463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10461 .coefficient, .predecessor 1 10462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10461 .coefficient)
      LeftBound10459.bound (LeftBound10459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10462 .coefficient)
      LeftBound10434.bound (LeftBound10434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10434.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10459.bound, LeftBound10434.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10459.bound, LeftBound10434.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10459.actual selector witness, LeftBound10434.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10463

namespace LeftBound10467
def owner : Owner := ⟨.program ⟨257⟩, ⟨32103⟩⟩
def transferEvent : Nat := 10467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10465 .coefficient, .predecessor 1 10466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10465 .coefficient)
      LeftBound10463.bound (LeftBound10463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10466 .coefficient)
      LeftBound10426.bound (LeftBound10426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10463.bound, LeftBound10426.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10463.bound, LeftBound10426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10463.actual selector witness, LeftBound10426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10467

namespace LeftBound10471
def owner : Owner := ⟨.program ⟨257⟩, ⟨51167⟩⟩
def transferEvent : Nat := 10471
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10469 .coefficient, .predecessor 1 10470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10469 .coefficient)
      LeftBound10467.bound (LeftBound10467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10470 .coefficient)
      LeftBound10418.bound (LeftBound10418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10467.bound, LeftBound10418.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10467.bound, LeftBound10418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10467.actual selector witness, LeftBound10418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10471

namespace LeftBound10475
def owner : Owner := ⟨.program ⟨257⟩, ⟨54147⟩⟩
def transferEvent : Nat := 10475
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10473 .coefficient, .predecessor 1 10474 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10473 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10474 .coefficient)
      LeftBound10410.bound (LeftBound10410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10471.bound, LeftBound10410.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, LeftBound10410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10471.actual selector witness, LeftBound10410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10475

namespace LeftBound10479
def owner : Owner := ⟨.program ⟨257⟩, ⟨57127⟩⟩
def transferEvent : Nat := 10479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10477 .coefficient, .predecessor 1 10478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10477 .coefficient)
      LeftBound10475.bound (LeftBound10475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10478 .coefficient)
      LeftBound10402.bound (LeftBound10402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10475.bound, LeftBound10402.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10475.bound, LeftBound10402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10475.actual selector witness, LeftBound10402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10479

namespace LeftBound10483
def owner : Owner := ⟨.program ⟨257⟩, ⟨60107⟩⟩
def transferEvent : Nat := 10483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10481 .coefficient, .predecessor 1 10482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10481 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10482 .coefficient)
      LeftBound10394.bound (LeftBound10394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10479.bound, LeftBound10394.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10479.bound, LeftBound10394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10479.actual selector witness, LeftBound10394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10483

namespace LeftBound10487
def owner : Owner := ⟨.program ⟨257⟩, ⟨63087⟩⟩
def transferEvent : Nat := 10487
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10485 .coefficient, .predecessor 1 10486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10485 .coefficient)
      LeftBound10483.bound (LeftBound10483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10486 .coefficient)
      LeftBound10386.bound (LeftBound10386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10483.bound, LeftBound10386.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10483.bound, LeftBound10386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10483.actual selector witness, LeftBound10386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10487

namespace LeftBound10491
def owner : Owner := ⟨.program ⟨257⟩, ⟨66590⟩⟩
def transferEvent : Nat := 10491
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10489 .coefficient, .predecessor 1 10490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10489 .coefficient)
      LeftBound10487.bound (LeftBound10487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10490 .coefficient)
      LeftBound10378.bound (LeftBound10378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10487.bound, LeftBound10378.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10487.bound, LeftBound10378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10487.actual selector witness, LeftBound10378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10491

namespace LeftBound10495
def owner : Owner := ⟨.program ⟨257⟩, ⟨66591⟩⟩
def transferEvent : Nat := 10495
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10493 .coefficient, .predecessor 1 10494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10493 .coefficient)
      LeftBound10491.bound (LeftBound10491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10494 .coefficient)
      LeftBound10370.bound (LeftBound10370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10491.bound, LeftBound10370.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10491.bound, LeftBound10370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10491.actual selector witness, LeftBound10370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10495

namespace LeftBound10499
def owner : Owner := ⟨.program ⟨257⟩, ⟨66592⟩⟩
def transferEvent : Nat := 10499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10497 .coefficient, .predecessor 1 10498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10497 .coefficient)
      LeftBound10495.bound (LeftBound10495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10498 .coefficient)
      LeftBound10362.bound (LeftBound10362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10495.bound, LeftBound10362.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10495.bound, LeftBound10362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10495.actual selector witness, LeftBound10362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10499

namespace LeftBound10503
def owner : Owner := ⟨.program ⟨257⟩, ⟨66593⟩⟩
def transferEvent : Nat := 10503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10501 .coefficient, .predecessor 1 10502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10501 .coefficient)
      LeftBound10499.bound (LeftBound10499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10502 .coefficient)
      LeftBound10354.bound (LeftBound10354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10499.bound, LeftBound10354.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10499.bound, LeftBound10354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10499.actual selector witness, LeftBound10354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10503

namespace LeftBound10507
def owner : Owner := ⟨.program ⟨257⟩, ⟨66594⟩⟩
def transferEvent : Nat := 10507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10505 .coefficient, .predecessor 1 10506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10505 .coefficient)
      LeftBound10503.bound (LeftBound10503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10506 .coefficient)
      LeftBound10346.bound (LeftBound10346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10503.bound, LeftBound10346.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10503.bound, LeftBound10346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10503.actual selector witness, LeftBound10346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10507

namespace LeftBound10511
def owner : Owner := ⟨.program ⟨257⟩, ⟨66595⟩⟩
def transferEvent : Nat := 10511
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10509 .coefficient, .predecessor 1 10510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 10509 .coefficient)
      LeftBound10507.bound (LeftBound10507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 10510 .coefficient)
      LeftBound10338.bound (LeftBound10338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10507.bound, LeftBound10338.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10507.bound, LeftBound10338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound10507.actual selector witness, LeftBound10338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10511

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
