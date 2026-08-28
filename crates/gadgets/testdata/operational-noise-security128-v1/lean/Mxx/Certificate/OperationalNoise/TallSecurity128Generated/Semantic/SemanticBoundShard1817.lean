import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1816

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound268638
def owner : Owner := ⟨.program ⟨257⟩, ⟨35995⟩⟩
def transferEvent : Nat := 268638
def frameStart : Nat := 268585
def rule : BoundRule := .identity (.predecessor 0 268637 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268637 .coefficient)
      LeftBound268635.bound (LeftBound268635.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound268635.derived selector witness)

def rawBound : CoeffClass := LeftBound268635.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound268635.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound268638

namespace LeftBound268644
def owner : Owner := ⟨.program ⟨257⟩, ⟨35996⟩⟩
def transferEvent : Nat := 268644
def frameStart : Nat := 268585
def rule : BoundRule := .product (.predecessor 0 268642 .coefficient) (.predecessor 1 268643 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268642 .coefficient)
      LeftAuthority268640.bound (LeftAuthority268640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268643 .coefficient)
      LeftBound268638.bound (LeftBound268638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority268640.bound LeftBound268638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268640.bound, LeftBound268638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority268640.actual selector witness) * (LeftBound268638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268644

namespace LeftBound268660
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 268660
def frameStart : Nat := 268585
def rule : BoundRule := .scale (.predecessor 0 268658 .coefficient) (.value (.predecessor 1 268659 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268658 .coefficient)
      LeftAuthority268656.bound (LeftAuthority268656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268656.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268659 .coefficient)
      LeftAuthority268647.bound (LeftAuthority268647.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority268647.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority268656.bound LeftAuthority268647.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268656.bound, LeftAuthority268647.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority268656.actual selector witness) * (LeftAuthority268647.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound268660

namespace LeftBound268663
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 268663
def frameStart : Nat := 268585
def rule : BoundRule := .identity (.predecessor 0 268662 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268662 .coefficient)
      LeftAuthority268650.bound (LeftAuthority268650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268650.derived selector witness)

def rawBound : CoeffClass := LeftAuthority268650.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority268650.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound268663

namespace LeftBound268667
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 268667
def frameStart : Nat := 268585
def rule : BoundRule := .product (.predecessor 0 268665 .coefficient) (.predecessor 1 268666 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268665 .coefficient)
      LeftBound268663.bound (LeftBound268663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268666 .coefficient)
      LeftBound268660.bound (LeftBound268660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268663.bound LeftBound268660.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268663.bound, LeftBound268660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268663.actual selector witness) * (LeftBound268660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268667

namespace LeftBound268672
def owner : Owner := ⟨.program ⟨257⟩, ⟨35997⟩⟩
def transferEvent : Nat := 268672
def frameStart : Nat := 268585
def rule : BoundRule := .sum [.predecessor 0 268670 .coefficient, .predecessor 1 268671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268670 .coefficient)
      LeftBound268667.bound (LeftBound268667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268671 .coefficient)
      LeftBound268644.bound (LeftBound268644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268644.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound268667.bound, LeftBound268644.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268667.bound, LeftBound268644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound268667.actual selector witness, LeftBound268644.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound268672

namespace LeftBound268676
def owner : Owner := ⟨.program ⟨257⟩, ⟨36171⟩⟩
def transferEvent : Nat := 268676
def frameStart : Nat := 268585
def rule : BoundRule := .product (.predecessor 0 268674 .coefficient) (.predecessor 1 268675 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268674 .coefficient)
      LeftBound268672.bound (LeftBound268672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268675 .coefficient)
      LeftAuthority268629.bound (LeftAuthority268629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268629.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268672.bound LeftAuthority268629.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268672.bound, LeftAuthority268629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268672.actual selector witness) * (LeftAuthority268629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268676

namespace LeftBound268687
def owner : Owner := ⟨.program ⟨257⟩, ⟨34684⟩⟩
def transferEvent : Nat := 268687
def frameStart : Nat := 268585
def rule : BoundRule := .product (.predecessor 0 268685 .coefficient) (.predecessor 1 268686 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268685 .coefficient)
      LeftAuthority268640.bound (LeftAuthority268640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268686 .coefficient)
      LeftAuthority268683.bound (LeftAuthority268683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority268640.bound LeftAuthority268683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268640.bound, LeftAuthority268683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority268640.actual selector witness) * (LeftAuthority268683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268687

namespace LeftBound268695
def owner : Owner := ⟨.program ⟨257⟩, ⟨34685⟩⟩
def transferEvent : Nat := 268695
def frameStart : Nat := 268585
def rule : BoundRule := .sum [.predecessor 0 268693 .coefficient, .predecessor 1 268694 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268693 .coefficient)
      LeftAuthority268691.bound (LeftAuthority268691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268694 .coefficient)
      LeftBound268687.bound (LeftBound268687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority268691.bound, LeftBound268687.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268691.bound, LeftBound268687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority268691.actual selector witness, LeftBound268687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound268695

namespace LeftBound268699
def owner : Owner := ⟨.program ⟨257⟩, ⟨36172⟩⟩
def transferEvent : Nat := 268699
def frameStart : Nat := 268585
def rule : BoundRule := .sum [.predecessor 0 268697 .coefficient, .predecessor 1 268698 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268697 .coefficient)
      LeftBound268695.bound (LeftBound268695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268698 .coefficient)
      LeftBound268676.bound (LeftBound268676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound268695.bound, LeftBound268676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268695.bound, LeftBound268676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound268695.actual selector witness, LeftBound268676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound268699

namespace LeftBound268712
def owner : Owner := ⟨.program ⟨257⟩, ⟨36170⟩⟩
def transferEvent : Nat := 268712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 268710 .coefficient, .predecessor 1 268711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268710 .coefficient)
      LeftBound268533.bound (LeftBound268533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268711 .coefficient)
      LeftBound268516.bound (LeftBound268516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1048.exact268523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound268533.bound, LeftBound268516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268533.bound, LeftBound268516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound268533.actual selector witness, LeftBound268516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound268712

namespace LeftBound268715
def owner : Owner := ⟨.program ⟨257⟩, ⟨36170⟩⟩
def transferEvent : Nat := 268715
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 268709 .summary, .result 268523 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268709 .summary)
      LeftBound268535.bound (LeftBound268535.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35109⟩⟩) (rawTerms := some (Proof.Events1049.exact268709RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268523 .summary)
      LeftBound268518.bound (LeftBound268518.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36169⟩⟩) (rawTerms := some (Proof.Events1048.exact268523RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound268535.bound, LeftBound268518.bound]
def bound : CoeffClass := .finite ⟨2998163902289379852288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268535.bound, LeftBound268518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound268535.actual selector witness, LeftBound268518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound268715

namespace LeftBound268719
def owner : Owner := ⟨.program ⟨257⟩, ⟨36424⟩⟩
def transferEvent : Nat := 268719
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 268717 .coefficient) (.predecessor 1 268718 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268717 .coefficient)
      LeftBound268712.bound (LeftBound268712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound268712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound268712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268718 .coefficient)
      LeftAuthority268438.bound (LeftAuthority268438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1048.exact268439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268438.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268712.bound LeftAuthority268438.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268712.bound, LeftAuthority268438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268712.actual selector witness) * (LeftAuthority268438.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268719

namespace LeftBound268720
def owner : Owner := ⟨.program ⟨257⟩, ⟨36424⟩⟩
def transferEvent : Nat := 268720
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36422⟩⟩]⟩ [⟨.result 268439 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268439 .coefficient)
      LeftAuthority268438.bound (LeftAuthority268438.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36422⟩⟩) (rawTerms := some (Proof.Events1048.exact268439RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268438.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority268438.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority268438.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound268720

namespace LeftBound268721
def owner : Owner := ⟨.program ⟨257⟩, ⟨36424⟩⟩
def transferEvent : Nat := 268721
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 268716 .summary) (.transfer 268720) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 268716 .summary)
      LeftBound268715.bound (LeftBound268715.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36170⟩⟩) (rawTerms := some (Proof.Events1049.exact268716RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound268715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 268720)
      LeftBound268720.bound (LeftBound268720.actual selector witness) := by
  exact .transfer (LeftBound268720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound268715.bound LeftBound268720.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound268715.bound, LeftBound268720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound268715.actual selector witness) * (LeftBound268720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound268721

namespace LeftBound268732
def owner : Owner := ⟨.program ⟨257⟩, ⟨35332⟩⟩
def transferEvent : Nat := 268732
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 268730 .coefficient) (.value (.predecessor 1 268731 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 268730 .coefficient)
      LeftAuthority268728.bound (LeftAuthority268728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1049.exact268729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority268728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority268728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 268731 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority268728.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority268728.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority268728.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound268732

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
