import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2027

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound298776
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 298776
def frameStart : Nat := 298710
def rule : BoundRule := .identity (.predecessor 0 298775 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298775 .coefficient)
      LeftAuthority298763.bound (LeftAuthority298763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298763.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298763.derived selector witness)

def rawBound : CoeffClass := LeftAuthority298763.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority298763.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound298776

namespace LeftBound298780
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 298780
def frameStart : Nat := 298710
def rule : BoundRule := .product (.predecessor 0 298778 .coefficient) (.predecessor 1 298779 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298778 .coefficient)
      LeftBound298776.bound (LeftBound298776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298779 .coefficient)
      LeftBound298773.bound (LeftBound298773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298776.bound LeftBound298773.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298776.bound, LeftBound298773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298776.actual selector witness) * (LeftBound298773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298780

namespace LeftBound298785
def owner : Owner := ⟨.program ⟨257⟩, ⟨68890⟩⟩
def transferEvent : Nat := 298785
def frameStart : Nat := 298710
def rule : BoundRule := .sum [.predecessor 0 298783 .coefficient, .predecessor 1 298784 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298783 .coefficient)
      LeftBound298780.bound (LeftBound298780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298780.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298784 .coefficient)
      LeftBound298757.bound (LeftBound298757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298757.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298780.bound, LeftBound298757.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298780.bound, LeftBound298757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298780.actual selector witness, LeftBound298757.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298785

namespace LeftBound298789
def owner : Owner := ⟨.program ⟨257⟩, ⟨69133⟩⟩
def transferEvent : Nat := 298789
def frameStart : Nat := 298710
def rule : BoundRule := .product (.predecessor 0 298787 .coefficient) (.predecessor 1 298788 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298787 .coefficient)
      LeftBound298785.bound (LeftBound298785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298788 .coefficient)
      LeftAuthority298742.bound (LeftAuthority298742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1166.exact298743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298742.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298785.bound LeftAuthority298742.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298785.bound, LeftAuthority298742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298785.actual selector witness) * (LeftAuthority298742.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298789

namespace LeftBound298800
def owner : Owner := ⟨.program ⟨257⟩, ⟨65710⟩⟩
def transferEvent : Nat := 298800
def frameStart : Nat := 298710
def rule : BoundRule := .product (.predecessor 0 298798 .coefficient) (.predecessor 1 298799 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298798 .coefficient)
      LeftAuthority298753.bound (LeftAuthority298753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298799 .coefficient)
      LeftAuthority298796.bound (LeftAuthority298796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298796.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298796.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority298753.bound LeftAuthority298796.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298753.bound, LeftAuthority298796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority298753.actual selector witness) * (LeftAuthority298796.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298800

namespace LeftBound298808
def owner : Owner := ⟨.program ⟨257⟩, ⟨65711⟩⟩
def transferEvent : Nat := 298808
def frameStart : Nat := 298710
def rule : BoundRule := .sum [.predecessor 0 298806 .coefficient, .predecessor 1 298807 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298806 .coefficient)
      LeftAuthority298804.bound (LeftAuthority298804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298807 .coefficient)
      LeftBound298800.bound (LeftBound298800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority298804.bound, LeftBound298800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298804.bound, LeftBound298800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority298804.actual selector witness, LeftBound298800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298808

namespace LeftBound298812
def owner : Owner := ⟨.program ⟨257⟩, ⟨69134⟩⟩
def transferEvent : Nat := 298812
def frameStart : Nat := 298710
def rule : BoundRule := .sum [.predecessor 0 298810 .coefficient, .predecessor 1 298811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298810 .coefficient)
      LeftBound298808.bound (LeftBound298808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298811 .coefficient)
      LeftBound298789.bound (LeftBound298789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298808.bound, LeftBound298789.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298808.bound, LeftBound298789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298808.actual selector witness, LeftBound298789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298812

namespace LeftBound298825
def owner : Owner := ⟨.program ⟨257⟩, ⟨69132⟩⟩
def transferEvent : Nat := 298825
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 298823 .coefficient, .predecessor 1 298824 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298823 .coefficient)
      LeftBound298670.bound (LeftBound298670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298824 .coefficient)
      LeftBound298653.bound (LeftBound298653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1166.exact298660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298653.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298670.bound, LeftBound298653.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298670.bound, LeftBound298653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298670.actual selector witness, LeftBound298653.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298825

namespace LeftBound298828
def owner : Owner := ⟨.program ⟨257⟩, ⟨69132⟩⟩
def transferEvent : Nat := 298828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 298822 .summary, .result 298660 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298822 .summary)
      LeftBound298672.bound (LeftBound298672.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67673⟩⟩) (rawTerms := some (Proof.Events1167.exact298822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298660 .summary)
      LeftBound298655.bound (LeftBound298655.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69131⟩⟩) (rawTerms := some (Proof.Events1166.exact298660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound298672.bound, LeftBound298655.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298672.bound, LeftBound298655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound298672.actual selector witness, LeftBound298655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound298828

namespace LeftBound298832
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def transferEvent : Nat := 298832
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 298830 .coefficient) (.predecessor 1 298831 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298830 .coefficient)
      LeftBound298825.bound (LeftBound298825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298831 .coefficient)
      LeftAuthority298575.bound (LeftAuthority298575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1166.exact298576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298575.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298825.bound LeftAuthority298575.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298825.bound, LeftAuthority298575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298825.actual selector witness) * (LeftAuthority298575.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298832

namespace LeftBound298833
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def transferEvent : Nat := 298833
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨69387⟩⟩]⟩ [⟨.result 298576 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298576 .coefficient)
      LeftAuthority298575.bound (LeftAuthority298575.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨69387⟩⟩) (rawTerms := some (Proof.Events1166.exact298576RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298575.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority298575.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority298575.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound298833

namespace LeftBound298834
def owner : Owner := ⟨.program ⟨257⟩, ⟨69389⟩⟩
def transferEvent : Nat := 298834
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 298829 .summary) (.transfer 298833) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298829 .summary)
      LeftBound298828.bound (LeftBound298828.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69132⟩⟩) (rawTerms := some (Proof.Events1167.exact298829RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound298828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 298833)
      LeftBound298833.bound (LeftBound298833.actual selector witness) := by
  exact .transfer (LeftBound298833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound298828.bound LeftBound298833.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound298828.bound, LeftBound298833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound298828.actual selector witness) * (LeftBound298833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298834

namespace LeftBound298845
def owner : Owner := ⟨.program ⟨257⟩, ⟨67879⟩⟩
def transferEvent : Nat := 298845
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 298843 .coefficient) (.value (.predecessor 1 298844 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298843 .coefficient)
      LeftAuthority298841.bound (LeftAuthority298841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298844 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority298841.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298841.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority298841.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound298845

namespace LeftBound298849
def owner : Owner := ⟨.program ⟨257⟩, ⟨67880⟩⟩
def transferEvent : Nat := 298849
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 298847 .coefficient) (.predecessor 1 298848 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 298847 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 298848 .coefficient)
      LeftBound298845.bound (LeftBound298845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1167.exact298846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound298845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound298845.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound298845.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound298845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound298845.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298849

namespace LeftBound298850
def owner : Owner := ⟨.program ⟨257⟩, ⟨67880⟩⟩
def transferEvent : Nat := 298850
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨67877⟩⟩]⟩ [⟨.result 298842 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 298842 .coefficient)
      LeftAuthority298841.bound (LeftAuthority298841.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67877⟩⟩) (rawTerms := some (Proof.Events1167.exact298842RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority298841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority298841.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority298841.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority298841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority298841.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound298850

namespace LeftBound298851
def owner : Owner := ⟨.program ⟨257⟩, ⟨67880⟩⟩
def transferEvent : Nat := 298851
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295195 .summary) (.transfer 298850) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295195 .summary)
      LeftBound295193.bound (LeftBound295193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨2380⟩⟩) (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 298850)
      LeftBound298850.bound (LeftBound298850.actual selector witness) := by
  exact .transfer (LeftBound298850.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295193.bound LeftBound298850.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295193.bound, LeftBound298850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295193.actual selector witness) * (LeftBound298850.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound298851

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
