import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard834

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound126623
def owner : Owner := ⟨.program ⟨257⟩, ⟨32352⟩⟩
def transferEvent : Nat := 126623
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 119870 .summary) (.transfer 126622) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119870 .summary)
      LeftBound119868.bound (LeftBound119868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5527⟩⟩) (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 126622)
      LeftBound126622.bound (LeftBound126622.actual selector witness) := by
  exact .transfer (LeftBound126622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119868.bound LeftBound126622.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119868.bound, LeftBound126622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119868.actual selector witness) * (LeftBound126622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126623

namespace LeftBound126702
def owner : Owner := ⟨.program ⟨257⟩, ⟨31378⟩⟩
def transferEvent : Nat := 126702
def frameStart : Nat := 126673
def rule : BoundRule := .product (.predecessor 0 126700 .coefficient) (.predecessor 1 126701 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126700 .coefficient)
      LeftAuthority126698.bound (LeftAuthority126698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126698.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126701 .coefficient)
      LeftAuthority126695.bound (LeftAuthority126695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126695.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority126698.bound LeftAuthority126695.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126698.bound, LeftAuthority126695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority126698.actual selector witness) * (LeftAuthority126695.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126702

namespace LeftBound126706
def owner : Owner := ⟨.program ⟨257⟩, ⟨31379⟩⟩
def transferEvent : Nat := 126706
def frameStart : Nat := 126673
def rule : BoundRule := .identity (.predecessor 0 126705 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126705 .coefficient)
      LeftBound126702.bound (LeftBound126702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126702.derived selector witness)

def rawBound : CoeffClass := LeftBound126702.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound126702.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound126706

namespace LeftBound126723
def owner : Owner := ⟨.program ⟨257⟩, ⟨33210⟩⟩
def transferEvent : Nat := 126723
def frameStart : Nat := 126673
def rule : BoundRule := .sum [.predecessor 0 126721 .coefficient, .predecessor 1 126722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126721 .coefficient)
      LeftBound126706.bound (LeftBound126706.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound126706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126722 .coefficient)
      LeftAuthority126719.bound (LeftAuthority126719.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority126719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound126706.bound, LeftAuthority126719.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126706.bound, LeftAuthority126719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound126706.actual selector witness, LeftAuthority126719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126723

namespace LeftBound126726
def owner : Owner := ⟨.program ⟨257⟩, ⟨33211⟩⟩
def transferEvent : Nat := 126726
def frameStart : Nat := 126673
def rule : BoundRule := .identity (.predecessor 0 126725 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126725 .coefficient)
      LeftBound126723.bound (LeftBound126723.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound126723.derived selector witness)

def rawBound : CoeffClass := LeftBound126723.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound126723.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound126726

namespace LeftBound126732
def owner : Owner := ⟨.program ⟨257⟩, ⟨33212⟩⟩
def transferEvent : Nat := 126732
def frameStart : Nat := 126673
def rule : BoundRule := .product (.predecessor 0 126730 .coefficient) (.predecessor 1 126731 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126730 .coefficient)
      LeftAuthority126728.bound (LeftAuthority126728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126731 .coefficient)
      LeftBound126726.bound (LeftBound126726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126726.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority126728.bound LeftBound126726.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126728.bound, LeftBound126726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority126728.actual selector witness) * (LeftBound126726.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126732

namespace LeftBound126748
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 126748
def frameStart : Nat := 126673
def rule : BoundRule := .scale (.predecessor 0 126746 .coefficient) (.value (.predecessor 1 126747 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126746 .coefficient)
      LeftAuthority126744.bound (LeftAuthority126744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126747 .coefficient)
      LeftAuthority126735.bound (LeftAuthority126735.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority126735.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority126744.bound LeftAuthority126735.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126744.bound, LeftAuthority126735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority126744.actual selector witness) * (LeftAuthority126735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound126748

namespace LeftBound126751
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 126751
def frameStart : Nat := 126673
def rule : BoundRule := .identity (.predecessor 0 126750 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126750 .coefficient)
      LeftAuthority126738.bound (LeftAuthority126738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126738.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126738.derived selector witness)

def rawBound : CoeffClass := LeftAuthority126738.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126738.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority126738.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound126751

namespace LeftBound126755
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 126755
def frameStart : Nat := 126673
def rule : BoundRule := .product (.predecessor 0 126753 .coefficient) (.predecessor 1 126754 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126753 .coefficient)
      LeftBound126751.bound (LeftBound126751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126754 .coefficient)
      LeftBound126748.bound (LeftBound126748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound126751.bound LeftBound126748.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126751.bound, LeftBound126748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound126751.actual selector witness) * (LeftBound126748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126755

namespace LeftBound126760
def owner : Owner := ⟨.program ⟨257⟩, ⟨33213⟩⟩
def transferEvent : Nat := 126760
def frameStart : Nat := 126673
def rule : BoundRule := .sum [.predecessor 0 126758 .coefficient, .predecessor 1 126759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126758 .coefficient)
      LeftBound126755.bound (LeftBound126755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126759 .coefficient)
      LeftBound126732.bound (LeftBound126732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound126755.bound, LeftBound126732.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126755.bound, LeftBound126732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound126755.actual selector witness, LeftBound126732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126760

namespace LeftBound126764
def owner : Owner := ⟨.program ⟨257⟩, ⟨33418⟩⟩
def transferEvent : Nat := 126764
def frameStart : Nat := 126673
def rule : BoundRule := .product (.predecessor 0 126762 .coefficient) (.predecessor 1 126763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126762 .coefficient)
      LeftBound126760.bound (LeftBound126760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126763 .coefficient)
      LeftAuthority126717.bound (LeftAuthority126717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound126760.bound LeftAuthority126717.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126760.bound, LeftAuthority126717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound126760.actual selector witness) * (LeftAuthority126717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126764

namespace LeftBound126775
def owner : Owner := ⟨.program ⟨257⟩, ⟨31798⟩⟩
def transferEvent : Nat := 126775
def frameStart : Nat := 126673
def rule : BoundRule := .product (.predecessor 0 126773 .coefficient) (.predecessor 1 126774 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126773 .coefficient)
      LeftAuthority126728.bound (LeftAuthority126728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126774 .coefficient)
      LeftAuthority126771.bound (LeftAuthority126771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126771.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority126728.bound LeftAuthority126771.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126728.bound, LeftAuthority126771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority126728.actual selector witness) * (LeftAuthority126771.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound126775

namespace LeftBound126783
def owner : Owner := ⟨.program ⟨257⟩, ⟨31799⟩⟩
def transferEvent : Nat := 126783
def frameStart : Nat := 126673
def rule : BoundRule := .sum [.predecessor 0 126781 .coefficient, .predecessor 1 126782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126781 .coefficient)
      LeftAuthority126779.bound (LeftAuthority126779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority126779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority126779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126782 .coefficient)
      LeftBound126775.bound (LeftBound126775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority126779.bound, LeftBound126775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority126779.bound, LeftBound126775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority126779.actual selector witness, LeftBound126775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126783

namespace LeftBound126787
def owner : Owner := ⟨.program ⟨257⟩, ⟨33419⟩⟩
def transferEvent : Nat := 126787
def frameStart : Nat := 126673
def rule : BoundRule := .sum [.predecessor 0 126785 .coefficient, .predecessor 1 126786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126785 .coefficient)
      LeftBound126783.bound (LeftBound126783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126786 .coefficient)
      LeftBound126764.bound (LeftBound126764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound126783.bound, LeftBound126764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126783.bound, LeftBound126764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound126783.actual selector witness, LeftBound126764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126787

namespace LeftBound126800
def owner : Owner := ⟨.program ⟨257⟩, ⟨33417⟩⟩
def transferEvent : Nat := 126800
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 126798 .coefficient, .predecessor 1 126799 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 126798 .coefficient)
      LeftBound126621.bound (LeftBound126621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events495.exact126797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 126799 .coefficient)
      LeftBound126604.bound (LeftBound126604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events494.exact126611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound126604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound126604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound126621.bound, LeftBound126604.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126621.bound, LeftBound126604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound126621.actual selector witness, LeftBound126604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126800

namespace LeftBound126803
def owner : Owner := ⟨.program ⟨257⟩, ⟨33417⟩⟩
def transferEvent : Nat := 126803
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 126797 .summary, .result 126611 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 126797 .summary)
      LeftBound126623.bound (LeftBound126623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32352⟩⟩) (rawTerms := some (Proof.Events495.exact126797RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound126623.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 126611 .summary)
      LeftBound126606.bound (LeftBound126606.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33416⟩⟩) (rawTerms := some (Proof.Events494.exact126611RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound126606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound126623.bound, LeftBound126606.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound126623.bound, LeftBound126606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound126623.actual selector witness, LeftBound126606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound126803

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
