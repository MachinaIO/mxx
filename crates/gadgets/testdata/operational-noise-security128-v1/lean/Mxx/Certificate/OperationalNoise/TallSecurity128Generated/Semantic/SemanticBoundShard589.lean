import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard588

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91714
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 91714
def frameStart : Nat := 91639
def rule : BoundRule := .scale (.predecessor 0 91712 .coefficient) (.value (.predecessor 1 91713 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91712 .coefficient)
      LeftAuthority91710.bound (LeftAuthority91710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91713 .coefficient)
      LeftAuthority91701.bound (LeftAuthority91701.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority91701.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91710.bound LeftAuthority91701.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91710.bound, LeftAuthority91701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91710.actual selector witness) * (LeftAuthority91701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91714

namespace LeftBound91717
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 91717
def frameStart : Nat := 91639
def rule : BoundRule := .identity (.predecessor 0 91716 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91716 .coefficient)
      LeftAuthority91704.bound (LeftAuthority91704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91704.derived selector witness)

def rawBound : CoeffClass := LeftAuthority91704.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority91704.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91717

namespace LeftBound91721
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 91721
def frameStart : Nat := 91639
def rule : BoundRule := .product (.predecessor 0 91719 .coefficient) (.predecessor 1 91720 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91719 .coefficient)
      LeftBound91717.bound (LeftBound91717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91720 .coefficient)
      LeftBound91714.bound (LeftBound91714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91714.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91717.bound LeftBound91714.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91717.bound, LeftBound91714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91717.actual selector witness) * (LeftBound91714.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91721

namespace LeftBound91726
def owner : Owner := ⟨.program ⟨257⟩, ⟨44089⟩⟩
def transferEvent : Nat := 91726
def frameStart : Nat := 91639
def rule : BoundRule := .sum [.predecessor 0 91724 .coefficient, .predecessor 1 91725 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91724 .coefficient)
      LeftBound91721.bound (LeftBound91721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91721.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91725 .coefficient)
      LeftBound91698.bound (LeftBound91698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91698.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91721.bound, LeftBound91698.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91721.bound, LeftBound91698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91721.actual selector witness, LeftBound91698.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91726

namespace LeftBound91730
def owner : Owner := ⟨.program ⟨257⟩, ⟨44357⟩⟩
def transferEvent : Nat := 91730
def frameStart : Nat := 91639
def rule : BoundRule := .product (.predecessor 0 91728 .coefficient) (.predecessor 1 91729 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91728 .coefficient)
      LeftBound91726.bound (LeftBound91726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91729 .coefficient)
      LeftAuthority91683.bound (LeftAuthority91683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91726.bound LeftAuthority91683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91726.bound, LeftAuthority91683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91726.actual selector witness) * (LeftAuthority91683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91730

namespace LeftBound91741
def owner : Owner := ⟨.program ⟨257⟩, ⟨42830⟩⟩
def transferEvent : Nat := 91741
def frameStart : Nat := 91639
def rule : BoundRule := .product (.predecessor 0 91739 .coefficient) (.predecessor 1 91740 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91739 .coefficient)
      LeftAuthority91694.bound (LeftAuthority91694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91740 .coefficient)
      LeftAuthority91737.bound (LeftAuthority91737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91737.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority91694.bound LeftAuthority91737.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91694.bound, LeftAuthority91737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority91694.actual selector witness) * (LeftAuthority91737.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91741

namespace LeftBound91749
def owner : Owner := ⟨.program ⟨257⟩, ⟨42831⟩⟩
def transferEvent : Nat := 91749
def frameStart : Nat := 91639
def rule : BoundRule := .sum [.predecessor 0 91747 .coefficient, .predecessor 1 91748 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91747 .coefficient)
      LeftAuthority91745.bound (LeftAuthority91745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91745.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91745.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91748 .coefficient)
      LeftBound91741.bound (LeftBound91741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91741.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91745.bound, LeftBound91741.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91745.bound, LeftBound91741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority91745.actual selector witness, LeftBound91741.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91749

namespace LeftBound91753
def owner : Owner := ⟨.program ⟨257⟩, ⟨44358⟩⟩
def transferEvent : Nat := 91753
def frameStart : Nat := 91639
def rule : BoundRule := .sum [.predecessor 0 91751 .coefficient, .predecessor 1 91752 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91751 .coefficient)
      LeftBound91749.bound (LeftBound91749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91752 .coefficient)
      LeftBound91730.bound (LeftBound91730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91749.bound, LeftBound91730.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91749.bound, LeftBound91730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91749.actual selector witness, LeftBound91730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91753

namespace LeftBound91766
def owner : Owner := ⟨.program ⟨257⟩, ⟨44356⟩⟩
def transferEvent : Nat := 91766
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91764 .coefficient, .predecessor 1 91765 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91764 .coefficient)
      LeftBound91587.bound (LeftBound91587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91765 .coefficient)
      LeftBound91570.bound (LeftBound91570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91570.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91587.bound, LeftBound91570.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91587.bound, LeftBound91570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91587.actual selector witness, LeftBound91570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91766

namespace LeftBound91769
def owner : Owner := ⟨.program ⟨257⟩, ⟨44356⟩⟩
def transferEvent : Nat := 91769
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91763 .summary, .result 91577 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91763 .summary)
      LeftBound91589.bound (LeftBound91589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43282⟩⟩) (rawTerms := some (Proof.Events358.exact91763RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91577 .summary)
      LeftBound91572.bound (LeftBound91572.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44355⟩⟩) (rawTerms := some (Proof.Events357.exact91577RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91589.bound, LeftBound91572.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91589.bound, LeftBound91572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound91589.actual selector witness, LeftBound91572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91769

namespace LeftBound91773
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def transferEvent : Nat := 91773
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91771 .coefficient) (.predecessor 1 91772 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91771 .coefficient)
      LeftBound91766.bound (LeftBound91766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91766.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91772 .coefficient)
      LeftAuthority91492.bound (LeftAuthority91492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91492.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91766.bound LeftAuthority91492.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91766.bound, LeftAuthority91492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91766.actual selector witness) * (LeftAuthority91492.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91773

namespace LeftBound91774
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def transferEvent : Nat := 91774
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44794⟩⟩]⟩ [⟨.result 91493 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91493 .coefficient)
      LeftAuthority91492.bound (LeftAuthority91492.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44794⟩⟩) (rawTerms := some (Proof.Events357.exact91493RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91492.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91492.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91492.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91774

namespace LeftBound91775
def owner : Owner := ⟨.program ⟨257⟩, ⟨44796⟩⟩
def transferEvent : Nat := 91775
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91770 .summary) (.transfer 91774) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91770 .summary)
      LeftBound91769.bound (LeftBound91769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44356⟩⟩) (rawTerms := some (Proof.Events358.exact91770RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 91774)
      LeftBound91774.bound (LeftBound91774.actual selector witness) := by
  exact .transfer (LeftBound91774.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound91769.bound LeftBound91774.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91769.bound, LeftBound91774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound91769.actual selector witness) * (LeftBound91774.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91775

namespace LeftBound91786
def owner : Owner := ⟨.program ⟨257⟩, ⟨43638⟩⟩
def transferEvent : Nat := 91786
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91784 .coefficient) (.value (.predecessor 1 91785 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91784 .coefficient)
      LeftAuthority91782.bound (LeftAuthority91782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91785 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91782.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91782.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91782.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91786

namespace LeftBound91790
def owner : Owner := ⟨.program ⟨257⟩, ⟨43639⟩⟩
def transferEvent : Nat := 91790
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91788 .coefficient) (.predecessor 1 91789 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 91788 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 91789 .coefficient)
      LeftBound91786.bound (LeftBound91786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91786.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound91786.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound91786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound91786.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91790

namespace LeftBound91791
def owner : Owner := ⟨.program ⟨257⟩, ⟨43639⟩⟩
def transferEvent : Nat := 91791
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43636⟩⟩]⟩ [⟨.result 91783 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 91783 .coefficient)
      LeftAuthority91782.bound (LeftAuthority91782.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43636⟩⟩) (rawTerms := some (Proof.Events358.exact91783RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91782.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91782.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority91782.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91791

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
