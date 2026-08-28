import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard073
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard143

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound27841
def owner : Owner := ⟨.program ⟨257⟩, ⟨49471⟩⟩
def transferEvent : Nat := 27841
def frameStart : Nat := 27782
def rule : BoundRule := .identity (.predecessor 0 27840 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27840 .coefficient)
      LeftBound27838.bound (LeftBound27838.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound27838.derived selector witness)

def rawBound : CoeffClass := LeftBound27838.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound27838.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound27841

namespace LeftBound27847
def owner : Owner := ⟨.program ⟨257⟩, ⟨49472⟩⟩
def transferEvent : Nat := 27847
def frameStart : Nat := 27782
def rule : BoundRule := .product (.predecessor 0 27845 .coefficient) (.predecessor 1 27846 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27845 .coefficient)
      LeftAuthority27843.bound (LeftAuthority27843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27846 .coefficient)
      LeftBound27841.bound (LeftBound27841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority27843.bound LeftBound27841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27843.bound, LeftBound27841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority27843.actual selector witness) * (LeftBound27841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27847

namespace LeftBound27855
def owner : Owner := ⟨.program ⟨257⟩, ⟨49473⟩⟩
def transferEvent : Nat := 27855
def frameStart : Nat := 27782
def rule : BoundRule := .sum [.predecessor 0 27853 .coefficient, .predecessor 1 27854 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27853 .coefficient)
      LeftAuthority27851.bound (LeftAuthority27851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27854 .coefficient)
      LeftBound27847.bound (LeftBound27847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27847.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27847.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27851.bound, LeftBound27847.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27851.bound, LeftBound27847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority27851.actual selector witness, LeftBound27847.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27855

namespace LeftBound27859
def owner : Owner := ⟨.program ⟨257⟩, ⟨49806⟩⟩
def transferEvent : Nat := 27859
def frameStart : Nat := 27782
def rule : BoundRule := .product (.predecessor 0 27857 .coefficient) (.predecessor 1 27858 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27857 .coefficient)
      LeftBound27855.bound (LeftBound27855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27858 .coefficient)
      LeftAuthority27832.bound (LeftAuthority27832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27832.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27855.bound LeftAuthority27832.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27855.bound, LeftAuthority27832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27855.actual selector witness) * (LeftAuthority27832.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27859

namespace LeftBound27870
def owner : Owner := ⟨.program ⟨257⟩, ⟨48247⟩⟩
def transferEvent : Nat := 27870
def frameStart : Nat := 27782
def rule : BoundRule := .product (.predecessor 0 27868 .coefficient) (.predecessor 1 27869 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27868 .coefficient)
      LeftAuthority27843.bound (LeftAuthority27843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27869 .coefficient)
      LeftAuthority27866.bound (LeftAuthority27866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27866.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27866.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority27843.bound LeftAuthority27866.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27843.bound, LeftAuthority27866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority27843.actual selector witness) * (LeftAuthority27866.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27870

namespace LeftBound27878
def owner : Owner := ⟨.program ⟨257⟩, ⟨48248⟩⟩
def transferEvent : Nat := 27878
def frameStart : Nat := 27782
def rule : BoundRule := .sum [.predecessor 0 27876 .coefficient, .predecessor 1 27877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27876 .coefficient)
      LeftAuthority27874.bound (LeftAuthority27874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27874.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27877 .coefficient)
      LeftBound27870.bound (LeftBound27870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27874.bound, LeftBound27870.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27874.bound, LeftBound27870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority27874.actual selector witness, LeftBound27870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27878

namespace LeftBound27882
def owner : Owner := ⟨.program ⟨257⟩, ⟨49810⟩⟩
def transferEvent : Nat := 27882
def frameStart : Nat := 27782
def rule : BoundRule := .sum [.predecessor 0 27880 .coefficient, .predecessor 1 27881 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27880 .coefficient)
      LeftBound27878.bound (LeftBound27878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27878.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27881 .coefficient)
      LeftBound27859.bound (LeftBound27859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27859.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27878.bound, LeftBound27859.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27878.bound, LeftBound27859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27878.actual selector witness, LeftBound27859.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27882

namespace LeftBound27895
def owner : Owner := ⟨.program ⟨257⟩, ⟨49808⟩⟩
def transferEvent : Nat := 27895
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 27893 .coefficient, .predecessor 1 27894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27893 .coefficient)
      LeftBound27724.bound (LeftBound27724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27894 .coefficient)
      LeftBound27707.bound (LeftBound27707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27707.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27724.bound, LeftBound27707.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27724.bound, LeftBound27707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27724.actual selector witness, LeftBound27707.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27895

namespace LeftBound27898
def owner : Owner := ⟨.program ⟨257⟩, ⟨49808⟩⟩
def transferEvent : Nat := 27898
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 27892 .summary, .result 27714 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27892 .summary)
      LeftBound27726.bound (LeftBound27726.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48721⟩⟩) (rawTerms := some (Proof.Events108.exact27892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27714 .summary)
      LeftBound27709.bound (LeftBound27709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49807⟩⟩) (rawTerms := some (Proof.Events108.exact27714RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27726.bound, LeftBound27709.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27726.bound, LeftBound27709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27726.actual selector witness, LeftBound27709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27898

namespace LeftBound27902
def owner : Owner := ⟨.program ⟨257⟩, ⟨49809⟩⟩
def transferEvent : Nat := 27902
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27900 .coefficient) (.predecessor 1 27901 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27900 .coefficient)
      LeftBound27895.bound (LeftBound27895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27901 .coefficient)
      LeftBound15541.bound (LeftBound15541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27895.bound LeftBound15541.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27895.bound, LeftBound15541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27895.actual selector witness) * (LeftBound15541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27902

namespace LeftBound27903
def owner : Owner := ⟨.program ⟨257⟩, ⟨49809⟩⟩
def transferEvent : Nat := 27903
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩ [⟨.result 15538 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15538 .coefficient)
      LeftAuthority15537.bound (LeftAuthority15537.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7147⟩⟩) (rawTerms := some (Proof.Events060.exact15538RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15537.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15537.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15537.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27903

namespace LeftBound27904
def owner : Owner := ⟨.program ⟨257⟩, ⟨49809⟩⟩
def transferEvent : Nat := 27904
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 27899 .summary) (.transfer 27903) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27899 .summary)
      LeftBound27898.bound (LeftBound27898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49808⟩⟩) (rawTerms := some (Proof.Events108.exact27899RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 27903)
      LeftBound27903.bound (LeftBound27903.actual selector witness) := by
  exact .transfer (LeftBound27903.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound27898.bound LeftBound27903.bound
def bound : CoeffClass := .finite ⟨345685857434530723496243679576218056785920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27898.bound, LeftBound27903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound27898.actual selector witness) * (LeftBound27903.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27904

namespace LeftBound27919
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def transferEvent : Nat := 27919
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 27917 .coefficient) (.predecessor 1 27918 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27917 .coefficient)
      LeftBound17852.bound (LeftBound17852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27918 .coefficient)
      LeftAuthority27915.bound (LeftAuthority27915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound17852.bound LeftAuthority27915.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17852.bound, LeftAuthority27915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound17852.actual selector witness) * (LeftAuthority27915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27919

namespace LeftBound27920
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def transferEvent : Nat := 27920
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47125⟩⟩]⟩ [⟨.result 27916 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27916 .coefficient)
      LeftAuthority27915.bound (LeftAuthority27915.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47125⟩⟩) (rawTerms := some (Proof.Events109.exact27916RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27915.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority27915.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority27915.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound27920

namespace LeftBound27921
def owner : Owner := ⟨.program ⟨257⟩, ⟨47127⟩⟩
def transferEvent : Nat := 27921
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17856 .summary) (.transfer 27920) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17856 .summary)
      LeftBound17855.bound (LeftBound17855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46885⟩⟩) (rawTerms := some (Proof.Events069.exact17856RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 27920)
      LeftBound27920.bound (LeftBound27920.actual selector witness) := by
  exact .transfer (LeftBound27920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound17855.bound LeftBound27920.bound
def bound : CoeffClass := .finite ⟨32194307824962751379413684715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17855.bound, LeftBound27920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound17855.actual selector witness) * (LeftBound27920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound27921

namespace LeftBound27932
def owner : Owner := ⟨.program ⟨257⟩, ⟨46040⟩⟩
def transferEvent : Nat := 27932
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 27930 .coefficient) (.value (.predecessor 1 27931 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27930 .coefficient)
      LeftAuthority27928.bound (LeftAuthority27928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27928.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27931 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority27928.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27928.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority27928.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound27932

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
