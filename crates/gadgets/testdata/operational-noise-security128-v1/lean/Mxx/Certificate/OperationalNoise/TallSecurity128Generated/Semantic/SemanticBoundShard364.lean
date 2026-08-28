import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard328
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard363

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59850
def owner : Owner := ⟨.program ⟨257⟩, ⟨52399⟩⟩
def transferEvent : Nat := 59850
def frameStart : Nat := 59791
def rule : BoundRule := .identity (.predecessor 0 59849 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59849 .coefficient)
      LeftBound59847.bound (LeftBound59847.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59847.derived selector witness)

def rawBound : CoeffClass := LeftBound59847.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59847.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound59847.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59850

namespace LeftBound59856
def owner : Owner := ⟨.program ⟨257⟩, ⟨52400⟩⟩
def transferEvent : Nat := 59856
def frameStart : Nat := 59791
def rule : BoundRule := .product (.predecessor 0 59854 .coefficient) (.predecessor 1 59855 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59854 .coefficient)
      LeftAuthority59852.bound (LeftAuthority59852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59855 .coefficient)
      LeftBound59850.bound (LeftBound59850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59850.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority59852.bound LeftBound59850.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59852.bound, LeftBound59850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority59852.actual selector witness) * (LeftBound59850.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59856

namespace LeftBound59864
def owner : Owner := ⟨.program ⟨257⟩, ⟨52401⟩⟩
def transferEvent : Nat := 59864
def frameStart : Nat := 59791
def rule : BoundRule := .sum [.predecessor 0 59862 .coefficient, .predecessor 1 59863 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59862 .coefficient)
      LeftAuthority59860.bound (LeftAuthority59860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59863 .coefficient)
      LeftBound59856.bound (LeftBound59856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59856.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59860.bound, LeftBound59856.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59860.bound, LeftBound59856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority59860.actual selector witness, LeftBound59856.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59864

namespace LeftBound59868
def owner : Owner := ⟨.program ⟨257⟩, ⟨53194⟩⟩
def transferEvent : Nat := 59868
def frameStart : Nat := 59791
def rule : BoundRule := .product (.predecessor 0 59866 .coefficient) (.predecessor 1 59867 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59866 .coefficient)
      LeftBound59864.bound (LeftBound59864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59867 .coefficient)
      LeftAuthority59841.bound (LeftAuthority59841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound59864.bound LeftAuthority59841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59864.bound, LeftAuthority59841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound59864.actual selector witness) * (LeftAuthority59841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59868

namespace LeftBound59879
def owner : Owner := ⟨.program ⟨257⟩, ⟨51320⟩⟩
def transferEvent : Nat := 59879
def frameStart : Nat := 59791
def rule : BoundRule := .product (.predecessor 0 59877 .coefficient) (.predecessor 1 59878 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59877 .coefficient)
      LeftAuthority59852.bound (LeftAuthority59852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59878 .coefficient)
      LeftAuthority59875.bound (LeftAuthority59875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59875.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59875.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority59852.bound LeftAuthority59875.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59852.bound, LeftAuthority59875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority59852.actual selector witness) * (LeftAuthority59875.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59879

namespace LeftBound59887
def owner : Owner := ⟨.program ⟨257⟩, ⟨51321⟩⟩
def transferEvent : Nat := 59887
def frameStart : Nat := 59791
def rule : BoundRule := .sum [.predecessor 0 59885 .coefficient, .predecessor 1 59886 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59885 .coefficient)
      LeftAuthority59883.bound (LeftAuthority59883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59886 .coefficient)
      LeftBound59879.bound (LeftBound59879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59883.bound, LeftBound59879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59883.bound, LeftBound59879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority59883.actual selector witness, LeftBound59879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59887

namespace LeftBound59891
def owner : Owner := ⟨.program ⟨257⟩, ⟨53199⟩⟩
def transferEvent : Nat := 59891
def frameStart : Nat := 59791
def rule : BoundRule := .sum [.predecessor 0 59889 .coefficient, .predecessor 1 59890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59889 .coefficient)
      LeftBound59887.bound (LeftBound59887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59890 .coefficient)
      LeftBound59868.bound (LeftBound59868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59868.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59887.bound, LeftBound59868.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59887.bound, LeftBound59868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59887.actual selector witness, LeftBound59868.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59891

namespace LeftBound59904
def owner : Owner := ⟨.program ⟨257⟩, ⟨53196⟩⟩
def transferEvent : Nat := 59904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59902 .coefficient, .predecessor 1 59903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59902 .coefficient)
      LeftBound59733.bound (LeftBound59733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59903 .coefficient)
      LeftBound59716.bound (LeftBound59716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events233.exact59723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59716.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59733.bound, LeftBound59716.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59733.bound, LeftBound59716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59733.actual selector witness, LeftBound59716.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59904

namespace LeftBound59907
def owner : Owner := ⟨.program ⟨257⟩, ⟨53196⟩⟩
def transferEvent : Nat := 59907
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59901 .summary, .result 59723 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59901 .summary)
      LeftBound59735.bound (LeftBound59735.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51915⟩⟩) (rawTerms := some (Proof.Events233.exact59901RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59723 .summary)
      LeftBound59718.bound (LeftBound59718.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53195⟩⟩) (rawTerms := some (Proof.Events233.exact59723RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59718.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59735.bound, LeftBound59718.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59735.bound, LeftBound59718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59735.actual selector witness, LeftBound59718.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59907

namespace LeftBound59911
def owner : Owner := ⟨.program ⟨257⟩, ⟨53197⟩⟩
def transferEvent : Nat := 59911
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59909 .coefficient) (.predecessor 1 59910 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59909 .coefficient)
      LeftBound59904.bound (LeftBound59904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact59908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59910 .coefficient)
      LeftBound15801.bound (LeftBound15801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound59904.bound LeftBound15801.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59904.bound, LeftBound15801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound59904.actual selector witness) * (LeftBound15801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59911

namespace LeftBound59912
def owner : Owner := ⟨.program ⟨257⟩, ⟨53197⟩⟩
def transferEvent : Nat := 59912
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩ [⟨.result 15798 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15798 .coefficient)
      LeftAuthority15797.bound (LeftAuthority15797.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7131⟩⟩) (rawTerms := some (Proof.Events061.exact15798RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15797.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15797.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15797.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15797.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59912

namespace LeftBound59913
def owner : Owner := ⟨.program ⟨257⟩, ⟨53197⟩⟩
def transferEvent : Nat := 59913
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 59908 .summary) (.transfer 59912) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59908 .summary)
      LeftBound59907.bound (LeftBound59907.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53196⟩⟩) (rawTerms := some (Proof.Events234.exact59908RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59907.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 59912)
      LeftBound59912.bound (LeftBound59912.actual selector witness) := by
  exact .transfer (LeftBound59912.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound59907.bound LeftBound59912.bound
def bound : CoeffClass := .finite ⟨345633123169561229153141416722874415185920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59907.bound, LeftBound59912.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound59907.actual selector witness) * (LeftBound59912.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59913

namespace LeftBound59928
def owner : Owner := ⟨.program ⟨257⟩, ⟨34135⟩⟩
def transferEvent : Nat := 59928
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59926 .coefficient) (.predecessor 1 59927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59926 .coefficient)
      LeftBound53675.bound (LeftBound53675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59927 .coefficient)
      LeftAuthority59924.bound (LeftAuthority59924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact59925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53675.bound LeftAuthority59924.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53675.bound, LeftAuthority59924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53675.actual selector witness) * (LeftAuthority59924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59928

namespace LeftBound59929
def owner : Owner := ⟨.program ⟨257⟩, ⟨34135⟩⟩
def transferEvent : Nat := 59929
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨34133⟩⟩]⟩ [⟨.result 59925 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59925 .coefficient)
      LeftAuthority59924.bound (LeftAuthority59924.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨34133⟩⟩) (rawTerms := some (Proof.Events234.exact59925RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59924.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority59924.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority59924.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59929

namespace LeftBound59930
def owner : Owner := ⟨.program ⟨257⟩, ⟨34135⟩⟩
def transferEvent : Nat := 59930
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53679 .summary) (.transfer 59929) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53679 .summary)
      LeftBound53678.bound (LeftBound53678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33549⟩⟩) (rawTerms := some (Proof.Events209.exact53679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 59929)
      LeftBound59929.bound (LeftBound59929.actual selector witness) := by
  exact .transfer (LeftBound59929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound53678.bound LeftBound59929.bound
def bound : CoeffClass := .finite ⟨32189200113374879571150551121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53678.bound, LeftBound59929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound53678.actual selector witness) * (LeftBound59929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59930

namespace LeftBound59941
def owner : Owner := ⟨.program ⟨257⟩, ⟨32854⟩⟩
def transferEvent : Nat := 59941
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 59939 .coefficient) (.value (.predecessor 1 59940 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59939 .coefficient)
      LeftAuthority59937.bound (LeftAuthority59937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact59938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59940 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority59937.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59937.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority59937.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound59941

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
