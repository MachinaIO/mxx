import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard192

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound34495
def owner : Owner := ⟨.program ⟨257⟩, ⟨13719⟩⟩
def transferEvent : Nat := 34495
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨123⟩⟩]⟩ [⟨.result 19618 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19618 .coefficient)
      LeftBound19617.bound (LeftBound19617.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨123⟩⟩) (rawTerms := some (Proof.Events076.exact19618RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19617.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19617.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19617.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34495

namespace LeftBound34500
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def transferEvent : Nat := 34500
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34498 .coefficient) (.predecessor 1 34499 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34498 .coefficient)
      LeftBound34494.bound (LeftBound34494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34494.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34499 .coefficient)
      LeftBound19614.bound (LeftBound19614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34494.bound LeftBound19614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34494.bound, LeftBound19614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34494.actual selector witness) * (LeftBound19614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34500

namespace LeftBound34501
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def transferEvent : Nat := 34501
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9550⟩⟩]⟩ [⟨.result 19611 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19611 .coefficient)
      LeftAuthority19610.bound (LeftAuthority19610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9550⟩⟩) (rawTerms := some (Proof.Events076.exact19611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19610.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority19610.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34501

namespace LeftBound34502
def owner : Owner := ⟨.program ⟨257⟩, ⟨13720⟩⟩
def transferEvent : Nat := 34502
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 34497 .summary) (.transfer 34501) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34497 .summary)
      LeftBound34495.bound (LeftBound34495.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13719⟩⟩) (rawTerms := some (Proof.Events134.exact34497RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 34501)
      LeftBound34501.bound (LeftBound34501.actual selector witness) := by
  exact .transfer (LeftBound34501.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34495.bound LeftBound34501.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34495.bound, LeftBound34501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34495.actual selector witness) * (LeftBound34501.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34502

namespace LeftBound34510
def owner : Owner := ⟨.program ⟨257⟩, ⟨34657⟩⟩
def transferEvent : Nat := 34510
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 34508 .coefficient, .predecessor 1 34509 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34508 .coefficient)
      LeftBound34500.bound (LeftBound34500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34509 .coefficient)
      LeftBound34472.bound (LeftBound34472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34472.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34500.bound, LeftBound34472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34500.bound, LeftBound34472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34500.actual selector witness, LeftBound34472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34510

namespace LeftBound34512
def owner : Owner := ⟨.program ⟨257⟩, ⟨34657⟩⟩
def transferEvent : Nat := 34512
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 34507 .summary, .result 34477 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34507 .summary)
      LeftBound34502.bound (LeftBound34502.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨13720⟩⟩) (rawTerms := some (Proof.Events134.exact34507RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34502.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34477 .summary)
      LeftBound34474.bound (LeftBound34474.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34656⟩⟩) (rawTerms := some (Proof.Events134.exact34477RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34502.bound, LeftBound34474.bound]
def bound : CoeffClass := .finite ⟨279206952960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34502.bound, LeftBound34474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34502.actual selector witness, LeftBound34474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34512

namespace LeftBound34516
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def transferEvent : Nat := 34516
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34514 .coefficient) (.predecessor 1 34515 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34514 .coefficient)
      LeftBound34510.bound (LeftBound34510.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34510.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34515 .coefficient)
      LeftAuthority34448.bound (LeftAuthority34448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34448.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34510.bound LeftAuthority34448.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34510.bound, LeftAuthority34448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34510.actual selector witness) * (LeftAuthority34448.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34516

namespace LeftBound34517
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def transferEvent : Nat := 34517
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36358⟩⟩]⟩ [⟨.result 34449 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34449 .coefficient)
      LeftAuthority34448.bound (LeftAuthority34448.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36358⟩⟩) (rawTerms := some (Proof.Events134.exact34449RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34448.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34448.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34448.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34517

namespace LeftBound34518
def owner : Owner := ⟨.program ⟨257⟩, ⟨36359⟩⟩
def transferEvent : Nat := 34518
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 34513 .summary) (.transfer 34517) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34513 .summary)
      LeftBound34512.bound (LeftBound34512.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34657⟩⟩) (rawTerms := some (Proof.Events134.exact34513RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34512.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 34517)
      LeftBound34517.bound (LeftBound34517.actual selector witness) := by
  exact .transfer (LeftBound34517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound34512.bound LeftBound34517.bound
def bound : CoeffClass := .finite ⟨2997961829447525990400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34512.bound, LeftBound34517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound34512.actual selector witness) * (LeftBound34517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34518

namespace LeftBound34529
def owner : Owner := ⟨.program ⟨257⟩, ⟨35281⟩⟩
def transferEvent : Nat := 34529
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 34527 .coefficient) (.value (.predecessor 1 34528 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34527 .coefficient)
      LeftAuthority34525.bound (LeftAuthority34525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34525.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34528 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority34525.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34525.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34525.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound34529

namespace LeftBound34533
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def transferEvent : Nat := 34533
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34531 .coefficient) (.predecessor 1 34532 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34531 .coefficient)
      LeftBound32117.bound (LeftBound32117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34532 .coefficient)
      LeftBound34529.bound (LeftBound34529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events134.exact34530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34529.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32117.bound LeftBound34529.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32117.bound, LeftBound34529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32117.actual selector witness) * (LeftBound34529.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34533

namespace LeftBound34534
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def transferEvent : Nat := 34534
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35279⟩⟩]⟩ [⟨.result 34526 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 34526 .coefficient)
      LeftAuthority34525.bound (LeftAuthority34525.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35279⟩⟩) (rawTerms := some (Proof.Events134.exact34526RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34525.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34525.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority34525.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34534

namespace LeftBound34535
def owner : Owner := ⟨.program ⟨257⟩, ⟨35282⟩⟩
def transferEvent : Nat := 34535
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 34534) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 34534)
      LeftBound34534.bound (LeftBound34534.actual selector witness) := by
  exact .transfer (LeftBound34534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound34534.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound34534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound34534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34535

namespace LeftBound34614
def owner : Owner := ⟨.program ⟨257⟩, ⟨34651⟩⟩
def transferEvent : Nat := 34614
def frameStart : Nat := 34585
def rule : BoundRule := .product (.predecessor 0 34612 .coefficient) (.predecessor 1 34613 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34612 .coefficient)
      LeftAuthority34610.bound (LeftAuthority34610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34613 .coefficient)
      LeftAuthority34607.bound (LeftAuthority34607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34607.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority34610.bound LeftAuthority34607.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34610.bound, LeftAuthority34607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority34610.actual selector witness) * (LeftAuthority34607.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34614

namespace LeftBound34618
def owner : Owner := ⟨.program ⟨257⟩, ⟨34652⟩⟩
def transferEvent : Nat := 34618
def frameStart : Nat := 34585
def rule : BoundRule := .identity (.predecessor 0 34617 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34617 .coefficient)
      LeftBound34614.bound (LeftBound34614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34614.derived selector witness)

def rawBound : CoeffClass := LeftBound34614.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound34614.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34618

namespace LeftBound34635
def owner : Owner := ⟨.program ⟨257⟩, ⟨36062⟩⟩
def transferEvent : Nat := 34635
def frameStart : Nat := 34585
def rule : BoundRule := .sum [.predecessor 0 34633 .coefficient, .predecessor 1 34634 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 34633 .coefficient)
      LeftBound34618.bound (LeftBound34618.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 34634 .coefficient)
      LeftAuthority34631.bound (LeftAuthority34631.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority34631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34618.bound, LeftAuthority34631.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34618.bound, LeftAuthority34631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound34618.actual selector witness, LeftAuthority34631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34635

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
