import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard260

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44897
def owner : Owner := ⟨.program ⟨257⟩, ⟨54915⟩⟩
def transferEvent : Nat := 44897
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨54912⟩⟩]⟩ [⟨.result 44889 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44889 .coefficient)
      LeftAuthority44888.bound (LeftAuthority44888.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54912⟩⟩) (rawTerms := some (Proof.Events175.exact44889RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44888.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44888.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority44888.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44897

namespace LeftBound44898
def owner : Owner := ⟨.program ⟨257⟩, ⟨54915⟩⟩
def transferEvent : Nat := 44898
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 44897) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 44897)
      LeftBound44897.bound (LeftBound44897.actual selector witness) := by
  exact .transfer (LeftBound44897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound44897.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound44897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound44897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44898

namespace LeftBound44993
def owner : Owner := ⟨.program ⟨257⟩, ⟨53941⟩⟩
def transferEvent : Nat := 44993
def frameStart : Nat := 44954
def rule : BoundRule := .identity (.predecessor 0 44992 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 44992 .coefficient)
      LeftAuthority44990.bound (LeftAuthority44990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44990.derived selector witness)

def rawBound : CoeffClass := LeftAuthority44990.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority44990.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44993

namespace LeftBound45010
def owner : Owner := ⟨.program ⟨257⟩, ⟨55382⟩⟩
def transferEvent : Nat := 45010
def frameStart : Nat := 44954
def rule : BoundRule := .sum [.predecessor 0 45008 .coefficient, .predecessor 1 45009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45008 .coefficient)
      LeftBound44993.bound (LeftBound44993.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45009 .coefficient)
      LeftAuthority45006.bound (LeftAuthority45006.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority45006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44993.bound, LeftAuthority45006.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44993.bound, LeftAuthority45006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44993.actual selector witness, LeftAuthority45006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45010

namespace LeftBound45013
def owner : Owner := ⟨.program ⟨257⟩, ⟨55383⟩⟩
def transferEvent : Nat := 45013
def frameStart : Nat := 44954
def rule : BoundRule := .identity (.predecessor 0 45012 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45012 .coefficient)
      LeftBound45010.bound (LeftBound45010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound45010.derived selector witness)

def rawBound : CoeffClass := LeftBound45010.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound45010.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound45013

namespace LeftBound45019
def owner : Owner := ⟨.program ⟨257⟩, ⟨55384⟩⟩
def transferEvent : Nat := 45019
def frameStart : Nat := 44954
def rule : BoundRule := .product (.predecessor 0 45017 .coefficient) (.predecessor 1 45018 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45017 .coefficient)
      LeftAuthority45015.bound (LeftAuthority45015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45018 .coefficient)
      LeftBound45013.bound (LeftBound45013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45013.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority45015.bound LeftBound45013.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45015.bound, LeftBound45013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority45015.actual selector witness) * (LeftBound45013.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45019

namespace LeftBound45027
def owner : Owner := ⟨.program ⟨257⟩, ⟨55385⟩⟩
def transferEvent : Nat := 45027
def frameStart : Nat := 44954
def rule : BoundRule := .sum [.predecessor 0 45025 .coefficient, .predecessor 1 45026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45025 .coefficient)
      LeftAuthority45023.bound (LeftAuthority45023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45023.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45026 .coefficient)
      LeftBound45019.bound (LeftBound45019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority45023.bound, LeftBound45019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45023.bound, LeftBound45019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority45023.actual selector witness, LeftBound45019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45027

namespace LeftBound45031
def owner : Owner := ⟨.program ⟨257⟩, ⟨56205⟩⟩
def transferEvent : Nat := 45031
def frameStart : Nat := 44954
def rule : BoundRule := .product (.predecessor 0 45029 .coefficient) (.predecessor 1 45030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45029 .coefficient)
      LeftBound45027.bound (LeftBound45027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45030 .coefficient)
      LeftAuthority45004.bound (LeftAuthority45004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45027.bound LeftAuthority45004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45027.bound, LeftAuthority45004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45027.actual selector witness) * (LeftAuthority45004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45031

namespace LeftBound45042
def owner : Owner := ⟨.program ⟨257⟩, ⟨54319⟩⟩
def transferEvent : Nat := 45042
def frameStart : Nat := 44954
def rule : BoundRule := .product (.predecessor 0 45040 .coefficient) (.predecessor 1 45041 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45040 .coefficient)
      LeftAuthority45015.bound (LeftAuthority45015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45041 .coefficient)
      LeftAuthority45038.bound (LeftAuthority45038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45038.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority45015.bound LeftAuthority45038.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45015.bound, LeftAuthority45038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority45015.actual selector witness) * (LeftAuthority45038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45042

namespace LeftBound45050
def owner : Owner := ⟨.program ⟨257⟩, ⟨54320⟩⟩
def transferEvent : Nat := 45050
def frameStart : Nat := 44954
def rule : BoundRule := .sum [.predecessor 0 45048 .coefficient, .predecessor 1 45049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45048 .coefficient)
      LeftAuthority45046.bound (LeftAuthority45046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45046.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45049 .coefficient)
      LeftBound45042.bound (LeftBound45042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority45046.bound, LeftBound45042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45046.bound, LeftBound45042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority45046.actual selector witness, LeftBound45042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45050

namespace LeftBound45054
def owner : Owner := ⟨.program ⟨257⟩, ⟨56210⟩⟩
def transferEvent : Nat := 45054
def frameStart : Nat := 44954
def rule : BoundRule := .sum [.predecessor 0 45052 .coefficient, .predecessor 1 45053 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45052 .coefficient)
      LeftBound45050.bound (LeftBound45050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45053 .coefficient)
      LeftBound45031.bound (LeftBound45031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact45036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45031.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45050.bound, LeftBound45031.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45050.bound, LeftBound45031.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound45050.actual selector witness, LeftBound45031.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45054

namespace LeftBound45067
def owner : Owner := ⟨.program ⟨257⟩, ⟨56207⟩⟩
def transferEvent : Nat := 45067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 45065 .coefficient, .predecessor 1 45066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45065 .coefficient)
      LeftBound44896.bound (LeftBound44896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events176.exact45064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45066 .coefficient)
      LeftBound44879.bound (LeftBound44879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events175.exact44886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44879.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44879.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44896.bound, LeftBound44879.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44896.bound, LeftBound44879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44896.actual selector witness, LeftBound44879.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45067

namespace LeftBound45070
def owner : Owner := ⟨.program ⟨257⟩, ⟨56207⟩⟩
def transferEvent : Nat := 45070
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 45064 .summary, .result 44886 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45064 .summary)
      LeftBound44898.bound (LeftBound44898.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54915⟩⟩) (rawTerms := some (Proof.Events176.exact45064RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44886 .summary)
      LeftBound44881.bound (LeftBound44881.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56206⟩⟩) (rawTerms := some (Proof.Events175.exact44886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44898.bound, LeftBound44881.bound]
def bound : CoeffClass := .finite ⟨32189789464712143775715074244608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44898.bound, LeftBound44881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound44898.actual selector witness, LeftBound44881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45070

namespace LeftBound45074
def owner : Owner := ⟨.program ⟨257⟩, ⟨56208⟩⟩
def transferEvent : Nat := 45074
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 45072 .coefficient) (.predecessor 1 45073 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45072 .coefficient)
      LeftBound45067.bound (LeftBound45067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events176.exact45071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45073 .coefficient)
      LeftBound15781.bound (LeftBound15781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15781.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45067.bound LeftBound15781.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45067.bound, LeftBound15781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45067.actual selector witness) * (LeftBound15781.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45074

namespace LeftBound45075
def owner : Owner := ⟨.program ⟨257⟩, ⟨56208⟩⟩
def transferEvent : Nat := 45075
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7125⟩⟩]⟩ [⟨.result 15778 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15778 .coefficient)
      LeftAuthority15777.bound (LeftAuthority15777.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7125⟩⟩) (rawTerms := some (Proof.Events061.exact15778RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15777.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15777.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15777.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound45075

namespace LeftBound45076
def owner : Owner := ⟨.program ⟨257⟩, ⟨56208⟩⟩
def transferEvent : Nat := 45076
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 45071 .summary) (.transfer 45075) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45071 .summary)
      LeftBound45070.bound (LeftBound45070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56207⟩⟩) (rawTerms := some (Proof.Events176.exact45071RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 45075)
      LeftBound45075.bound (LeftBound45075.actual selector witness) := by
  exact .transfer (LeftBound45075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45070.bound LeftBound45075.bound
def bound : CoeffClass := .finite ⟨345635232540160008926865507237008160849920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45070.bound, LeftBound45075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45070.actual selector witness) * (LeftBound45075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45076

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
