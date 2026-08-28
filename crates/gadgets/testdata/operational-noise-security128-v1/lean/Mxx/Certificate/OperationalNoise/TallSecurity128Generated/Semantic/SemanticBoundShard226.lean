import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard225

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound38854
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def transferEvent : Nat := 38854
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38852 .coefficient) (.predecessor 1 38853 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38852 .coefficient)
      LeftBound38848.bound (LeftBound38848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38853 .coefficient)
      LeftAuthority38786.bound (LeftAuthority38786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38786.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38786.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound38848.bound LeftAuthority38786.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38848.bound, LeftAuthority38786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound38848.actual selector witness) * (LeftAuthority38786.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38854

namespace LeftBound38855
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def transferEvent : Nat := 38855
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33558⟩⟩]⟩ [⟨.result 38787 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 38787 .coefficient)
      LeftAuthority38786.bound (LeftAuthority38786.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33558⟩⟩) (rawTerms := some (Proof.Events151.exact38787RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38786.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38786.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38786.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority38786.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38855

namespace LeftBound38856
def owner : Owner := ⟨.program ⟨257⟩, ⟨33559⟩⟩
def transferEvent : Nat := 38856
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38851 .summary) (.transfer 38855) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 38851 .summary)
      LeftBound38850.bound (LeftBound38850.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31736⟩⟩) (rawTerms := some (Proof.Events151.exact38851RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 38855)
      LeftBound38855.bound (LeftBound38855.actual selector witness) := by
  exact .transfer (LeftBound38855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound38850.bound LeftBound38855.bound
def bound : CoeffClass := .finite ⟨2997650799598260715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38850.bound, LeftBound38855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound38850.actual selector witness) * (LeftBound38855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38856

namespace LeftBound38867
def owner : Owner := ⟨.program ⟨257⟩, ⟨32481⟩⟩
def transferEvent : Nat := 38867
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 38865 .coefficient) (.value (.predecessor 1 38866 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38865 .coefficient)
      LeftAuthority38863.bound (LeftAuthority38863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38866 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38863.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38863.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority38863.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38867

namespace LeftBound38871
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def transferEvent : Nat := 38871
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 38869 .coefficient) (.predecessor 1 38870 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38869 .coefficient)
      LeftBound32117.bound (LeftBound32117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38870 .coefficient)
      LeftBound38867.bound (LeftBound38867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38867.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32117.bound LeftBound38867.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32117.bound, LeftBound38867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32117.actual selector witness) * (LeftBound38867.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38871

namespace LeftBound38872
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def transferEvent : Nat := 38872
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32479⟩⟩]⟩ [⟨.result 38864 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 38864 .coefficient)
      LeftAuthority38863.bound (LeftAuthority38863.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32479⟩⟩) (rawTerms := some (Proof.Events151.exact38864RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38863.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38863.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38863.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority38863.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound38872

namespace LeftBound38873
def owner : Owner := ⟨.program ⟨257⟩, ⟨32482⟩⟩
def transferEvent : Nat := 38873
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 38872) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 38872)
      LeftBound38872.bound (LeftBound38872.actual selector witness) := by
  exact .transfer (LeftBound38872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound38872.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound38872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound38872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38873

namespace LeftBound38952
def owner : Owner := ⟨.program ⟨257⟩, ⟨31729⟩⟩
def transferEvent : Nat := 38952
def frameStart : Nat := 38923
def rule : BoundRule := .product (.predecessor 0 38950 .coefficient) (.predecessor 1 38951 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38950 .coefficient)
      LeftAuthority38948.bound (LeftAuthority38948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38948.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38951 .coefficient)
      LeftAuthority38945.bound (LeftAuthority38945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38945.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38945.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority38948.bound LeftAuthority38945.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38948.bound, LeftAuthority38945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority38948.actual selector witness) * (LeftAuthority38945.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38952

namespace LeftBound38956
def owner : Owner := ⟨.program ⟨257⟩, ⟨31730⟩⟩
def transferEvent : Nat := 38956
def frameStart : Nat := 38923
def rule : BoundRule := .identity (.predecessor 0 38955 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38955 .coefficient)
      LeftBound38952.bound (LeftBound38952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38952.derived selector witness)

def rawBound : CoeffClass := LeftBound38952.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound38952.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38956

namespace LeftBound38973
def owner : Owner := ⟨.program ⟨257⟩, ⟨33262⟩⟩
def transferEvent : Nat := 38973
def frameStart : Nat := 38923
def rule : BoundRule := .sum [.predecessor 0 38971 .coefficient, .predecessor 1 38972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38971 .coefficient)
      LeftBound38956.bound (LeftBound38956.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38972 .coefficient)
      LeftAuthority38969.bound (LeftAuthority38969.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38956.bound, LeftAuthority38969.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38956.bound, LeftAuthority38969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound38956.actual selector witness, LeftAuthority38969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound38973

namespace LeftBound38976
def owner : Owner := ⟨.program ⟨257⟩, ⟨33263⟩⟩
def transferEvent : Nat := 38976
def frameStart : Nat := 38923
def rule : BoundRule := .identity (.predecessor 0 38975 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38975 .coefficient)
      LeftBound38973.bound (LeftBound38973.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound38973.derived selector witness)

def rawBound : CoeffClass := LeftBound38973.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound38973.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound38976

namespace LeftBound38982
def owner : Owner := ⟨.program ⟨257⟩, ⟨33264⟩⟩
def transferEvent : Nat := 38982
def frameStart : Nat := 38923
def rule : BoundRule := .product (.predecessor 0 38980 .coefficient) (.predecessor 1 38981 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38980 .coefficient)
      LeftAuthority38978.bound (LeftAuthority38978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38981 .coefficient)
      LeftBound38976.bound (LeftBound38976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority38978.bound LeftBound38976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38978.bound, LeftBound38976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority38978.actual selector witness) * (LeftBound38976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound38982

namespace LeftBound38998
def owner : Owner := ⟨.program ⟨257⟩, ⟨9578⟩⟩
def transferEvent : Nat := 38998
def frameStart : Nat := 38923
def rule : BoundRule := .scale (.predecessor 0 38996 .coefficient) (.value (.predecessor 1 38997 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 38996 .coefficient)
      LeftAuthority38994.bound (LeftAuthority38994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 38997 .coefficient)
      LeftAuthority38985.bound (LeftAuthority38985.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority38985.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority38994.bound LeftAuthority38985.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38994.bound, LeftAuthority38985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority38994.actual selector witness) * (LeftAuthority38985.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound38998

namespace LeftBound39001
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 39001
def frameStart : Nat := 38923
def rule : BoundRule := .identity (.predecessor 0 39000 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39000 .coefficient)
      LeftAuthority38988.bound (LeftAuthority38988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38988.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38988.derived selector witness)

def rawBound : CoeffClass := LeftAuthority38988.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority38988.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39001

namespace LeftBound39005
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 39005
def frameStart : Nat := 38923
def rule : BoundRule := .product (.predecessor 0 39003 .coefficient) (.predecessor 1 39004 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39003 .coefficient)
      LeftBound39001.bound (LeftBound39001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39004 .coefficient)
      LeftBound38998.bound (LeftBound38998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound39001.bound LeftBound38998.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39001.bound, LeftBound38998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound39001.actual selector witness) * (LeftBound38998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39005

namespace LeftBound39010
def owner : Owner := ⟨.program ⟨257⟩, ⟨33265⟩⟩
def transferEvent : Nat := 39010
def frameStart : Nat := 38923
def rule : BoundRule := .sum [.predecessor 0 39008 .coefficient, .predecessor 1 39009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 39008 .coefficient)
      LeftBound39005.bound (LeftBound39005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 39009 .coefficient)
      LeftBound38982.bound (LeftBound38982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39005.bound, LeftBound38982.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39005.bound, LeftBound38982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound39005.actual selector witness, LeftBound38982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39010

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
