import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1214

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound181730
def owner : Owner := ⟨.program ⟨257⟩, ⟨27953⟩⟩
def transferEvent : Nat := 181730
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 181728 .coefficient) (.predecessor 1 181729 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181728 .coefficient)
      LeftBound181724.bound (LeftBound181724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events709.exact181727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181729 .coefficient)
      LeftAuthority181662.bound (LeftAuthority181662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events709.exact181663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound181724.bound LeftAuthority181662.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181724.bound, LeftAuthority181662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound181724.actual selector witness) * (LeftAuthority181662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181730

namespace LeftBound181731
def owner : Owner := ⟨.program ⟨257⟩, ⟨27953⟩⟩
def transferEvent : Nat := 181731
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27952⟩⟩]⟩ [⟨.result 181663 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181663 .coefficient)
      LeftAuthority181662.bound (LeftAuthority181662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27952⟩⟩) (rawTerms := some (Proof.Events709.exact181663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181662.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority181662.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority181662.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound181731

namespace LeftBound181732
def owner : Owner := ⟨.program ⟨257⟩, ⟨27953⟩⟩
def transferEvent : Nat := 181732
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 181727 .summary) (.transfer 181731) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181727 .summary)
      LeftBound181726.bound (LeftBound181726.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨26173⟩⟩) (rawTerms := some (Proof.Events709.exact181727RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound181726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 181731)
      LeftBound181731.bound (LeftBound181731.actual selector witness) := by
  exact .transfer (LeftBound181731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound181726.bound LeftBound181731.bound
def bound : CoeffClass := .finite ⟨2997870350080095027200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181726.bound, LeftBound181731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound181726.actual selector witness) * (LeftBound181731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181732

namespace LeftBound181743
def owner : Owner := ⟨.program ⟨257⟩, ⟨26881⟩⟩
def transferEvent : Nat := 181743
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 181741 .coefficient) (.value (.predecessor 1 181742 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181741 .coefficient)
      LeftAuthority181739.bound (LeftAuthority181739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events709.exact181740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181742 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority181739.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181739.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority181739.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound181743

namespace LeftBound181747
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def transferEvent : Nat := 181747
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 181745 .coefficient) (.predecessor 1 181746 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181745 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181746 .coefficient)
      LeftBound181743.bound (LeftBound181743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events709.exact181744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181743.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound181743.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound181743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound181743.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181747

namespace LeftBound181748
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def transferEvent : Nat := 181748
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨26879⟩⟩]⟩ [⟨.result 181740 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181740 .coefficient)
      LeftAuthority181739.bound (LeftAuthority181739.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26879⟩⟩) (rawTerms := some (Proof.Events709.exact181740RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181739.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority181739.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority181739.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound181748

namespace LeftBound181749
def owner : Owner := ⟨.program ⟨257⟩, ⟨26882⟩⟩
def transferEvent : Nat := 181749
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 181748) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 181748)
      LeftBound181748.bound (LeftBound181748.actual selector witness) := by
  exact .transfer (LeftBound181748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound181748.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound181748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound181748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181749

namespace LeftBound181828
def owner : Owner := ⟨.program ⟨257⟩, ⟨26167⟩⟩
def transferEvent : Nat := 181828
def frameStart : Nat := 181799
def rule : BoundRule := .product (.predecessor 0 181826 .coefficient) (.predecessor 1 181827 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181826 .coefficient)
      LeftAuthority181824.bound (LeftAuthority181824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181824.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181827 .coefficient)
      LeftAuthority181821.bound (LeftAuthority181821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181821.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority181824.bound LeftAuthority181821.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181824.bound, LeftAuthority181821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority181824.actual selector witness) * (LeftAuthority181821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181828

namespace LeftBound181832
def owner : Owner := ⟨.program ⟨257⟩, ⟨26168⟩⟩
def transferEvent : Nat := 181832
def frameStart : Nat := 181799
def rule : BoundRule := .identity (.predecessor 0 181831 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181831 .coefficient)
      LeftBound181828.bound (LeftBound181828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181828.derived selector witness)

def rawBound : CoeffClass := LeftBound181828.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound181828.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound181832

namespace LeftBound181849
def owner : Owner := ⟨.program ⟨257⟩, ⟨27698⟩⟩
def transferEvent : Nat := 181849
def frameStart : Nat := 181799
def rule : BoundRule := .sum [.predecessor 0 181847 .coefficient, .predecessor 1 181848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181847 .coefficient)
      LeftBound181832.bound (LeftBound181832.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound181832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181848 .coefficient)
      LeftAuthority181845.bound (LeftAuthority181845.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority181845.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound181832.bound, LeftAuthority181845.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181832.bound, LeftAuthority181845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound181832.actual selector witness, LeftAuthority181845.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound181849

namespace LeftBound181852
def owner : Owner := ⟨.program ⟨257⟩, ⟨27699⟩⟩
def transferEvent : Nat := 181852
def frameStart : Nat := 181799
def rule : BoundRule := .identity (.predecessor 0 181851 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181851 .coefficient)
      LeftBound181849.bound (LeftBound181849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound181849.derived selector witness)

def rawBound : CoeffClass := LeftBound181849.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181849.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound181849.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound181852

namespace LeftBound181858
def owner : Owner := ⟨.program ⟨257⟩, ⟨27700⟩⟩
def transferEvent : Nat := 181858
def frameStart : Nat := 181799
def rule : BoundRule := .product (.predecessor 0 181856 .coefficient) (.predecessor 1 181857 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181856 .coefficient)
      LeftAuthority181854.bound (LeftAuthority181854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181857 .coefficient)
      LeftBound181852.bound (LeftBound181852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181852.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority181854.bound LeftBound181852.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181854.bound, LeftBound181852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority181854.actual selector witness) * (LeftBound181852.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181858

namespace LeftBound181874
def owner : Owner := ⟨.program ⟨257⟩, ⟨9545⟩⟩
def transferEvent : Nat := 181874
def frameStart : Nat := 181799
def rule : BoundRule := .scale (.predecessor 0 181872 .coefficient) (.value (.predecessor 1 181873 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181872 .coefficient)
      LeftAuthority181870.bound (LeftAuthority181870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181873 .coefficient)
      LeftAuthority181861.bound (LeftAuthority181861.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority181861.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority181870.bound LeftAuthority181861.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181870.bound, LeftAuthority181861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority181870.actual selector witness) * (LeftAuthority181861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound181874

namespace LeftBound181877
def owner : Owner := ⟨.program ⟨257⟩, ⟨7295⟩⟩
def transferEvent : Nat := 181877
def frameStart : Nat := 181799
def rule : BoundRule := .identity (.predecessor 0 181876 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181876 .coefficient)
      LeftAuthority181864.bound (LeftAuthority181864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority181864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority181864.derived selector witness)

def rawBound : CoeffClass := LeftAuthority181864.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority181864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority181864.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound181877

namespace LeftBound181881
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def transferEvent : Nat := 181881
def frameStart : Nat := 181799
def rule : BoundRule := .product (.predecessor 0 181879 .coefficient) (.predecessor 1 181880 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181879 .coefficient)
      LeftBound181877.bound (LeftBound181877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181880 .coefficient)
      LeftBound181874.bound (LeftBound181874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181874.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound181877.bound LeftBound181874.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181877.bound, LeftBound181874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound181877.actual selector witness) * (LeftBound181874.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound181881

namespace LeftBound181886
def owner : Owner := ⟨.program ⟨257⟩, ⟨27701⟩⟩
def transferEvent : Nat := 181886
def frameStart : Nat := 181799
def rule : BoundRule := .sum [.predecessor 0 181884 .coefficient, .predecessor 1 181885 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 181884 .coefficient)
      LeftBound181881.bound (LeftBound181881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 181885 .coefficient)
      LeftBound181858.bound (LeftBound181858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound181881.bound, LeftBound181858.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181881.bound, LeftBound181858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound181881.actual selector witness, LeftBound181858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound181886

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
