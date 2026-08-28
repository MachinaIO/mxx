import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1245

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound191765
def owner : Owner := ⟨.program ⟨257⟩, ⟨23960⟩⟩
def transferEvent : Nat := 191765
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 191763 .coefficient) (.predecessor 1 191764 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191763 .coefficient)
      LeftBound185782.bound (LeftBound185782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events725.exact185786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191764 .coefficient)
      LeftAuthority191761.bound (LeftAuthority191761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound185782.bound LeftAuthority191761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185782.bound, LeftAuthority191761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound185782.actual selector witness) * (LeftAuthority191761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191765

namespace LeftBound191766
def owner : Owner := ⟨.program ⟨257⟩, ⟨23960⟩⟩
def transferEvent : Nat := 191766
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23958⟩⟩]⟩ [⟨.result 191762 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 191762 .coefficient)
      LeftAuthority191761.bound (LeftAuthority191761.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23958⟩⟩) (rawTerms := some (Proof.Events749.exact191762RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191761.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191761.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority191761.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority191761.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound191766

namespace LeftBound191767
def owner : Owner := ⟨.program ⟨257⟩, ⟨23960⟩⟩
def transferEvent : Nat := 191767
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 185786 .summary) (.transfer 191766) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185786 .summary)
      LeftBound185785.bound (LeftBound185785.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23474⟩⟩) (rawTerms := some (Proof.Events725.exact185786RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 191766)
      LeftBound191766.bound (LeftBound191766.actual selector witness) := by
  exact .transfer (LeftBound191766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound185785.bound LeftBound191766.bound
def bound : CoeffClass := .finite ⟨32189003662929192193909661368320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185785.bound, LeftBound191766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound185785.actual selector witness) * (LeftBound191766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191767

namespace LeftBound191778
def owner : Owner := ⟨.program ⟨257⟩, ⟨22734⟩⟩
def transferEvent : Nat := 191778
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 191776 .coefficient) (.value (.predecessor 1 191777 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191776 .coefficient)
      LeftAuthority191774.bound (LeftAuthority191774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191777 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority191774.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191774.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority191774.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound191778

namespace LeftBound191782
def owner : Owner := ⟨.program ⟨257⟩, ⟨22735⟩⟩
def transferEvent : Nat := 191782
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 191780 .coefficient) (.predecessor 1 191781 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191780 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191781 .coefficient)
      LeftBound191778.bound (LeftBound191778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound191778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound191778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound191778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191782

namespace LeftBound191783
def owner : Owner := ⟨.program ⟨257⟩, ⟨22735⟩⟩
def transferEvent : Nat := 191783
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22732⟩⟩]⟩ [⟨.result 191775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 191775 .coefficient)
      LeftAuthority191774.bound (LeftAuthority191774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22732⟩⟩) (rawTerms := some (Proof.Events749.exact191775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority191774.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority191774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound191783

namespace LeftBound191784
def owner : Owner := ⟨.program ⟨257⟩, ⟨22735⟩⟩
def transferEvent : Nat := 191784
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 191783) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 191783)
      LeftBound191783.bound (LeftBound191783.actual selector witness) := by
  exact .transfer (LeftBound191783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound191783.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound191783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound191783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191784

namespace LeftBound191879
def owner : Owner := ⟨.program ⟨257⟩, ⟨21833⟩⟩
def transferEvent : Nat := 191879
def frameStart : Nat := 191840
def rule : BoundRule := .identity (.predecessor 0 191878 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191878 .coefficient)
      LeftAuthority191876.bound (LeftAuthority191876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191876.derived selector witness)

def rawBound : CoeffClass := LeftAuthority191876.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority191876.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound191879

namespace LeftBound191896
def owner : Owner := ⟨.program ⟨257⟩, ⟨23298⟩⟩
def transferEvent : Nat := 191896
def frameStart : Nat := 191840
def rule : BoundRule := .sum [.predecessor 0 191894 .coefficient, .predecessor 1 191895 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191894 .coefficient)
      LeftBound191879.bound (LeftBound191879.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound191879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191895 .coefficient)
      LeftAuthority191892.bound (LeftAuthority191892.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority191892.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound191879.bound, LeftAuthority191892.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound191879.bound, LeftAuthority191892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound191879.actual selector witness, LeftAuthority191892.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound191896

namespace LeftBound191899
def owner : Owner := ⟨.program ⟨257⟩, ⟨23299⟩⟩
def transferEvent : Nat := 191899
def frameStart : Nat := 191840
def rule : BoundRule := .identity (.predecessor 0 191898 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191898 .coefficient)
      LeftBound191896.bound (LeftBound191896.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound191896.derived selector witness)

def rawBound : CoeffClass := LeftBound191896.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound191896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound191896.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound191899

namespace LeftBound191905
def owner : Owner := ⟨.program ⟨257⟩, ⟨23300⟩⟩
def transferEvent : Nat := 191905
def frameStart : Nat := 191840
def rule : BoundRule := .product (.predecessor 0 191903 .coefficient) (.predecessor 1 191904 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191903 .coefficient)
      LeftAuthority191901.bound (LeftAuthority191901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191904 .coefficient)
      LeftBound191899.bound (LeftBound191899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority191901.bound LeftBound191899.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191901.bound, LeftBound191899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority191901.actual selector witness) * (LeftBound191899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191905

namespace LeftBound191913
def owner : Owner := ⟨.program ⟨257⟩, ⟨23301⟩⟩
def transferEvent : Nat := 191913
def frameStart : Nat := 191840
def rule : BoundRule := .sum [.predecessor 0 191911 .coefficient, .predecessor 1 191912 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191911 .coefficient)
      LeftAuthority191909.bound (LeftAuthority191909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191912 .coefficient)
      LeftBound191905.bound (LeftBound191905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority191909.bound, LeftBound191905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191909.bound, LeftBound191905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority191909.actual selector witness, LeftBound191905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound191913

namespace LeftBound191917
def owner : Owner := ⟨.program ⟨257⟩, ⟨23959⟩⟩
def transferEvent : Nat := 191917
def frameStart : Nat := 191840
def rule : BoundRule := .product (.predecessor 0 191915 .coefficient) (.predecessor 1 191916 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191915 .coefficient)
      LeftBound191913.bound (LeftBound191913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191916 .coefficient)
      LeftAuthority191890.bound (LeftAuthority191890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound191913.bound LeftAuthority191890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound191913.bound, LeftAuthority191890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound191913.actual selector witness) * (LeftAuthority191890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191917

namespace LeftBound191928
def owner : Owner := ⟨.program ⟨257⟩, ⟨22141⟩⟩
def transferEvent : Nat := 191928
def frameStart : Nat := 191840
def rule : BoundRule := .product (.predecessor 0 191926 .coefficient) (.predecessor 1 191927 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191926 .coefficient)
      LeftAuthority191901.bound (LeftAuthority191901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191927 .coefficient)
      LeftAuthority191924.bound (LeftAuthority191924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority191901.bound LeftAuthority191924.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191901.bound, LeftAuthority191924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority191901.actual selector witness) * (LeftAuthority191924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound191928

namespace LeftBound191936
def owner : Owner := ⟨.program ⟨257⟩, ⟨22142⟩⟩
def transferEvent : Nat := 191936
def frameStart : Nat := 191840
def rule : BoundRule := .sum [.predecessor 0 191934 .coefficient, .predecessor 1 191935 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191934 .coefficient)
      LeftAuthority191932.bound (LeftAuthority191932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority191932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority191932.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191935 .coefficient)
      LeftBound191928.bound (LeftBound191928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority191932.bound, LeftBound191928.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority191932.bound, LeftBound191928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority191932.actual selector witness, LeftBound191928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound191936

namespace LeftBound191940
def owner : Owner := ⟨.program ⟨257⟩, ⟨23964⟩⟩
def transferEvent : Nat := 191940
def frameStart : Nat := 191840
def rule : BoundRule := .sum [.predecessor 0 191938 .coefficient, .predecessor 1 191939 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 191938 .coefficient)
      LeftBound191936.bound (LeftBound191936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 191939 .coefficient)
      LeftBound191917.bound (LeftBound191917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events749.exact191922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound191917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound191917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound191936.bound, LeftBound191917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound191936.bound, LeftBound191917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound191936.actual selector witness, LeftBound191917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound191940

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
