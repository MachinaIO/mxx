import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard378
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard404

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound73069
def owner : Owner := ⟨.program ⟨257⟩, ⟨28460⟩⟩
def transferEvent : Nat := 73069
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73067 .coefficient) (.predecessor 1 73068 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73067 .coefficient)
      LeftBound64926.bound (LeftBound64926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73068 .coefficient)
      LeftAuthority73065.bound (LeftAuthority73065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64926.bound LeftAuthority73065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64926.bound, LeftAuthority73065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64926.actual selector witness) * (LeftAuthority73065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73069

namespace LeftBound73070
def owner : Owner := ⟨.program ⟨257⟩, ⟨28460⟩⟩
def transferEvent : Nat := 73070
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨28458⟩⟩]⟩ [⟨.result 73066 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73066 .coefficient)
      LeftAuthority73065.bound (LeftAuthority73065.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨28458⟩⟩) (rawTerms := some (Proof.Events285.exact73066RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73065.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73065.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73065.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority73065.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73070

namespace LeftBound73071
def owner : Owner := ⟨.program ⟨257⟩, ⟨28460⟩⟩
def transferEvent : Nat := 73071
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64930 .summary) (.transfer 73070) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64930 .summary)
      LeftBound64929.bound (LeftBound64929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27998⟩⟩) (rawTerms := some (Proof.Events253.exact64930RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64929.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 73070)
      LeftBound73070.bound (LeftBound73070.actual selector witness) := by
  exact .transfer (LeftBound73070.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound64929.bound LeftBound73070.bound
def bound : CoeffClass := .finite ⟨32191557518723128098041228165120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64929.bound, LeftBound73070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound64929.actual selector witness) * (LeftBound73070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73071

namespace LeftBound73082
def owner : Owner := ⟨.program ⟨257⟩, ⟨27294⟩⟩
def transferEvent : Nat := 73082
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 73080 .coefficient) (.value (.predecessor 1 73081 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73080 .coefficient)
      LeftAuthority73078.bound (LeftAuthority73078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73081 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority73078.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73078.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority73078.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound73082

namespace LeftBound73086
def owner : Owner := ⟨.program ⟨257⟩, ⟨27295⟩⟩
def transferEvent : Nat := 73086
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 73084 .coefficient) (.predecessor 1 73085 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73084 .coefficient)
      LeftBound61367.bound (LeftBound61367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73085 .coefficient)
      LeftBound73082.bound (LeftBound73082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73082.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61367.bound LeftBound73082.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61367.bound, LeftBound73082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61367.actual selector witness) * (LeftBound73082.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73086

namespace LeftBound73087
def owner : Owner := ⟨.program ⟨257⟩, ⟨27295⟩⟩
def transferEvent : Nat := 73087
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27292⟩⟩]⟩ [⟨.result 73079 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 73079 .coefficient)
      LeftAuthority73078.bound (LeftAuthority73078.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27292⟩⟩) (rawTerms := some (Proof.Events285.exact73079RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73078.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority73078.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority73078.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound73087

namespace LeftBound73088
def owner : Owner := ⟨.program ⟨257⟩, ⟨27295⟩⟩
def transferEvent : Nat := 73088
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 61370 .summary) (.transfer 73087) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 61370 .summary)
      LeftBound61368.bound (LeftBound61368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10792⟩⟩) (rawTerms := some (Proof.Events239.exact61370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 73087)
      LeftBound73087.bound (LeftBound73087.actual selector witness) := by
  exact .transfer (LeftBound73087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound61368.bound LeftBound73087.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61368.bound, LeftBound73087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound61368.actual selector witness) * (LeftBound73087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73088

namespace LeftBound73183
def owner : Owner := ⟨.program ⟨257⟩, ⟨26465⟩⟩
def transferEvent : Nat := 73183
def frameStart : Nat := 73144
def rule : BoundRule := .identity (.predecessor 0 73182 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73182 .coefficient)
      LeftAuthority73180.bound (LeftAuthority73180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73180.derived selector witness)

def rawBound : CoeffClass := LeftAuthority73180.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority73180.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73183

namespace LeftBound73200
def owner : Owner := ⟨.program ⟨257⟩, ⟨27794⟩⟩
def transferEvent : Nat := 73200
def frameStart : Nat := 73144
def rule : BoundRule := .sum [.predecessor 0 73198 .coefficient, .predecessor 1 73199 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73198 .coefficient)
      LeftBound73183.bound (LeftBound73183.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73199 .coefficient)
      LeftAuthority73196.bound (LeftAuthority73196.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority73196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73183.bound, LeftAuthority73196.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73183.bound, LeftAuthority73196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound73183.actual selector witness, LeftAuthority73196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73200

namespace LeftBound73203
def owner : Owner := ⟨.program ⟨257⟩, ⟨27795⟩⟩
def transferEvent : Nat := 73203
def frameStart : Nat := 73144
def rule : BoundRule := .identity (.predecessor 0 73202 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73202 .coefficient)
      LeftBound73200.bound (LeftBound73200.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound73200.derived selector witness)

def rawBound : CoeffClass := LeftBound73200.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound73200.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound73203

namespace LeftBound73209
def owner : Owner := ⟨.program ⟨257⟩, ⟨27796⟩⟩
def transferEvent : Nat := 73209
def frameStart : Nat := 73144
def rule : BoundRule := .product (.predecessor 0 73207 .coefficient) (.predecessor 1 73208 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73207 .coefficient)
      LeftAuthority73205.bound (LeftAuthority73205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73205.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73208 .coefficient)
      LeftBound73203.bound (LeftBound73203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73203.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority73205.bound LeftBound73203.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73205.bound, LeftBound73203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority73205.actual selector witness) * (LeftBound73203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73209

namespace LeftBound73217
def owner : Owner := ⟨.program ⟨257⟩, ⟨27797⟩⟩
def transferEvent : Nat := 73217
def frameStart : Nat := 73144
def rule : BoundRule := .sum [.predecessor 0 73215 .coefficient, .predecessor 1 73216 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73215 .coefficient)
      LeftAuthority73213.bound (LeftAuthority73213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73213.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73213.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73216 .coefficient)
      LeftBound73209.bound (LeftBound73209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73209.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73213.bound, LeftBound73209.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73213.bound, LeftBound73209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority73213.actual selector witness, LeftBound73209.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73217

namespace LeftBound73221
def owner : Owner := ⟨.program ⟨257⟩, ⟨28459⟩⟩
def transferEvent : Nat := 73221
def frameStart : Nat := 73144
def rule : BoundRule := .product (.predecessor 0 73219 .coefficient) (.predecessor 1 73220 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73219 .coefficient)
      LeftBound73217.bound (LeftBound73217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73220 .coefficient)
      LeftAuthority73194.bound (LeftAuthority73194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73194.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound73217.bound LeftAuthority73194.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73217.bound, LeftAuthority73194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound73217.actual selector witness) * (LeftAuthority73194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73221

namespace LeftBound73232
def owner : Owner := ⟨.program ⟨257⟩, ⟨26715⟩⟩
def transferEvent : Nat := 73232
def frameStart : Nat := 73144
def rule : BoundRule := .product (.predecessor 0 73230 .coefficient) (.predecessor 1 73231 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73230 .coefficient)
      LeftAuthority73205.bound (LeftAuthority73205.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact73206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73205.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73205.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73231 .coefficient)
      LeftAuthority73228.bound (LeftAuthority73228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73228.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73228.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority73205.bound LeftAuthority73228.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73205.bound, LeftAuthority73228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority73205.actual selector witness) * (LeftAuthority73228.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound73232

namespace LeftBound73240
def owner : Owner := ⟨.program ⟨257⟩, ⟨26716⟩⟩
def transferEvent : Nat := 73240
def frameStart : Nat := 73144
def rule : BoundRule := .sum [.predecessor 0 73238 .coefficient, .predecessor 1 73239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73238 .coefficient)
      LeftAuthority73236.bound (LeftAuthority73236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority73236.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority73236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73239 .coefficient)
      LeftBound73232.bound (LeftBound73232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73232.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority73236.bound, LeftBound73232.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority73236.bound, LeftBound73232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority73236.actual selector witness, LeftBound73232.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73240

namespace LeftBound73244
def owner : Owner := ⟨.program ⟨257⟩, ⟨28463⟩⟩
def transferEvent : Nat := 73244
def frameStart : Nat := 73144
def rule : BoundRule := .sum [.predecessor 0 73242 .coefficient, .predecessor 1 73243 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 73242 .coefficient)
      LeftBound73240.bound (LeftBound73240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 73243 .coefficient)
      LeftBound73221.bound (LeftBound73221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound73240.bound, LeftBound73221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73240.bound, LeftBound73221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound73240.actual selector witness, LeftBound73221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound73244

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
