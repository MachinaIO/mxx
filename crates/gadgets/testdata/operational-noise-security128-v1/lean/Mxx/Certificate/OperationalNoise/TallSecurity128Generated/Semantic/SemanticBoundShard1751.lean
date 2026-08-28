import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1696
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1697
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1750

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound258697
def owner : Owner := ⟨.program ⟨257⟩, ⟨21030⟩⟩
def transferEvent : Nat := 258697
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 258692 .summary) (.transfer 258696) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258692 .summary)
      LeftBound258690.bound (LeftBound258690.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21029⟩⟩) (rawTerms := some (Proof.Events1010.exact258692RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 258696)
      LeftBound258696.bound (LeftBound258696.actual selector witness) := by
  exact .transfer (LeftBound258696.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258690.bound LeftBound258696.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258690.bound, LeftBound258696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258690.actual selector witness) * (LeftBound258696.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258697

namespace LeftBound258705
def owner : Owner := ⟨.program ⟨257⟩, ⟨21381⟩⟩
def transferEvent : Nat := 258705
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 258703 .coefficient, .predecessor 1 258704 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258703 .coefficient)
      LeftBound258695.bound (LeftBound258695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258704 .coefficient)
      LeftBound258667.bound (LeftBound258667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258695.bound, LeftBound258667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258695.bound, LeftBound258667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258695.actual selector witness, LeftBound258667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258705

namespace LeftBound258707
def owner : Owner := ⟨.program ⟨257⟩, ⟨21381⟩⟩
def transferEvent : Nat := 258707
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 258702 .summary, .result 258672 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258702 .summary)
      LeftBound258697.bound (LeftBound258697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21030⟩⟩) (rawTerms := some (Proof.Events1010.exact258702RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258672 .summary)
      LeftBound258669.bound (LeftBound258669.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21380⟩⟩) (rawTerms := some (Proof.Events1010.exact258672RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258669.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258697.bound, LeftBound258669.bound]
def bound : CoeffClass := .finite ⟨279176282112, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258697.bound, LeftBound258669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258697.actual selector witness, LeftBound258669.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258707

namespace LeftBound258711
def owner : Owner := ⟨.program ⟨257⟩, ⟨23385⟩⟩
def transferEvent : Nat := 258711
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 258709 .coefficient) (.predecessor 1 258710 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258709 .coefficient)
      LeftBound258705.bound (LeftBound258705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258710 .coefficient)
      LeftAuthority258643.bound (LeftAuthority258643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258643.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258705.bound LeftAuthority258643.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258705.bound, LeftAuthority258643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258705.actual selector witness) * (LeftAuthority258643.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258711

namespace LeftBound258712
def owner : Owner := ⟨.program ⟨257⟩, ⟨23385⟩⟩
def transferEvent : Nat := 258712
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23384⟩⟩]⟩ [⟨.result 258644 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258644 .coefficient)
      LeftAuthority258643.bound (LeftAuthority258643.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23384⟩⟩) (rawTerms := some (Proof.Events1010.exact258644RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258643.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority258643.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority258643.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound258712

namespace LeftBound258713
def owner : Owner := ⟨.program ⟨257⟩, ⟨23385⟩⟩
def transferEvent : Nat := 258713
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 258708 .summary) (.transfer 258712) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258708 .summary)
      LeftBound258707.bound (LeftBound258707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨21381⟩⟩) (rawTerms := some (Proof.Events1010.exact258708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound258707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 258712)
      LeftBound258712.bound (LeftBound258712.actual selector witness) := by
  exact .transfer (LeftBound258712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound258707.bound LeftBound258712.bound
def bound : CoeffClass := .finite ⟨2997632503724774522880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258707.bound, LeftBound258712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound258707.actual selector witness) * (LeftBound258712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258713

namespace LeftBound258724
def owner : Owner := ⟨.program ⟨257⟩, ⟨22321⟩⟩
def transferEvent : Nat := 258724
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 258722 .coefficient) (.value (.predecessor 1 258723 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258722 .coefficient)
      LeftAuthority258720.bound (LeftAuthority258720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258723 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority258720.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258720.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority258720.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound258724

namespace LeftBound258728
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def transferEvent : Nat := 258728
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 258726 .coefficient) (.predecessor 1 258727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258726 .coefficient)
      LeftBound251492.bound (LeftBound251492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258727 .coefficient)
      LeftBound258724.bound (LeftBound258724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251492.bound LeftBound258724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251492.bound, LeftBound258724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251492.actual selector witness) * (LeftBound258724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258728

namespace LeftBound258729
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def transferEvent : Nat := 258729
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22319⟩⟩]⟩ [⟨.result 258721 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 258721 .coefficient)
      LeftAuthority258720.bound (LeftAuthority258720.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22319⟩⟩) (rawTerms := some (Proof.Events1010.exact258721RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258720.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority258720.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority258720.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound258729

namespace LeftBound258730
def owner : Owner := ⟨.program ⟨257⟩, ⟨22322⟩⟩
def transferEvent : Nat := 258730
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 251495 .summary) (.transfer 258729) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251495 .summary)
      LeftBound251493.bound (LeftBound251493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events982.exact251495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 258729)
      LeftBound258729.bound (LeftBound258729.actual selector witness) := by
  exact .transfer (LeftBound258729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound251493.bound LeftBound258729.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251493.bound, LeftBound258729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound251493.actual selector witness) * (LeftBound258729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258730

namespace LeftBound258809
def owner : Owner := ⟨.program ⟨257⟩, ⟨21375⟩⟩
def transferEvent : Nat := 258809
def frameStart : Nat := 258780
def rule : BoundRule := .product (.predecessor 0 258807 .coefficient) (.predecessor 1 258808 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258807 .coefficient)
      LeftAuthority258805.bound (LeftAuthority258805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258805.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258808 .coefficient)
      LeftAuthority258802.bound (LeftAuthority258802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority258805.bound LeftAuthority258802.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258805.bound, LeftAuthority258802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority258805.actual selector witness) * (LeftAuthority258802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258809

namespace LeftBound258813
def owner : Owner := ⟨.program ⟨257⟩, ⟨21376⟩⟩
def transferEvent : Nat := 258813
def frameStart : Nat := 258780
def rule : BoundRule := .identity (.predecessor 0 258812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258812 .coefficient)
      LeftBound258809.bound (LeftBound258809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1010.exact258811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258809.derived selector witness)

def rawBound : CoeffClass := LeftBound258809.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound258809.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound258813

namespace LeftBound258830
def owner : Owner := ⟨.program ⟨257⟩, ⟨23186⟩⟩
def transferEvent : Nat := 258830
def frameStart : Nat := 258780
def rule : BoundRule := .sum [.predecessor 0 258828 .coefficient, .predecessor 1 258829 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258828 .coefficient)
      LeftBound258813.bound (LeftBound258813.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound258813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258829 .coefficient)
      LeftAuthority258826.bound (LeftAuthority258826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority258826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound258813.bound, LeftAuthority258826.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258813.bound, LeftAuthority258826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound258813.actual selector witness, LeftAuthority258826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound258830

namespace LeftBound258833
def owner : Owner := ⟨.program ⟨257⟩, ⟨23187⟩⟩
def transferEvent : Nat := 258833
def frameStart : Nat := 258780
def rule : BoundRule := .identity (.predecessor 0 258832 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258832 .coefficient)
      LeftBound258830.bound (LeftBound258830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound258830.derived selector witness)

def rawBound : CoeffClass := LeftBound258830.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound258830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound258830.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound258833

namespace LeftBound258839
def owner : Owner := ⟨.program ⟨257⟩, ⟨23188⟩⟩
def transferEvent : Nat := 258839
def frameStart : Nat := 258780
def rule : BoundRule := .product (.predecessor 0 258837 .coefficient) (.predecessor 1 258838 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258837 .coefficient)
      LeftAuthority258835.bound (LeftAuthority258835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1011.exact258836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258838 .coefficient)
      LeftBound258833.bound (LeftBound258833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1011.exact258834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound258833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound258833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority258835.bound LeftBound258833.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258835.bound, LeftBound258833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority258835.actual selector witness) * (LeftBound258833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound258839

namespace LeftBound258855
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 258855
def frameStart : Nat := 258780
def rule : BoundRule := .scale (.predecessor 0 258853 .coefficient) (.value (.predecessor 1 258854 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 258853 .coefficient)
      LeftAuthority258851.bound (LeftAuthority258851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1011.exact258852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority258851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority258851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 258854 .coefficient)
      LeftAuthority258842.bound (LeftAuthority258842.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority258842.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority258851.bound LeftAuthority258842.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority258851.bound, LeftAuthority258842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority258851.actual selector witness) * (LeftAuthority258842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound258855

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
