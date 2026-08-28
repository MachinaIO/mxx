import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard055
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1693
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1695
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1792

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound265648
def owner : Owner := ⟨.program ⟨257⟩, ⟨9237⟩⟩
def transferEvent : Nat := 265648
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265646 .coefficient, .predecessor 1 265647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265646 .coefficient)
      LeftBound265643.bound (LeftBound265643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265643.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265647 .coefficient)
      LeftBound251401.bound (LeftBound251401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events982.exact251403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265643.bound, LeftBound251401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265643.bound, LeftBound251401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265643.actual selector witness, LeftBound251401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265648

namespace LeftBound265652
def owner : Owner := ⟨.program ⟨257⟩, ⟨9238⟩⟩
def transferEvent : Nat := 265652
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265650 .coefficient, .predecessor 1 265651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265650 .coefficient)
      LeftBound265648.bound (LeftBound265648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265651 .coefficient)
      LeftAuthority265639.bound (LeftAuthority265639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265648.bound, LeftAuthority265639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265648.bound, LeftAuthority265639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265648.actual selector witness, LeftAuthority265639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265652

namespace LeftBound265653
def owner : Owner := ⟨.program ⟨257⟩, ⟨9238⟩⟩
def transferEvent : Nat := 265653
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨77⟩⟩]⟩ [⟨.result 265640 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265640 .coefficient)
      LeftAuthority265639.bound (LeftAuthority265639.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨77⟩⟩) (rawTerms := some (Proof.Events1037.exact265640RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265639.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265639.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority265639.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority265639.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265653

namespace LeftBound265658
def owner : Owner := ⟨.program ⟨257⟩, ⟨9629⟩⟩
def transferEvent : Nat := 265658
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265656 .coefficient) (.predecessor 1 265657 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265656 .coefficient)
      LeftBound265652.bound (LeftBound265652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265657 .coefficient)
      LeftBound15983.bound (LeftBound15983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15983.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265652.bound LeftBound15983.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265652.bound, LeftBound15983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265652.actual selector witness) * (LeftBound15983.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265658

namespace LeftBound265659
def owner : Owner := ⟨.program ⟨257⟩, ⟨9629⟩⟩
def transferEvent : Nat := 265659
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9583⟩⟩]⟩ [⟨.result 15980 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15980 .coefficient)
      LeftAuthority15979.bound (LeftAuthority15979.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9583⟩⟩) (rawTerms := some (Proof.Events062.exact15980RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15979.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15979.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15979.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265659

namespace LeftBound265660
def owner : Owner := ⟨.program ⟨257⟩, ⟨9629⟩⟩
def transferEvent : Nat := 265660
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 265655 .summary) (.transfer 265659) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265655 .summary)
      LeftBound265653.bound (LeftBound265653.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9238⟩⟩) (rawTerms := some (Proof.Events1037.exact265655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 265659)
      LeftBound265659.bound (LeftBound265659.actual selector witness) := by
  exact .transfer (LeftBound265659.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265653.bound LeftBound265659.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265653.bound, LeftBound265659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265653.actual selector witness) * (LeftBound265659.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265660

namespace LeftBound265686
def owner : Owner := ⟨.program ⟨257⟩, ⟨71089⟩⟩
def transferEvent : Nat := 265686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265684 .coefficient, .predecessor 1 265685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265684 .coefficient)
      LeftBound265658.bound (LeftBound265658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265685 .coefficient)
      LeftBound265636.bound (LeftBound265636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265636.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265658.bound, LeftBound265636.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265658.bound, LeftBound265636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265658.actual selector witness, LeftBound265636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265686

namespace LeftBound265706
def owner : Owner := ⟨.program ⟨257⟩, ⟨71089⟩⟩
def transferEvent : Nat := 265706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265683 .summary, .result 265638 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265683 .summary)
      LeftBound265660.bound (LeftBound265660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9629⟩⟩) (rawTerms := some (Proof.Events1037.exact265683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265638 .summary)
      LeftBound265637.bound (LeftBound265637.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71088⟩⟩) (rawTerms := some (Proof.Events1037.exact265638RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265660.bound, LeftBound265637.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002375679672372, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265660.bound, LeftBound265637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265660.actual selector witness, LeftBound265637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265706

namespace LeftBound265710
def owner : Owner := ⟨.program ⟨257⟩, ⟨71090⟩⟩
def transferEvent : Nat := 265710
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265708 .coefficient) (.predecessor 1 265709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265708 .coefficient)
      LeftBound265686.bound (LeftBound265686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265709 .coefficient)
      LeftBound16623.bound (LeftBound16623.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16623.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16623.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265686.bound LeftBound16623.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265686.bound, LeftBound16623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265686.actual selector witness) * (LeftBound16623.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265710

namespace LeftBound265711
def owner : Owner := ⟨.program ⟨257⟩, ⟨71090⟩⟩
def transferEvent : Nat := 265711
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9523⟩⟩]⟩ [⟨.result 16620 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16620 .coefficient)
      LeftAuthority16619.bound (LeftAuthority16619.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9523⟩⟩) (rawTerms := some (Proof.Events064.exact16620RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16619.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16619.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16619.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16619.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265711

namespace LeftBound265712
def owner : Owner := ⟨.program ⟨257⟩, ⟨71090⟩⟩
def transferEvent : Nat := 265712
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 265707 .summary) (.transfer 265711) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265707 .summary)
      LeftBound265706.bound (LeftBound265706.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71089⟩⟩) (rawTerms := some (Proof.Events1037.exact265707RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 265711)
      LeftBound265711.bound (LeftBound265711.actual selector witness) := by
  exact .transfer (LeftBound265711.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265706.bound LeftBound265711.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167293655866246524336865280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265706.bound, LeftBound265711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265706.actual selector witness) * (LeftBound265711.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265712

namespace LeftBound265774
def owner : Owner := ⟨.program ⟨257⟩, ⟨71091⟩⟩
def transferEvent : Nat := 265774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265772 .coefficient, .predecessor 1 265773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265772 .coefficient)
      LeftBound265710.bound (LeftBound265710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265773 .coefficient)
      LeftBound251291.bound (LeftBound251291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events981.exact251368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound251291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound251291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265710.bound, LeftBound251291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265710.bound, LeftBound251291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265710.actual selector witness, LeftBound251291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265774

namespace LeftBound265794
def owner : Owner := ⟨.program ⟨257⟩, ⟨71091⟩⟩
def transferEvent : Nat := 265794
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265771 .summary, .result 251368 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265771 .summary)
      LeftBound265712.bound (LeftBound265712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71090⟩⟩) (rawTerms := some (Proof.Events1038.exact265771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 251368 .summary)
      LeftBound251329.bound (LeftBound251329.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67369⟩⟩) (rawTerms := some (Proof.Events981.exact251368RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound251329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265712.bound, LeftBound251329.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854474880280923984914290088855535616, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265712.bound, LeftBound251329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265712.actual selector witness, LeftBound251329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265794

namespace LeftBound265798
def owner : Owner := ⟨.program ⟨257⟩, ⟨71092⟩⟩
def transferEvent : Nat := 265798
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265796 .coefficient) (.predecessor 1 265797 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265796 .coefficient)
      LeftBound265774.bound (LeftBound265774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265797 .coefficient)
      LeftBound16613.bound (LeftBound16613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16613.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265774.bound LeftBound16613.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265774.bound, LeftBound16613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265774.actual selector witness) * (LeftBound16613.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265798

namespace LeftBound265799
def owner : Owner := ⟨.program ⟨257⟩, ⟨71092⟩⟩
def transferEvent : Nat := 265799
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7105⟩⟩]⟩ [⟨.result 16610 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16610 .coefficient)
      LeftAuthority16609.bound (LeftAuthority16609.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7105⟩⟩) (rawTerms := some (Proof.Events064.exact16610RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16609.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16609.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16609.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16609.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265799

namespace LeftBound265800
def owner : Owner := ⟨.program ⟨257⟩, ⟨71092⟩⟩
def transferEvent : Nat := 265800
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 265795 .summary) (.transfer 265799) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265795 .summary)
      LeftBound265794.bound (LeftBound265794.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71091⟩⟩) (rawTerms := some (Proof.Events1038.exact265795RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 265799)
      LeftBound265799.bound (LeftBound265799.actual selector witness) := by
  exact .transfer (LeftBound265799.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265794.bound LeftBound265799.bound
def bound : CoeffClass := .finite ⟨7702113697398803698856913678033037845150209519672183236728648848208035840, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265794.bound, LeftBound265799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265794.actual selector witness) * (LeftBound265799.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265800

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
