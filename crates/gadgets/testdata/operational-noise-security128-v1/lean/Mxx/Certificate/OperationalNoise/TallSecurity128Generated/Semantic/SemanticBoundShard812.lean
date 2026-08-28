import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard099
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard779
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard782
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard811

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound123663
def owner : Owner := ⟨.program ⟨257⟩, ⟨25685⟩⟩
def transferEvent : Nat := 123663
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩ [⟨.result 21080 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21080 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21079.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21079.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound123663

namespace LeftBound123668
def owner : Owner := ⟨.program ⟨257⟩, ⟨65340⟩⟩
def transferEvent : Nat := 123668
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 123666 .coefficient) (.predecessor 1 123667 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123666 .coefficient)
      LeftBound123662.bound (LeftBound123662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123667 .coefficient)
      LeftAuthority5516.bound (LeftAuthority5516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound123662.bound LeftAuthority5516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123662.bound, LeftAuthority5516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound123662.actual selector witness) * (LeftAuthority5516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123668

namespace LeftBound123669
def owner : Owner := ⟨.program ⟨257⟩, ⟨65340⟩⟩
def transferEvent : Nat := 123669
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨65337⟩⟩], []⟩ [⟨.result 5517 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 5517 .coefficient)
      LeftAuthority5516.bound (LeftAuthority5516.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨65337⟩⟩) (rawTerms := some (Proof.Events021.exact5517RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5516.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5516.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority5516.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound123669

namespace LeftBound123670
def owner : Owner := ⟨.program ⟨257⟩, ⟨65340⟩⟩
def transferEvent : Nat := 123670
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 123665 .summary) (.transfer 123669) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123665 .summary)
      LeftBound123663.bound (LeftBound123663.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25685⟩⟩) (rawTerms := some (Proof.Events483.exact123665RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 123669)
      LeftBound123669.bound (LeftBound123669.actual selector witness) := by
  exact .transfer (LeftBound123669.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound123663.bound LeftBound123669.bound
def bound : CoeffClass := .finite ⟨23855104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123663.bound, LeftBound123669.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound123663.actual selector witness) * (LeftBound123669.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123670

namespace LeftBound123676
def owner : Owner := ⟨.program ⟨257⟩, ⟨65341⟩⟩
def transferEvent : Nat := 123676
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 123674 .coefficient) (.predecessor 1 123675 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123674 .coefficient)
      LeftAuthority5516.bound (LeftAuthority5516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5516.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123675 .coefficient)
      LeftBound119776.bound (LeftBound119776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority5516.bound LeftBound119776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5516.bound, LeftBound119776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority5516.actual selector witness) * (LeftBound119776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound123676

namespace LeftBound123681
def owner : Owner := ⟨.program ⟨257⟩, ⟨8144⟩⟩
def transferEvent : Nat := 123681
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 123679 .coefficient) (.predecessor 1 123680 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123679 .coefficient)
      LeftBound119647.bound (LeftBound119647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events467.exact119648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123680 .coefficient)
      LeftBound21128.bound (LeftBound21128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound119647.bound LeftBound21128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119647.bound, LeftBound21128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound119647.actual selector witness) * (LeftBound21128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123681

namespace LeftBound123686
def owner : Owner := ⟨.program ⟨257⟩, ⟨65342⟩⟩
def transferEvent : Nat := 123686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123684 .coefficient, .predecessor 1 123685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123684 .coefficient)
      LeftBound123681.bound (LeftBound123681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123685 .coefficient)
      LeftBound123676.bound (LeftBound123676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123681.bound, LeftBound123676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123681.bound, LeftBound123676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123681.actual selector witness, LeftBound123676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123686

namespace LeftBound123690
def owner : Owner := ⟨.program ⟨257⟩, ⟨65343⟩⟩
def transferEvent : Nat := 123690
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123688 .coefficient, .predecessor 1 123689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123688 .coefficient)
      LeftBound123686.bound (LeftBound123686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123689 .coefficient)
      LeftBound21120.bound (LeftBound21120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123686.bound, LeftBound21120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123686.bound, LeftBound21120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123686.actual selector witness, LeftBound21120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123690

namespace LeftBound123691
def owner : Owner := ⟨.program ⟨257⟩, ⟨65343⟩⟩
def transferEvent : Nat := 123691
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨120⟩⟩]⟩ [⟨.result 21121 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21121 .coefficient)
      LeftBound21120.bound (LeftBound21120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨120⟩⟩) (rawTerms := some (Proof.Events082.exact21121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21120.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound123691

namespace LeftBound123696
def owner : Owner := ⟨.program ⟨257⟩, ⟨65344⟩⟩
def transferEvent : Nat := 123696
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 123694 .coefficient) (.predecessor 1 123695 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123694 .coefficient)
      LeftBound123690.bound (LeftBound123690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123695 .coefficient)
      LeftBound21117.bound (LeftBound21117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21117.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound123690.bound LeftBound21117.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123690.bound, LeftBound21117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound123690.actual selector witness) * (LeftBound21117.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123696

namespace LeftBound123697
def owner : Owner := ⟨.program ⟨257⟩, ⟨65344⟩⟩
def transferEvent : Nat := 123697
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩ [⟨.result 21114 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21114 .coefficient)
      LeftAuthority21113.bound (LeftAuthority21113.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9541⟩⟩) (rawTerms := some (Proof.Events082.exact21114RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21113.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21113.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21113.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound123697

namespace LeftBound123698
def owner : Owner := ⟨.program ⟨257⟩, ⟨65344⟩⟩
def transferEvent : Nat := 123698
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 123693 .summary) (.transfer 123697) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123693 .summary)
      LeftBound123691.bound (LeftBound123691.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65343⟩⟩) (rawTerms := some (Proof.Events483.exact123693RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 123697)
      LeftBound123697.bound (LeftBound123697.actual selector witness) := by
  exact .transfer (LeftBound123697.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound123691.bound LeftBound123697.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123691.bound, LeftBound123697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound123691.actual selector witness) * (LeftBound123697.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123698

namespace LeftBound123706
def owner : Owner := ⟨.program ⟨257⟩, ⟨65345⟩⟩
def transferEvent : Nat := 123706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 123704 .coefficient, .predecessor 1 123705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123704 .coefficient)
      LeftBound123696.bound (LeftBound123696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123705 .coefficient)
      LeftBound123668.bound (LeftBound123668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123696.bound, LeftBound123668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123696.bound, LeftBound123668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123696.actual selector witness, LeftBound123668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123706

namespace LeftBound123708
def owner : Owner := ⟨.program ⟨257⟩, ⟨65345⟩⟩
def transferEvent : Nat := 123708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 123703 .summary, .result 123673 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123703 .summary)
      LeftBound123698.bound (LeftBound123698.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65344⟩⟩) (rawTerms := some (Proof.Events483.exact123703RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123673 .summary)
      LeftBound123670.bound (LeftBound123670.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65340⟩⟩) (rawTerms := some (Proof.Events483.exact123673RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound123670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound123698.bound, LeftBound123670.bound]
def bound : CoeffClass := .finite ⟨279196729344, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123698.bound, LeftBound123670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound123698.actual selector witness, LeftBound123670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound123708

namespace LeftBound123712
def owner : Owner := ⟨.program ⟨257⟩, ⟨69197⟩⟩
def transferEvent : Nat := 123712
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 123710 .coefficient) (.predecessor 1 123711 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 123710 .coefficient)
      LeftBound123706.bound (LeftBound123706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events483.exact123709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound123706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound123706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 123711 .coefficient)
      LeftAuthority123644.bound (LeftAuthority123644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events482.exact123645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound123706.bound LeftAuthority123644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound123706.bound, LeftAuthority123644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound123706.actual selector witness) * (LeftAuthority123644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound123712

namespace LeftBound123713
def owner : Owner := ⟨.program ⟨257⟩, ⟨69197⟩⟩
def transferEvent : Nat := 123713
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨69196⟩⟩]⟩ [⟨.result 123645 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 123645 .coefficient)
      LeftAuthority123644.bound (LeftAuthority123644.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨69196⟩⟩) (rawTerms := some (Proof.Events482.exact123645RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority123644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority123644.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority123644.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority123644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority123644.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound123713

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
