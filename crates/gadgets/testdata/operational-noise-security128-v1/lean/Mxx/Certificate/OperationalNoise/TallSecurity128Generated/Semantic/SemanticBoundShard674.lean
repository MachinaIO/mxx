import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard579
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard672
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard673

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104627
def owner : Owner := ⟨.program ⟨257⟩, ⟨17897⟩⟩
def transferEvent : Nat := 104627
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104625 .coefficient, .predecessor 1 104626 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104625 .coefficient)
      LeftBound104456.bound (LeftBound104456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104456.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104456.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104626 .coefficient)
      LeftBound104439.bound (LeftBound104439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104439.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104456.bound, LeftBound104439.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104456.bound, LeftBound104439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104456.actual selector witness, LeftBound104439.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104627

namespace LeftBound104630
def owner : Owner := ⟨.program ⟨257⟩, ⟨17897⟩⟩
def transferEvent : Nat := 104630
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104624 .summary, .result 104446 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104624 .summary)
      LeftBound104458.bound (LeftBound104458.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16695⟩⟩) (rawTerms := some (Proof.Events408.exact104624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104446 .summary)
      LeftBound104441.bound (LeftBound104441.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17896⟩⟩) (rawTerms := some (Proof.Events407.exact104446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104441.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104458.bound, LeftBound104441.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104458.bound, LeftBound104441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104458.actual selector witness, LeftBound104441.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104630

namespace LeftBound104634
def owner : Owner := ⟨.program ⟨257⟩, ⟨17898⟩⟩
def transferEvent : Nat := 104634
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104632 .coefficient) (.predecessor 1 104633 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104632 .coefficient)
      LeftBound104627.bound (LeftBound104627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104633 .coefficient)
      LeftBound15881.bound (LeftBound15881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound104627.bound LeftBound15881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104627.bound, LeftBound15881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound104627.actual selector witness) * (LeftBound15881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104634

namespace LeftBound104635
def owner : Owner := ⟨.program ⟨257⟩, ⟨17898⟩⟩
def transferEvent : Nat := 104635
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩ [⟨.result 15878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15878 .coefficient)
      LeftAuthority15877.bound (LeftAuthority15877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7171⟩⟩) (rawTerms := some (Proof.Events062.exact15878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15877.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104635

namespace LeftBound104636
def owner : Owner := ⟨.program ⟨257⟩, ⟨17898⟩⟩
def transferEvent : Nat := 104636
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104631 .summary) (.transfer 104635) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104631 .summary)
      LeftBound104630.bound (LeftBound104630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17897⟩⟩) (rawTerms := some (Proof.Events408.exact104631RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 104635)
      LeftBound104635.bound (LeftBound104635.actual selector witness) := by
  exact .transfer (LeftBound104635.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound104630.bound LeftBound104635.bound
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104630.bound, LeftBound104635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound104630.actual selector witness) * (LeftBound104635.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104636

namespace LeftBound104644
def owner : Owner := ⟨.program ⟨257⟩, ⟨9948⟩⟩
def transferEvent : Nat := 104644
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 104642 .coefficient) (.predecessor 1 104643 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104642 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104643 .coefficient)
      LeftBound90526.bound (LeftBound90526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound90526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound90526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound90526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104644

namespace LeftBound104649
def owner : Owner := ⟨.program ⟨257⟩, ⟨9926⟩⟩
def transferEvent : Nat := 104649
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104647 .coefficient) (.predecessor 1 104648 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104647 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104648 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftBound15895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftBound15895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104649

namespace LeftBound104654
def owner : Owner := ⟨.program ⟨257⟩, ⟨9949⟩⟩
def transferEvent : Nat := 104654
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104652 .coefficient, .predecessor 1 104653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104652 .coefficient)
      LeftBound104649.bound (LeftBound104649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104653 .coefficient)
      LeftBound104644.bound (LeftBound104644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104644.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104649.bound, LeftBound104644.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104649.bound, LeftBound104644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104649.actual selector witness, LeftBound104644.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104654

namespace LeftBound104658
def owner : Owner := ⟨.program ⟨257⟩, ⟨9950⟩⟩
def transferEvent : Nat := 104658
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104656 .coefficient, .predecessor 1 104657 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104656 .coefficient)
      LeftBound104654.bound (LeftBound104654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104657 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104654.bound, LeftBound31515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104654.bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104654.actual selector witness, LeftBound31515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104658

namespace LeftBound104659
def owner : Owner := ⟨.program ⟨257⟩, ⟨9950⟩⟩
def transferEvent : Nat := 104659
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨118⟩⟩]⟩ [⟨.result 31516 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31516 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨118⟩⟩) (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound31515.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound31515.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104659

namespace LeftBound104664
def owner : Owner := ⟨.program ⟨257⟩, ⟨9951⟩⟩
def transferEvent : Nat := 104664
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104662 .coefficient, .predecessor 1 104663 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104662 .coefficient)
      LeftBound104658.bound (LeftBound104658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104663 .coefficient)
      LeftBound104658.bound (LeftBound104658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104658.bound, LeftBound104658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104658.bound, LeftBound104658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104658.actual selector witness, LeftBound104658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104664

namespace LeftBound104667
def owner : Owner := ⟨.program ⟨257⟩, ⟨9951⟩⟩
def transferEvent : Nat := 104667
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104661 .summary, .result 104661 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104661 .summary)
      LeftBound104659.bound (LeftBound104659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9950⟩⟩) (rawTerms := some (Proof.Events408.exact104661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104661 .summary)
      LeftBound104659.bound (LeftBound104659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9950⟩⟩) (rawTerms := some (Proof.Events408.exact104661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104659.bound, LeftBound104659.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104659.bound, LeftBound104659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104659.actual selector witness, LeftBound104659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104667

namespace LeftBound104671
def owner : Owner := ⟨.program ⟨257⟩, ⟨17899⟩⟩
def transferEvent : Nat := 104671
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104669 .coefficient, .predecessor 1 104670 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104669 .coefficient)
      LeftBound104664.bound (LeftBound104664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104670 .coefficient)
      LeftBound104634.bound (LeftBound104634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104634.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104664.bound, LeftBound104634.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104664.bound, LeftBound104634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104664.actual selector witness, LeftBound104634.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104671

namespace LeftBound104672
def owner : Owner := ⟨.program ⟨257⟩, ⟨17899⟩⟩
def transferEvent : Nat := 104672
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104668 .summary, .result 104641 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104668 .summary)
      LeftBound104667.bound (LeftBound104667.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9951⟩⟩) (rawTerms := some (Proof.Events408.exact104668RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104641 .summary)
      LeftBound104636.bound (LeftBound104636.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17898⟩⟩) (rawTerms := some (Proof.Events408.exact104641RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104636.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104667.bound, LeftBound104636.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104667.bound, LeftBound104636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104667.actual selector witness, LeftBound104636.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104672

namespace LeftBound104676
def owner : Owner := ⟨.program ⟨257⟩, ⟨20805⟩⟩
def transferEvent : Nat := 104676
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104674 .coefficient, .predecessor 1 104675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104674 .coefficient)
      LeftBound104671.bound (LeftBound104671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104675 .coefficient)
      LeftBound104422.bound (LeftBound104422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104422.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104671.bound, LeftBound104422.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104671.bound, LeftBound104422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104671.actual selector witness, LeftBound104422.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104676

namespace LeftBound104677
def owner : Owner := ⟨.program ⟨257⟩, ⟨20805⟩⟩
def transferEvent : Nat := 104677
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104673 .summary, .result 104429 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104673 .summary)
      LeftBound104672.bound (LeftBound104672.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17899⟩⟩) (rawTerms := some (Proof.Events408.exact104673RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104429 .summary)
      LeftBound104424.bound (LeftBound104424.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20804⟩⟩) (rawTerms := some (Proof.Events407.exact104429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104672.bound, LeftBound104424.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104672.bound, LeftBound104424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104672.actual selector witness, LeftBound104424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104677

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
