import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard054
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard167
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard272
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard275
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard364
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard365
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard366
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard367
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard368
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard369

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound60769
def owner : Owner := ⟨.program ⟨257⟩, ⟨11220⟩⟩
def transferEvent : Nat := 60769
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 60767 .coefficient) (.predecessor 1 60768 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60767 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60768 .coefficient)
      LeftBound46651.bound (LeftBound46651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound46651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound46651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound46651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound60769

namespace LeftBound60774
def owner : Owner := ⟨.program ⟨257⟩, ⟨11198⟩⟩
def transferEvent : Nat := 60774
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 60772 .coefficient) (.predecessor 1 60773 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60772 .coefficient)
      LeftBound46522.bound (LeftBound46522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60773 .coefficient)
      LeftBound15895.bound (LeftBound15895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15895.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound46522.bound LeftBound15895.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46522.bound, LeftBound15895.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound46522.actual selector witness) * (LeftBound15895.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound60774

namespace LeftBound60779
def owner : Owner := ⟨.program ⟨257⟩, ⟨11221⟩⟩
def transferEvent : Nat := 60779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60777 .coefficient, .predecessor 1 60778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60777 .coefficient)
      LeftBound60774.bound (LeftBound60774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60778 .coefficient)
      LeftBound60769.bound (LeftBound60769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60769.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60774.bound, LeftBound60769.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60774.bound, LeftBound60769.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60774.actual selector witness, LeftBound60769.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60779

namespace LeftBound60783
def owner : Owner := ⟨.program ⟨257⟩, ⟨11222⟩⟩
def transferEvent : Nat := 60783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60781 .coefficient, .predecessor 1 60782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60781 .coefficient)
      LeftBound60779.bound (LeftBound60779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60782 .coefficient)
      LeftBound31515.bound (LeftBound31515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60779.bound, LeftBound31515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60779.bound, LeftBound31515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60779.actual selector witness, LeftBound31515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60783

namespace LeftBound60784
def owner : Owner := ⟨.program ⟨257⟩, ⟨11222⟩⟩
def transferEvent : Nat := 60784
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
end LeftBound60784

namespace LeftBound60789
def owner : Owner := ⟨.program ⟨257⟩, ⟨11223⟩⟩
def transferEvent : Nat := 60789
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60787 .coefficient, .predecessor 1 60788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60787 .coefficient)
      LeftBound60783.bound (LeftBound60783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60788 .coefficient)
      LeftBound60783.bound (LeftBound60783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60783.bound, LeftBound60783.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60783.bound, LeftBound60783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60783.actual selector witness, LeftBound60783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60789

namespace LeftBound60792
def owner : Owner := ⟨.program ⟨257⟩, ⟨11223⟩⟩
def transferEvent : Nat := 60792
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60786 .summary, .result 60786 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60786 .summary)
      LeftBound60784.bound (LeftBound60784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11222⟩⟩) (rawTerms := some (Proof.Events237.exact60786RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60784.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60786 .summary)
      LeftBound60784.bound (LeftBound60784.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11222⟩⟩) (rawTerms := some (Proof.Events237.exact60786RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60784.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60784.bound, LeftBound60784.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60784.bound, LeftBound60784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60784.actual selector witness, LeftBound60784.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60792

namespace LeftBound60796
def owner : Owner := ⟨.program ⟨257⟩, ⟨17983⟩⟩
def transferEvent : Nat := 60796
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60794 .coefficient, .predecessor 1 60795 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60794 .coefficient)
      LeftBound60789.bound (LeftBound60789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60795 .coefficient)
      LeftBound60759.bound (LeftBound60759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60759.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60789.bound, LeftBound60759.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60789.bound, LeftBound60759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60789.actual selector witness, LeftBound60759.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60796

namespace LeftBound60797
def owner : Owner := ⟨.program ⟨257⟩, ⟨17983⟩⟩
def transferEvent : Nat := 60797
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60793 .summary, .result 60766 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60793 .summary)
      LeftBound60792.bound (LeftBound60792.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11223⟩⟩) (rawTerms := some (Proof.Events237.exact60793RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60766 .summary)
      LeftBound60761.bound (LeftBound60761.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17982⟩⟩) (rawTerms := some (Proof.Events237.exact60766RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60761.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60792.bound, LeftBound60761.bound]
def bound : CoeffClass := .finite ⟨345624685687166110058245054666339432529972, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60792.bound, LeftBound60761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60792.actual selector witness, LeftBound60761.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60797

namespace LeftBound60801
def owner : Owner := ⟨.program ⟨257⟩, ⟨20898⟩⟩
def transferEvent : Nat := 60801
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60799 .coefficient, .predecessor 1 60800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60799 .coefficient)
      LeftBound60796.bound (LeftBound60796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60800 .coefficient)
      LeftBound60547.bound (LeftBound60547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events236.exact60554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60796.bound, LeftBound60547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60796.bound, LeftBound60547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60796.actual selector witness, LeftBound60547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60801

namespace LeftBound60802
def owner : Owner := ⟨.program ⟨257⟩, ⟨20898⟩⟩
def transferEvent : Nat := 60802
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60798 .summary, .result 60554 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60798 .summary)
      LeftBound60797.bound (LeftBound60797.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17983⟩⟩) (rawTerms := some (Proof.Events237.exact60798RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60554 .summary)
      LeftBound60549.bound (LeftBound60549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20897⟩⟩) (rawTerms := some (Proof.Events236.exact60554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60797.bound, LeftBound60549.bound]
def bound : CoeffClass := .finite ⟨691250426059631610003352154589745737891892, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60797.bound, LeftBound60549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60797.actual selector witness, LeftBound60549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60802

namespace LeftBound60806
def owner : Owner := ⟨.program ⟨257⟩, ⟨24118⟩⟩
def transferEvent : Nat := 60806
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60804 .coefficient, .predecessor 1 60805 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60804 .coefficient)
      LeftBound60801.bound (LeftBound60801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60805 .coefficient)
      LeftBound60335.bound (LeftBound60335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events235.exact60342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60801.bound, LeftBound60335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60801.bound, LeftBound60335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60801.actual selector witness, LeftBound60335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60806

namespace LeftBound60807
def owner : Owner := ⟨.program ⟨257⟩, ⟨24118⟩⟩
def transferEvent : Nat := 60807
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60803 .summary, .result 60342 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60803 .summary)
      LeftBound60802.bound (LeftBound60802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20898⟩⟩) (rawTerms := some (Proof.Events237.exact60803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60342 .summary)
      LeftBound60337.bound (LeftBound60337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24117⟩⟩) (rawTerms := some (Proof.Events235.exact60342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60802.bound, LeftBound60337.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60802.bound, LeftBound60337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60802.actual selector witness, LeftBound60337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60807

namespace LeftBound60811
def owner : Owner := ⟨.program ⟨257⟩, ⟨34138⟩⟩
def transferEvent : Nat := 60811
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60809 .coefficient, .predecessor 1 60810 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60809 .coefficient)
      LeftBound60806.bound (LeftBound60806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60810 .coefficient)
      LeftBound60123.bound (LeftBound60123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact60130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60806.bound, LeftBound60123.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60806.bound, LeftBound60123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60806.actual selector witness, LeftBound60123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60811

namespace LeftBound60812
def owner : Owner := ⟨.program ⟨257⟩, ⟨34138⟩⟩
def transferEvent : Nat := 60812
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 60808 .summary, .result 60130 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60808 .summary)
      LeftBound60807.bound (LeftBound60807.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24118⟩⟩) (rawTerms := some (Proof.Events237.exact60808RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60807.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 60130 .summary)
      LeftBound60125.bound (LeftBound60125.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34137⟩⟩) (rawTerms := some (Proof.Events234.exact60130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60125.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60807.bound, LeftBound60125.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60807.bound, LeftBound60125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60807.actual selector witness, LeftBound60125.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60812

namespace LeftBound60816
def owner : Owner := ⟨.program ⟨257⟩, ⟨53198⟩⟩
def transferEvent : Nat := 60816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 60814 .coefficient, .predecessor 1 60815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 60814 .coefficient)
      LeftBound60811.bound (LeftBound60811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events237.exact60813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound60811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound60811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 60815 .coefficient)
      LeftBound59911.bound (LeftBound59911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events234.exact59918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59911.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound60811.bound, LeftBound59911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound60811.bound, LeftBound59911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound60811.actual selector witness, LeftBound59911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound60816

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
