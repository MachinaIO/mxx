import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard133
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound215736
def owner : Owner := ⟨.program ⟨257⟩, ⟨15477⟩⟩
def transferEvent : Nat := 215736
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 215734 .coefficient) (.predecessor 1 215735 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215734 .coefficient)
      LeftAuthority10208.bound (LeftAuthority10208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215735 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10208.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10208.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10208.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound215736

namespace LeftBound215741
def owner : Owner := ⟨.program ⟨257⟩, ⟨8610⟩⟩
def transferEvent : Nat := 215741
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 215739 .coefficient) (.predecessor 1 215740 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215739 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215740 .coefficient)
      LeftBound25596.bound (LeftBound25596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound25596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound25596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound25596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215741

namespace LeftBound215746
def owner : Owner := ⟨.program ⟨257⟩, ⟨15478⟩⟩
def transferEvent : Nat := 215746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 215744 .coefficient, .predecessor 1 215745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215744 .coefficient)
      LeftBound215741.bound (LeftBound215741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215745 .coefficient)
      LeftBound215736.bound (LeftBound215736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215736.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound215741.bound, LeftBound215736.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215741.bound, LeftBound215736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound215741.actual selector witness, LeftBound215736.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound215746

namespace LeftBound215750
def owner : Owner := ⟨.program ⟨257⟩, ⟨15479⟩⟩
def transferEvent : Nat := 215750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 215748 .coefficient, .predecessor 1 215749 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215748 .coefficient)
      LeftBound215746.bound (LeftBound215746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215747RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215749 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound215746.bound, LeftBound25588.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215746.bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound215746.actual selector witness, LeftBound25588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound215750

namespace LeftBound215751
def owner : Owner := ⟨.program ⟨257⟩, ⟨15479⟩⟩
def transferEvent : Nat := 215751
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨130⟩⟩]⟩ [⟨.result 25589 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25589 .coefficient)
      LeftBound25588.bound (LeftBound25588.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨130⟩⟩) (rawTerms := some (Proof.Events099.exact25589RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25588.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25588.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25588.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound215751

namespace LeftBound215756
def owner : Owner := ⟨.program ⟨257⟩, ⟨15480⟩⟩
def transferEvent : Nat := 215756
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 215754 .coefficient) (.predecessor 1 215755 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215754 .coefficient)
      LeftBound215750.bound (LeftBound215750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215755 .coefficient)
      LeftAuthority10211.bound (LeftAuthority10211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound215750.bound LeftAuthority10211.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215750.bound, LeftAuthority10211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound215750.actual selector witness) * (LeftAuthority10211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215756

namespace LeftBound215757
def owner : Owner := ⟨.program ⟨257⟩, ⟨15480⟩⟩
def transferEvent : Nat := 215757
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12381⟩⟩], []⟩ [⟨.result 10212 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 10212 .coefficient)
      LeftAuthority10211.bound (LeftAuthority10211.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12381⟩⟩) (rawTerms := some (Proof.Events039.exact10212RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10211.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10211.bound []
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority10211.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound215757

namespace LeftBound215758
def owner : Owner := ⟨.program ⟨257⟩, ⟨15480⟩⟩
def transferEvent : Nat := 215758
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 215753 .summary) (.transfer 215757) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 215753 .summary)
      LeftBound215751.bound (LeftBound215751.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨15479⟩⟩) (rawTerms := some (Proof.Events842.exact215753RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound215751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 215757)
      LeftBound215757.bound (LeftBound215757.actual selector witness) := by
  exact .transfer (LeftBound215757.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound215751.bound LeftBound215757.bound
def bound : CoeffClass := .finite ⟨1703936, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215751.bound, LeftBound215757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound215751.actual selector witness) * (LeftBound215757.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215758

namespace LeftBound215764
def owner : Owner := ⟨.program ⟨257⟩, ⟨12382⟩⟩
def transferEvent : Nat := 215764
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 215762 .coefficient) (.predecessor 1 215763 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215762 .coefficient)
      LeftAuthority10211.bound (LeftAuthority10211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215763 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10211.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10211.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10211.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound215764

namespace LeftBound215769
def owner : Owner := ⟨.program ⟨257⟩, ⟨8609⟩⟩
def transferEvent : Nat := 215769
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 215767 .coefficient) (.predecessor 1 215768 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215767 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215768 .coefficient)
      LeftBound25637.bound (LeftBound25637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25637.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25637.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound25637.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound25637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound25637.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215769

namespace LeftBound215774
def owner : Owner := ⟨.program ⟨257⟩, ⟨12383⟩⟩
def transferEvent : Nat := 215774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 215772 .coefficient, .predecessor 1 215773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215772 .coefficient)
      LeftBound215769.bound (LeftBound215769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215769.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215773 .coefficient)
      LeftBound215764.bound (LeftBound215764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound215769.bound, LeftBound215764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215769.bound, LeftBound215764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound215769.actual selector witness, LeftBound215764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound215774

namespace LeftBound215778
def owner : Owner := ⟨.program ⟨257⟩, ⟨12384⟩⟩
def transferEvent : Nat := 215778
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 215776 .coefficient, .predecessor 1 215777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215776 .coefficient)
      LeftBound215774.bound (LeftBound215774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215777 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound215774.bound, LeftBound25629.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215774.bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound215774.actual selector witness, LeftBound25629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound215778

namespace LeftBound215779
def owner : Owner := ⟨.program ⟨257⟩, ⟨12384⟩⟩
def transferEvent : Nat := 215779
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨129⟩⟩]⟩ [⟨.result 25630 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25630 .coefficient)
      LeftBound25629.bound (LeftBound25629.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨129⟩⟩) (rawTerms := some (Proof.Events100.exact25630RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25629.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25629.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25629.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25629.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound215779

namespace LeftBound215784
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def transferEvent : Nat := 215784
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 215782 .coefficient) (.predecessor 1 215783 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 215782 .coefficient)
      LeftBound215778.bound (LeftBound215778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215778.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 215783 .coefficient)
      LeftBound25626.bound (LeftBound25626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25626.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound215778.bound LeftBound25626.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215778.bound, LeftBound25626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound215778.actual selector witness) * (LeftBound25626.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215784

namespace LeftBound215785
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def transferEvent : Nat := 215785
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9568⟩⟩]⟩ [⟨.result 25623 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25623 .coefficient)
      LeftAuthority25622.bound (LeftAuthority25622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9568⟩⟩) (rawTerms := some (Proof.Events100.exact25623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25622.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25622.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority25622.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound215785

namespace LeftBound215786
def owner : Owner := ⟨.program ⟨257⟩, ⟨12385⟩⟩
def transferEvent : Nat := 215786
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 215781 .summary) (.transfer 215785) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 215781 .summary)
      LeftBound215779.bound (LeftBound215779.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨12384⟩⟩) (rawTerms := some (Proof.Events842.exact215781RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound215779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 215785)
      LeftBound215785.bound (LeftBound215785.actual selector witness) := by
  exact .transfer (LeftBound215785.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound215779.bound LeftBound215785.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215779.bound, LeftBound215785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound215779.actual selector witness) * (LeftBound215785.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound215786

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
