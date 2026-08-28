import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1661

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound245620
def owner : Owner := ⟨.program ⟨257⟩, ⟨68353⟩⟩
def transferEvent : Nat := 245620
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 245619) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 245619)
      LeftBound245619.bound (LeftBound245619.actual selector witness) := by
  exact .transfer (LeftBound245619.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound245619.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound245619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound245619.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound245620

namespace LeftBound246648
def owner : Owner := ⟨.program ⟨257⟩, ⟨18829⟩⟩
def transferEvent : Nat := 246648
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246646 .coefficient, .predecessor 1 246647 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246646 .coefficient)
      LeftAuthority246644.bound (LeftAuthority246644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246647 .coefficient)
      LeftAuthority246621.bound (LeftAuthority246621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246621.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246621.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority246644.bound, LeftAuthority246621.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority246644.bound, LeftAuthority246621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority246644.actual selector witness, LeftAuthority246621.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246648

namespace LeftBound246652
def owner : Owner := ⟨.program ⟨257⟩, ⟨22049⟩⟩
def transferEvent : Nat := 246652
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246650 .coefficient, .predecessor 1 246651 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246650 .coefficient)
      LeftBound246648.bound (LeftBound246648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246651 .coefficient)
      LeftAuthority246598.bound (LeftAuthority246598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246598.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246648.bound, LeftAuthority246598.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246648.bound, LeftAuthority246598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246648.actual selector witness, LeftAuthority246598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246652

namespace LeftBound246656
def owner : Owner := ⟨.program ⟨257⟩, ⟨32069⟩⟩
def transferEvent : Nat := 246656
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246654 .coefficient, .predecessor 1 246655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246654 .coefficient)
      LeftBound246652.bound (LeftBound246652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246655 .coefficient)
      LeftAuthority246575.bound (LeftAuthority246575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246575.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246652.bound, LeftAuthority246575.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246652.bound, LeftAuthority246575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246652.actual selector witness, LeftAuthority246575.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246656

namespace LeftBound246660
def owner : Owner := ⟨.program ⟨257⟩, ⟨51124⟩⟩
def transferEvent : Nat := 246660
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246658 .coefficient, .predecessor 1 246659 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246658 .coefficient)
      LeftBound246656.bound (LeftBound246656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246659 .coefficient)
      LeftAuthority246552.bound (LeftAuthority246552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246656.bound, LeftAuthority246552.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246656.bound, LeftAuthority246552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246656.actual selector witness, LeftAuthority246552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246660

namespace LeftBound246664
def owner : Owner := ⟨.program ⟨257⟩, ⟨54104⟩⟩
def transferEvent : Nat := 246664
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246662 .coefficient, .predecessor 1 246663 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246662 .coefficient)
      LeftBound246660.bound (LeftBound246660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246663 .coefficient)
      LeftAuthority246529.bound (LeftAuthority246529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246529.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246660.bound, LeftAuthority246529.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246660.bound, LeftAuthority246529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246660.actual selector witness, LeftAuthority246529.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246664

namespace LeftBound246668
def owner : Owner := ⟨.program ⟨257⟩, ⟨57084⟩⟩
def transferEvent : Nat := 246668
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246666 .coefficient, .predecessor 1 246667 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246666 .coefficient)
      LeftBound246664.bound (LeftBound246664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246667 .coefficient)
      LeftAuthority246506.bound (LeftAuthority246506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246506.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246664.bound, LeftAuthority246506.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246664.bound, LeftAuthority246506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246664.actual selector witness, LeftAuthority246506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246668

namespace LeftBound246672
def owner : Owner := ⟨.program ⟨257⟩, ⟨60064⟩⟩
def transferEvent : Nat := 246672
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246670 .coefficient, .predecessor 1 246671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246670 .coefficient)
      LeftBound246668.bound (LeftBound246668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246671 .coefficient)
      LeftAuthority246483.bound (LeftAuthority246483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246483.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246668.bound, LeftAuthority246483.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246668.bound, LeftAuthority246483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246668.actual selector witness, LeftAuthority246483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246672

namespace LeftBound246676
def owner : Owner := ⟨.program ⟨257⟩, ⟨63044⟩⟩
def transferEvent : Nat := 246676
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246674 .coefficient, .predecessor 1 246675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246674 .coefficient)
      LeftBound246672.bound (LeftBound246672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246675 .coefficient)
      LeftAuthority246460.bound (LeftAuthority246460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246672.bound, LeftAuthority246460.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246672.bound, LeftAuthority246460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246672.actual selector witness, LeftAuthority246460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246676

namespace LeftBound246680
def owner : Owner := ⟨.program ⟨257⟩, ⟨66462⟩⟩
def transferEvent : Nat := 246680
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246678 .coefficient, .predecessor 1 246679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246678 .coefficient)
      LeftBound246676.bound (LeftBound246676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246679 .coefficient)
      LeftAuthority246437.bound (LeftAuthority246437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246676.bound, LeftAuthority246437.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246676.bound, LeftAuthority246437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246676.actual selector witness, LeftAuthority246437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246680

namespace LeftBound246684
def owner : Owner := ⟨.program ⟨257⟩, ⟨66463⟩⟩
def transferEvent : Nat := 246684
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246682 .coefficient, .predecessor 1 246683 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246682 .coefficient)
      LeftBound246680.bound (LeftBound246680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246683 .coefficient)
      LeftAuthority246414.bound (LeftAuthority246414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246680.bound, LeftAuthority246414.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246680.bound, LeftAuthority246414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246680.actual selector witness, LeftAuthority246414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246684

namespace LeftBound246688
def owner : Owner := ⟨.program ⟨257⟩, ⟨66464⟩⟩
def transferEvent : Nat := 246688
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246686 .coefficient, .predecessor 1 246687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246686 .coefficient)
      LeftBound246684.bound (LeftBound246684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246687 .coefficient)
      LeftAuthority246391.bound (LeftAuthority246391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246684.bound, LeftAuthority246391.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246684.bound, LeftAuthority246391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246684.actual selector witness, LeftAuthority246391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246688

namespace LeftBound246692
def owner : Owner := ⟨.program ⟨257⟩, ⟨66465⟩⟩
def transferEvent : Nat := 246692
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246690 .coefficient, .predecessor 1 246691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246690 .coefficient)
      LeftBound246688.bound (LeftBound246688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246691 .coefficient)
      LeftAuthority246368.bound (LeftAuthority246368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246368.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246688.bound, LeftAuthority246368.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246688.bound, LeftAuthority246368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246688.actual selector witness, LeftAuthority246368.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246692

namespace LeftBound246696
def owner : Owner := ⟨.program ⟨257⟩, ⟨66466⟩⟩
def transferEvent : Nat := 246696
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246694 .coefficient, .predecessor 1 246695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246694 .coefficient)
      LeftBound246692.bound (LeftBound246692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246695 .coefficient)
      LeftAuthority246345.bound (LeftAuthority246345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246692.bound, LeftAuthority246345.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246692.bound, LeftAuthority246345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246692.actual selector witness, LeftAuthority246345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246696

namespace LeftBound246700
def owner : Owner := ⟨.program ⟨257⟩, ⟨66467⟩⟩
def transferEvent : Nat := 246700
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246698 .coefficient, .predecessor 1 246699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246698 .coefficient)
      LeftBound246696.bound (LeftBound246696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246699 .coefficient)
      LeftAuthority246322.bound (LeftAuthority246322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246696.bound, LeftAuthority246322.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246696.bound, LeftAuthority246322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246696.actual selector witness, LeftAuthority246322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246700

namespace LeftBound246704
def owner : Owner := ⟨.program ⟨257⟩, ⟨66468⟩⟩
def transferEvent : Nat := 246704
def frameStart : Nat := 246211
def rule : BoundRule := .sum [.predecessor 0 246702 .coefficient, .predecessor 1 246703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 246702 .coefficient)
      LeftBound246700.bound (LeftBound246700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events963.exact246701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound246700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound246700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 246703 .coefficient)
      LeftAuthority246299.bound (LeftAuthority246299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events962.exact246300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority246299.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority246299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound246700.bound, LeftAuthority246299.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound246700.bound, LeftAuthority246299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound246700.actual selector witness, LeftAuthority246299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound246704

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
