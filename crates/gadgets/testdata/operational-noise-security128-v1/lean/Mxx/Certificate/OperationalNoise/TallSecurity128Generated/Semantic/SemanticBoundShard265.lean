import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard264

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound45534
def owner : Owner := ⟨.program ⟨257⟩, ⟨22855⟩⟩
def transferEvent : Nat := 45534
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 45533) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 45533)
      LeftBound45533.bound (LeftBound45533.actual selector witness) := by
  exact .transfer (LeftBound45533.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound45533.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound45533.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound45533.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45534

namespace LeftBound45629
def owner : Owner := ⟨.program ⟨257⟩, ⟨21881⟩⟩
def transferEvent : Nat := 45629
def frameStart : Nat := 45590
def rule : BoundRule := .identity (.predecessor 0 45628 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45628 .coefficient)
      LeftAuthority45626.bound (LeftAuthority45626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45626.derived selector witness)

def rawBound : CoeffClass := LeftAuthority45626.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority45626.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound45629

namespace LeftBound45646
def owner : Owner := ⟨.program ⟨257⟩, ⟨23322⟩⟩
def transferEvent : Nat := 45646
def frameStart : Nat := 45590
def rule : BoundRule := .sum [.predecessor 0 45644 .coefficient, .predecessor 1 45645 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45644 .coefficient)
      LeftBound45629.bound (LeftBound45629.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound45629.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45645 .coefficient)
      LeftAuthority45642.bound (LeftAuthority45642.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority45642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45629.bound, LeftAuthority45642.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45629.bound, LeftAuthority45642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound45629.actual selector witness, LeftAuthority45642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45646

namespace LeftBound45649
def owner : Owner := ⟨.program ⟨257⟩, ⟨23323⟩⟩
def transferEvent : Nat := 45649
def frameStart : Nat := 45590
def rule : BoundRule := .identity (.predecessor 0 45648 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45648 .coefficient)
      LeftBound45646.bound (LeftBound45646.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound45646.derived selector witness)

def rawBound : CoeffClass := LeftBound45646.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound45646.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound45649

namespace LeftBound45655
def owner : Owner := ⟨.program ⟨257⟩, ⟨23324⟩⟩
def transferEvent : Nat := 45655
def frameStart : Nat := 45590
def rule : BoundRule := .product (.predecessor 0 45653 .coefficient) (.predecessor 1 45654 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45653 .coefficient)
      LeftAuthority45651.bound (LeftAuthority45651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45654 .coefficient)
      LeftBound45649.bound (LeftBound45649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45649.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority45651.bound LeftBound45649.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45651.bound, LeftBound45649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority45651.actual selector witness) * (LeftBound45649.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45655

namespace LeftBound45663
def owner : Owner := ⟨.program ⟨257⟩, ⟨23325⟩⟩
def transferEvent : Nat := 45663
def frameStart : Nat := 45590
def rule : BoundRule := .sum [.predecessor 0 45661 .coefficient, .predecessor 1 45662 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45661 .coefficient)
      LeftAuthority45659.bound (LeftAuthority45659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45662 .coefficient)
      LeftBound45655.bound (LeftBound45655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45655.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority45659.bound, LeftBound45655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45659.bound, LeftBound45655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority45659.actual selector witness, LeftBound45655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45663

namespace LeftBound45667
def owner : Owner := ⟨.program ⟨257⟩, ⟨24145⟩⟩
def transferEvent : Nat := 45667
def frameStart : Nat := 45590
def rule : BoundRule := .product (.predecessor 0 45665 .coefficient) (.predecessor 1 45666 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45665 .coefficient)
      LeftBound45663.bound (LeftBound45663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45666 .coefficient)
      LeftAuthority45640.bound (LeftAuthority45640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45663.bound LeftAuthority45640.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45663.bound, LeftAuthority45640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45663.actual selector witness) * (LeftAuthority45640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45667

namespace LeftBound45678
def owner : Owner := ⟨.program ⟨257⟩, ⟨22255⟩⟩
def transferEvent : Nat := 45678
def frameStart : Nat := 45590
def rule : BoundRule := .product (.predecessor 0 45676 .coefficient) (.predecessor 1 45677 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45676 .coefficient)
      LeftAuthority45651.bound (LeftAuthority45651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45677 .coefficient)
      LeftAuthority45674.bound (LeftAuthority45674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority45651.bound LeftAuthority45674.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45651.bound, LeftAuthority45674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority45651.actual selector witness) * (LeftAuthority45674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45678

namespace LeftBound45686
def owner : Owner := ⟨.program ⟨257⟩, ⟨22256⟩⟩
def transferEvent : Nat := 45686
def frameStart : Nat := 45590
def rule : BoundRule := .sum [.predecessor 0 45684 .coefficient, .predecessor 1 45685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45684 .coefficient)
      LeftAuthority45682.bound (LeftAuthority45682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45685 .coefficient)
      LeftBound45678.bound (LeftBound45678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority45682.bound, LeftBound45678.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority45682.bound, LeftBound45678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority45682.actual selector witness, LeftBound45678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45686

namespace LeftBound45690
def owner : Owner := ⟨.program ⟨257⟩, ⟨24150⟩⟩
def transferEvent : Nat := 45690
def frameStart : Nat := 45590
def rule : BoundRule := .sum [.predecessor 0 45688 .coefficient, .predecessor 1 45689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45688 .coefficient)
      LeftBound45686.bound (LeftBound45686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45689 .coefficient)
      LeftBound45667.bound (LeftBound45667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45686.bound, LeftBound45667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45686.bound, LeftBound45667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound45686.actual selector witness, LeftBound45667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45690

namespace LeftBound45703
def owner : Owner := ⟨.program ⟨257⟩, ⟨24147⟩⟩
def transferEvent : Nat := 45703
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 45701 .coefficient, .predecessor 1 45702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45701 .coefficient)
      LeftBound45532.bound (LeftBound45532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45702 .coefficient)
      LeftBound45515.bound (LeftBound45515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events177.exact45522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45532.bound, LeftBound45515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45532.bound, LeftBound45515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound45532.actual selector witness, LeftBound45515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45703

namespace LeftBound45706
def owner : Owner := ⟨.program ⟨257⟩, ⟨24147⟩⟩
def transferEvent : Nat := 45706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 45700 .summary, .result 45522 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45700 .summary)
      LeftBound45534.bound (LeftBound45534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22855⟩⟩) (rawTerms := some (Proof.Events178.exact45700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45522 .summary)
      LeftBound45517.bound (LeftBound45517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24146⟩⟩) (rawTerms := some (Proof.Events177.exact45522RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound45534.bound, LeftBound45517.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45534.bound, LeftBound45517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound45534.actual selector witness, LeftBound45517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound45706

namespace LeftBound45710
def owner : Owner := ⟨.program ⟨257⟩, ⟨24148⟩⟩
def transferEvent : Nat := 45710
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 45708 .coefficient) (.predecessor 1 45709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45708 .coefficient)
      LeftBound45703.bound (LeftBound45703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound45703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound45703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45709 .coefficient)
      LeftBound15841.bound (LeftBound15841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45703.bound LeftBound15841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45703.bound, LeftBound15841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45703.actual selector witness) * (LeftBound15841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45710

namespace LeftBound45711
def owner : Owner := ⟨.program ⟨257⟩, ⟨24148⟩⟩
def transferEvent : Nat := 45711
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩ [⟨.result 15838 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15838 .coefficient)
      LeftAuthority15837.bound (LeftAuthority15837.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7155⟩⟩) (rawTerms := some (Proof.Events061.exact15838RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15837.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15837.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15837.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound45711

namespace LeftBound45712
def owner : Owner := ⟨.program ⟨257⟩, ⟨24148⟩⟩
def transferEvent : Nat := 45712
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 45707 .summary) (.transfer 45711) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 45707 .summary)
      LeftBound45706.bound (LeftBound45706.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24147⟩⟩) (rawTerms := some (Proof.Events178.exact45707RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound45706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 45711)
      LeftBound45711.bound (LeftBound45711.actual selector witness) := by
  exact .transfer (LeftBound45711.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound45706.bound LeftBound45711.bound
def bound : CoeffClass := .finite ⟨345626795057764889831969145180473178193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound45706.bound, LeftBound45711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound45706.actual selector witness) * (LeftBound45711.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45712

namespace LeftBound45727
def owner : Owner := ⟨.program ⟨257⟩, ⟨20926⟩⟩
def transferEvent : Nat := 45727
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 45725 .coefficient) (.predecessor 1 45726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 45725 .coefficient)
      LeftBound40014.bound (LeftBound40014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events156.exact40018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 45726 .coefficient)
      LeftAuthority45723.bound (LeftAuthority45723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events178.exact45724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority45723.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority45723.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound40014.bound LeftAuthority45723.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40014.bound, LeftAuthority45723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound40014.actual selector witness) * (LeftAuthority45723.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound45727

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
