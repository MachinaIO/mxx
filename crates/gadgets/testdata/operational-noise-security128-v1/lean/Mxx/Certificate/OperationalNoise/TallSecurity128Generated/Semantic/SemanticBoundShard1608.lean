import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1591
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1594
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1607

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound238524
def owner : Owner := ⟨.program ⟨257⟩, ⟨40819⟩⟩
def transferEvent : Nat := 238524
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 238523) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 238523)
      LeftBound238523.bound (LeftBound238523.actual selector witness) := by
  exact .transfer (LeftBound238523.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound238523.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound238523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound238523.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238524

namespace LeftBound238619
def owner : Owner := ⟨.program ⟨257⟩, ⟨40093⟩⟩
def transferEvent : Nat := 238619
def frameStart : Nat := 238580
def rule : BoundRule := .identity (.predecessor 0 238618 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238618 .coefficient)
      LeftAuthority238616.bound (LeftAuthority238616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238616.derived selector witness)

def rawBound : CoeffClass := LeftAuthority238616.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority238616.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound238619

namespace LeftBound238636
def owner : Owner := ⟨.program ⟨257⟩, ⟨41458⟩⟩
def transferEvent : Nat := 238636
def frameStart : Nat := 238580
def rule : BoundRule := .sum [.predecessor 0 238634 .coefficient, .predecessor 1 238635 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238634 .coefficient)
      LeftBound238619.bound (LeftBound238619.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound238619.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238635 .coefficient)
      LeftAuthority238632.bound (LeftAuthority238632.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority238632.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238619.bound, LeftAuthority238632.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238619.bound, LeftAuthority238632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238619.actual selector witness, LeftAuthority238632.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238636

namespace LeftBound238639
def owner : Owner := ⟨.program ⟨257⟩, ⟨41459⟩⟩
def transferEvent : Nat := 238639
def frameStart : Nat := 238580
def rule : BoundRule := .identity (.predecessor 0 238638 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238638 .coefficient)
      LeftBound238636.bound (LeftBound238636.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound238636.derived selector witness)

def rawBound : CoeffClass := LeftBound238636.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound238636.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound238639

namespace LeftBound238645
def owner : Owner := ⟨.program ⟨257⟩, ⟨41460⟩⟩
def transferEvent : Nat := 238645
def frameStart : Nat := 238580
def rule : BoundRule := .product (.predecessor 0 238643 .coefficient) (.predecessor 1 238644 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238643 .coefficient)
      LeftAuthority238641.bound (LeftAuthority238641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238641.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238644 .coefficient)
      LeftBound238639.bound (LeftBound238639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238639.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority238641.bound LeftBound238639.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238641.bound, LeftBound238639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority238641.actual selector witness) * (LeftBound238639.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238645

namespace LeftBound238653
def owner : Owner := ⟨.program ⟨257⟩, ⟨41461⟩⟩
def transferEvent : Nat := 238653
def frameStart : Nat := 238580
def rule : BoundRule := .sum [.predecessor 0 238651 .coefficient, .predecessor 1 238652 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238651 .coefficient)
      LeftAuthority238649.bound (LeftAuthority238649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238652 .coefficient)
      LeftBound238645.bound (LeftBound238645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238645.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority238649.bound, LeftBound238645.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238649.bound, LeftBound238645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority238649.actual selector witness, LeftBound238645.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238653

namespace LeftBound238657
def owner : Owner := ⟨.program ⟨257⟩, ⟨41940⟩⟩
def transferEvent : Nat := 238657
def frameStart : Nat := 238580
def rule : BoundRule := .product (.predecessor 0 238655 .coefficient) (.predecessor 1 238656 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238655 .coefficient)
      LeftBound238653.bound (LeftBound238653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238656 .coefficient)
      LeftAuthority238630.bound (LeftAuthority238630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238630.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound238653.bound LeftAuthority238630.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238653.bound, LeftAuthority238630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound238653.actual selector witness) * (LeftAuthority238630.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238657

namespace LeftBound238668
def owner : Owner := ⟨.program ⟨257⟩, ⟨40294⟩⟩
def transferEvent : Nat := 238668
def frameStart : Nat := 238580
def rule : BoundRule := .product (.predecessor 0 238666 .coefficient) (.predecessor 1 238667 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238666 .coefficient)
      LeftAuthority238641.bound (LeftAuthority238641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238641.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238667 .coefficient)
      LeftAuthority238664.bound (LeftAuthority238664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority238641.bound LeftAuthority238664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238641.bound, LeftAuthority238664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority238641.actual selector witness) * (LeftAuthority238664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238668

namespace LeftBound238676
def owner : Owner := ⟨.program ⟨257⟩, ⟨40295⟩⟩
def transferEvent : Nat := 238676
def frameStart : Nat := 238580
def rule : BoundRule := .sum [.predecessor 0 238674 .coefficient, .predecessor 1 238675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238674 .coefficient)
      LeftAuthority238672.bound (LeftAuthority238672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority238672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority238672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238675 .coefficient)
      LeftBound238668.bound (LeftBound238668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority238672.bound, LeftBound238668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority238672.bound, LeftBound238668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority238672.actual selector witness, LeftBound238668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238676

namespace LeftBound238680
def owner : Owner := ⟨.program ⟨257⟩, ⟨41943⟩⟩
def transferEvent : Nat := 238680
def frameStart : Nat := 238580
def rule : BoundRule := .sum [.predecessor 0 238678 .coefficient, .predecessor 1 238679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238678 .coefficient)
      LeftBound238676.bound (LeftBound238676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238679 .coefficient)
      LeftBound238657.bound (LeftBound238657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238657.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238676.bound, LeftBound238657.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238676.bound, LeftBound238657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238676.actual selector witness, LeftBound238657.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238680

namespace LeftBound238693
def owner : Owner := ⟨.program ⟨257⟩, ⟨41942⟩⟩
def transferEvent : Nat := 238693
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 238691 .coefficient, .predecessor 1 238692 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238691 .coefficient)
      LeftBound238522.bound (LeftBound238522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238692 .coefficient)
      LeftBound238505.bound (LeftBound238505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events931.exact238512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238522.bound, LeftBound238505.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238522.bound, LeftBound238505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238522.actual selector witness, LeftBound238505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238693

namespace LeftBound238696
def owner : Owner := ⟨.program ⟨257⟩, ⟨41942⟩⟩
def transferEvent : Nat := 238696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 238690 .summary, .result 238512 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238690 .summary)
      LeftBound238524.bound (LeftBound238524.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40819⟩⟩) (rawTerms := some (Proof.Events932.exact238690RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 238512 .summary)
      LeftBound238507.bound (LeftBound238507.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41941⟩⟩) (rawTerms := some (Proof.Events931.exact238512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound238507.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238524.bound, LeftBound238507.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238524.bound, LeftBound238507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238524.actual selector witness, LeftBound238507.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238696

namespace LeftBound238720
def owner : Owner := ⟨.program ⟨257⟩, ⟨37069⟩⟩
def transferEvent : Nat := 238720
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 238718 .coefficient) (.predecessor 1 238719 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238718 .coefficient)
      LeftAuthority11405.bound (LeftAuthority11405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238719 .coefficient)
      LeftBound236776.bound (LeftBound236776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236776.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority11405.bound LeftBound236776.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11405.bound, LeftBound236776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority11405.actual selector witness) * (LeftBound236776.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound238720

namespace LeftBound238725
def owner : Owner := ⟨.program ⟨257⟩, ⟨8359⟩⟩
def transferEvent : Nat := 238725
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 238723 .coefficient) (.predecessor 1 238724 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238723 .coefficient)
      LeftBound236647.bound (LeftBound236647.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events924.exact236648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236647.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238724 .coefficient)
      LeftBound19083.bound (LeftBound19083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound236647.bound LeftBound19083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236647.bound, LeftBound19083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound236647.actual selector witness) * (LeftBound19083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound238725

namespace LeftBound238730
def owner : Owner := ⟨.program ⟨257⟩, ⟨37070⟩⟩
def transferEvent : Nat := 238730
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 238728 .coefficient, .predecessor 1 238729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238728 .coefficient)
      LeftBound238725.bound (LeftBound238725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238725.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238729 .coefficient)
      LeftBound238720.bound (LeftBound238720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238725.bound, LeftBound238720.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238725.bound, LeftBound238720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238725.actual selector witness, LeftBound238720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238730

namespace LeftBound238734
def owner : Owner := ⟨.program ⟨257⟩, ⟨37071⟩⟩
def transferEvent : Nat := 238734
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 238732 .coefficient, .predecessor 1 238733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 238732 .coefficient)
      LeftBound238730.bound (LeftBound238730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events932.exact238731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound238730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound238730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 238733 .coefficient)
      LeftBound19075.bound (LeftBound19075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound238730.bound, LeftBound19075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound238730.bound, LeftBound19075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound238730.actual selector witness, LeftBound19075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound238734

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
