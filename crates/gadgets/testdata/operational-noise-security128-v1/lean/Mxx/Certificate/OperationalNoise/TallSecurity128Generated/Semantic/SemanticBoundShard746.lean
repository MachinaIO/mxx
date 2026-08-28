import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard731
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard735
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard739
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard742
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard745

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound113762
def owner : Owner := ⟨.program ⟨257⟩, ⟨17211⟩⟩
def transferEvent : Nat := 113762
def frameStart : Nat := 113703
def rule : BoundRule := .identity (.predecessor 0 113761 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113761 .coefficient)
      LeftBound113759.bound (LeftBound113759.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound113759.derived selector witness)

def rawBound : CoeffClass := LeftBound113759.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound113759.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound113762

namespace LeftBound113768
def owner : Owner := ⟨.program ⟨257⟩, ⟨17212⟩⟩
def transferEvent : Nat := 113768
def frameStart : Nat := 113703
def rule : BoundRule := .product (.predecessor 0 113766 .coefficient) (.predecessor 1 113767 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113766 .coefficient)
      LeftAuthority113764.bound (LeftAuthority113764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113767 .coefficient)
      LeftBound113762.bound (LeftBound113762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113762.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority113764.bound LeftBound113762.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113764.bound, LeftBound113762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority113764.actual selector witness) * (LeftBound113762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113768

namespace LeftBound113776
def owner : Owner := ⟨.program ⟨257⟩, ⟨17213⟩⟩
def transferEvent : Nat := 113776
def frameStart : Nat := 113703
def rule : BoundRule := .sum [.predecessor 0 113774 .coefficient, .predecessor 1 113775 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113774 .coefficient)
      LeftAuthority113772.bound (LeftAuthority113772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113772.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113775 .coefficient)
      LeftBound113768.bound (LeftBound113768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority113772.bound, LeftBound113768.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113772.bound, LeftBound113768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority113772.actual selector witness, LeftBound113768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113776

namespace LeftBound113780
def owner : Owner := ⟨.program ⟨257⟩, ⟨17790⟩⟩
def transferEvent : Nat := 113780
def frameStart : Nat := 113703
def rule : BoundRule := .product (.predecessor 0 113778 .coefficient) (.predecessor 1 113779 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113778 .coefficient)
      LeftBound113776.bound (LeftBound113776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113777RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113776.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113779 .coefficient)
      LeftAuthority113753.bound (LeftAuthority113753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound113776.bound LeftAuthority113753.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113776.bound, LeftAuthority113753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound113776.actual selector witness) * (LeftAuthority113753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113780

namespace LeftBound113791
def owner : Owner := ⟨.program ⟨257⟩, ⟨16052⟩⟩
def transferEvent : Nat := 113791
def frameStart : Nat := 113703
def rule : BoundRule := .product (.predecessor 0 113789 .coefficient) (.predecessor 1 113790 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113789 .coefficient)
      LeftAuthority113764.bound (LeftAuthority113764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113790 .coefficient)
      LeftAuthority113787.bound (LeftAuthority113787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113787.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113787.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority113764.bound LeftAuthority113787.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113764.bound, LeftAuthority113787.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority113764.actual selector witness) * (LeftAuthority113787.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound113791

namespace LeftBound113799
def owner : Owner := ⟨.program ⟨257⟩, ⟨16053⟩⟩
def transferEvent : Nat := 113799
def frameStart : Nat := 113703
def rule : BoundRule := .sum [.predecessor 0 113797 .coefficient, .predecessor 1 113798 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113797 .coefficient)
      LeftAuthority113795.bound (LeftAuthority113795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority113795.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority113795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113798 .coefficient)
      LeftBound113791.bound (LeftBound113791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority113795.bound, LeftBound113791.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority113795.bound, LeftBound113791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority113795.actual selector witness, LeftBound113791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113799

namespace LeftBound113803
def owner : Owner := ⟨.program ⟨257⟩, ⟨17793⟩⟩
def transferEvent : Nat := 113803
def frameStart : Nat := 113703
def rule : BoundRule := .sum [.predecessor 0 113801 .coefficient, .predecessor 1 113802 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113801 .coefficient)
      LeftBound113799.bound (LeftBound113799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113802 .coefficient)
      LeftBound113780.bound (LeftBound113780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113785RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113780.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113799.bound, LeftBound113780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113799.bound, LeftBound113780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113799.actual selector witness, LeftBound113780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113803

namespace LeftBound113816
def owner : Owner := ⟨.program ⟨257⟩, ⟨17792⟩⟩
def transferEvent : Nat := 113816
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113814 .coefficient, .predecessor 1 113815 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113814 .coefficient)
      LeftBound113645.bound (LeftBound113645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113815 .coefficient)
      LeftBound113628.bound (LeftBound113628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events443.exact113635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113645.bound, LeftBound113628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113645.bound, LeftBound113628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113645.actual selector witness, LeftBound113628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113816

namespace LeftBound113819
def owner : Owner := ⟨.program ⟨257⟩, ⟨17792⟩⟩
def transferEvent : Nat := 113819
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113813 .summary, .result 113635 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113813 .summary)
      LeftBound113647.bound (LeftBound113647.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16619⟩⟩) (rawTerms := some (Proof.Events444.exact113813RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113647.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113635 .summary)
      LeftBound113630.bound (LeftBound113630.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17791⟩⟩) (rawTerms := some (Proof.Events443.exact113635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113647.bound, LeftBound113630.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113647.bound, LeftBound113630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113647.actual selector witness, LeftBound113630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113819

namespace LeftBound113823
def owner : Owner := ⟨.program ⟨257⟩, ⟨20687⟩⟩
def transferEvent : Nat := 113823
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113821 .coefficient, .predecessor 1 113822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113821 .coefficient)
      LeftBound113816.bound (LeftBound113816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113816.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113822 .coefficient)
      LeftBound113334.bound (LeftBound113334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events442.exact113338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113816.bound, LeftBound113334.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113816.bound, LeftBound113334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113816.actual selector witness, LeftBound113334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113823

namespace LeftBound113824
def owner : Owner := ⟨.program ⟨257⟩, ⟨20687⟩⟩
def transferEvent : Nat := 113824
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113820 .summary, .result 113338 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113820 .summary)
      LeftBound113819.bound (LeftBound113819.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17792⟩⟩) (rawTerms := some (Proof.Events444.exact113820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113338 .summary)
      LeftBound113337.bound (LeftBound113337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20686⟩⟩) (rawTerms := some (Proof.Events442.exact113338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113337.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113819.bound, LeftBound113337.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113819.bound, LeftBound113337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113819.actual selector witness, LeftBound113337.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113824

namespace LeftBound113828
def owner : Owner := ⟨.program ⟨257⟩, ⟨23907⟩⟩
def transferEvent : Nat := 113828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113826 .coefficient, .predecessor 1 113827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113826 .coefficient)
      LeftBound113823.bound (LeftBound113823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113827 .coefficient)
      LeftBound112852.bound (LeftBound112852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events440.exact112856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113823.bound, LeftBound112852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113823.bound, LeftBound112852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113823.actual selector witness, LeftBound112852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113828

namespace LeftBound113829
def owner : Owner := ⟨.program ⟨257⟩, ⟨23907⟩⟩
def transferEvent : Nat := 113829
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113825 .summary, .result 112856 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113825 .summary)
      LeftBound113824.bound (LeftBound113824.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20687⟩⟩) (rawTerms := some (Proof.Events444.exact113825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112856 .summary)
      LeftBound112855.bound (LeftBound112855.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23906⟩⟩) (rawTerms := some (Proof.Events440.exact112856RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound112855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113824.bound, LeftBound112855.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113824.bound, LeftBound112855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113824.actual selector witness, LeftBound112855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113829

namespace LeftBound113833
def owner : Owner := ⟨.program ⟨257⟩, ⟨33927⟩⟩
def transferEvent : Nat := 113833
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113831 .coefficient, .predecessor 1 113832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113831 .coefficient)
      LeftBound113828.bound (LeftBound113828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113832 .coefficient)
      LeftBound112370.bound (LeftBound112370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113828.bound, LeftBound112370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113828.bound, LeftBound112370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113828.actual selector witness, LeftBound112370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113833

namespace LeftBound113834
def owner : Owner := ⟨.program ⟨257⟩, ⟨33927⟩⟩
def transferEvent : Nat := 113834
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113830 .summary, .result 112374 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113830 .summary)
      LeftBound113829.bound (LeftBound113829.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23907⟩⟩) (rawTerms := some (Proof.Events444.exact113830RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112374 .summary)
      LeftBound112373.bound (LeftBound112373.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33926⟩⟩) (rawTerms := some (Proof.Events438.exact112374RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound112373.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113829.bound, LeftBound112373.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113829.bound, LeftBound112373.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113829.actual selector witness, LeftBound112373.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113834

namespace LeftBound113838
def owner : Owner := ⟨.program ⟨257⟩, ⟨52987⟩⟩
def transferEvent : Nat := 113838
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113836 .coefficient, .predecessor 1 113837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113836 .coefficient)
      LeftBound113833.bound (LeftBound113833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113837 .coefficient)
      LeftBound111888.bound (LeftBound111888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact111892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111888.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113833.bound, LeftBound111888.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113833.bound, LeftBound111888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113833.actual selector witness, LeftBound111888.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113838

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
