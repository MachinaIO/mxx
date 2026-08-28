import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9630
def owner : Owner := ⟨.program ⟨257⟩, ⟨66729⟩⟩
def transferEvent : Nat := 9630
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9628 .coefficient) (.predecessor 1 9629 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9628 .coefficient)
      LeftAuthority9626.bound (LeftAuthority9626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9629 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9626.bound LeftAuthority622.bound
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9626.bound, LeftAuthority622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9626.actual selector witness) * (LeftAuthority622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9630

namespace LeftBound9638
def owner : Owner := ⟨.program ⟨257⟩, ⟨63124⟩⟩
def transferEvent : Nat := 9638
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9636 .coefficient) (.predecessor 1 9637 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9636 .coefficient)
      LeftAuthority9634.bound (LeftAuthority9634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9637 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9634.bound LeftAuthority632.bound
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9634.bound, LeftAuthority632.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9634.actual selector witness) * (LeftAuthority632.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9638

namespace LeftBound9646
def owner : Owner := ⟨.program ⟨257⟩, ⟨60144⟩⟩
def transferEvent : Nat := 9646
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9644 .coefficient) (.predecessor 1 9645 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9644 .coefficient)
      LeftAuthority9642.bound (LeftAuthority9642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9645 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9642.bound LeftAuthority642.bound
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9642.bound, LeftAuthority642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9642.actual selector witness) * (LeftAuthority642.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9646

namespace LeftBound9654
def owner : Owner := ⟨.program ⟨257⟩, ⟨57164⟩⟩
def transferEvent : Nat := 9654
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9652 .coefficient) (.predecessor 1 9653 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9652 .coefficient)
      LeftAuthority9650.bound (LeftAuthority9650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9650.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9653 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9650.bound LeftAuthority652.bound
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9650.bound, LeftAuthority652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9650.actual selector witness) * (LeftAuthority652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9654

namespace LeftBound9662
def owner : Owner := ⟨.program ⟨257⟩, ⟨54184⟩⟩
def transferEvent : Nat := 9662
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9660 .coefficient) (.predecessor 1 9661 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9660 .coefficient)
      LeftAuthority9658.bound (LeftAuthority9658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9661 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9658.bound LeftAuthority662.bound
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9658.bound, LeftAuthority662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9658.actual selector witness) * (LeftAuthority662.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9662

namespace LeftBound9670
def owner : Owner := ⟨.program ⟨257⟩, ⟨51204⟩⟩
def transferEvent : Nat := 9670
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9668 .coefficient) (.predecessor 1 9669 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9668 .coefficient)
      LeftAuthority9666.bound (LeftAuthority9666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9666.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9669 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9666.bound LeftAuthority672.bound
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9666.bound, LeftAuthority672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9666.actual selector witness) * (LeftAuthority672.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9670

namespace LeftBound9678
def owner : Owner := ⟨.program ⟨257⟩, ⟨32140⟩⟩
def transferEvent : Nat := 9678
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9676 .coefficient) (.predecessor 1 9677 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9676 .coefficient)
      LeftAuthority9674.bound (LeftAuthority9674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9677 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9674.bound LeftAuthority682.bound
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9674.bound, LeftAuthority682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9674.actual selector witness) * (LeftAuthority682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9678

namespace LeftBound9686
def owner : Owner := ⟨.program ⟨257⟩, ⟨22120⟩⟩
def transferEvent : Nat := 9686
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9684 .coefficient) (.predecessor 1 9685 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9684 .coefficient)
      LeftAuthority9682.bound (LeftAuthority9682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9685 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9682.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9682.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9682.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9686

namespace LeftBound9694
def owner : Owner := ⟨.program ⟨257⟩, ⟨18900⟩⟩
def transferEvent : Nat := 9694
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9692 .coefficient) (.predecessor 1 9693 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9692 .coefficient)
      LeftAuthority9690.bound (LeftAuthority9690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9693 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9690.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9690.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9690.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9694

namespace LeftBound9702
def owner : Owner := ⟨.program ⟨257⟩, ⟨16063⟩⟩
def transferEvent : Nat := 9702
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9700 .coefficient) (.predecessor 1 9701 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9700 .coefficient)
      LeftAuthority9698.bound (LeftAuthority9698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9698.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9701 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9698.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9698.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority9698.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9702

namespace LeftBound9707
def owner : Owner := ⟨.program ⟨257⟩, ⟨16064⟩⟩
def transferEvent : Nat := 9707
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9705 .coefficient, .predecessor 1 9706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9705 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9706 .coefficient)
      LeftBound9702.bound (LeftBound9702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound9702.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound9702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound9702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9707

namespace LeftBound9711
def owner : Owner := ⟨.program ⟨257⟩, ⟨18901⟩⟩
def transferEvent : Nat := 9711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9709 .coefficient, .predecessor 1 9710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9709 .coefficient)
      LeftBound9707.bound (LeftBound9707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9710 .coefficient)
      LeftBound9694.bound (LeftBound9694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9707.bound, LeftBound9694.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9707.bound, LeftBound9694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9707.actual selector witness, LeftBound9694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9711

namespace LeftBound9715
def owner : Owner := ⟨.program ⟨257⟩, ⟨22121⟩⟩
def transferEvent : Nat := 9715
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9713 .coefficient, .predecessor 1 9714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9713 .coefficient)
      LeftBound9711.bound (LeftBound9711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9714 .coefficient)
      LeftBound9686.bound (LeftBound9686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9711.bound, LeftBound9686.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9711.bound, LeftBound9686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9711.actual selector witness, LeftBound9686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9715

namespace LeftBound9719
def owner : Owner := ⟨.program ⟨257⟩, ⟨32141⟩⟩
def transferEvent : Nat := 9719
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9717 .coefficient, .predecessor 1 9718 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9717 .coefficient)
      LeftBound9715.bound (LeftBound9715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9718 .coefficient)
      LeftBound9678.bound (LeftBound9678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9715.bound, LeftBound9678.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9715.bound, LeftBound9678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9715.actual selector witness, LeftBound9678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9719

namespace LeftBound9723
def owner : Owner := ⟨.program ⟨257⟩, ⟨51205⟩⟩
def transferEvent : Nat := 9723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9721 .coefficient, .predecessor 1 9722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9721 .coefficient)
      LeftBound9719.bound (LeftBound9719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9722 .coefficient)
      LeftBound9670.bound (LeftBound9670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9719.bound, LeftBound9670.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9719.bound, LeftBound9670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9719.actual selector witness, LeftBound9670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9723

namespace LeftBound9727
def owner : Owner := ⟨.program ⟨257⟩, ⟨54185⟩⟩
def transferEvent : Nat := 9727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9725 .coefficient, .predecessor 1 9726 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9725 .coefficient)
      LeftBound9723.bound (LeftBound9723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9726 .coefficient)
      LeftBound9662.bound (LeftBound9662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9723.bound, LeftBound9662.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9723.bound, LeftBound9662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9723.actual selector witness, LeftBound9662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9727

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
