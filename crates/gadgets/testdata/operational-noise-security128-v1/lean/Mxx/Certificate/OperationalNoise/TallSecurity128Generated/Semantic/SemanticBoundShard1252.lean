import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1251

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound186694
def owner : Owner := ⟨.program ⟨257⟩, ⟨9569⟩⟩
def transferEvent : Nat := 186694
def frameStart : Nat := 186619
def rule : BoundRule := .scale (.predecessor 0 186692 .coefficient) (.value (.predecessor 1 186693 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186692 .coefficient)
      LeftAuthority186690.bound (LeftAuthority186690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186693 .coefficient)
      LeftAuthority186681.bound (LeftAuthority186681.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority186681.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority186690.bound LeftAuthority186681.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186690.bound, LeftAuthority186681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority186690.actual selector witness) * (LeftAuthority186681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound186694

namespace LeftBound186697
def owner : Owner := ⟨.program ⟨257⟩, ⟨7303⟩⟩
def transferEvent : Nat := 186697
def frameStart : Nat := 186619
def rule : BoundRule := .identity (.predecessor 0 186696 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186696 .coefficient)
      LeftAuthority186684.bound (LeftAuthority186684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186684.derived selector witness)

def rawBound : CoeffClass := LeftAuthority186684.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority186684.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound186697

namespace LeftBound186701
def owner : Owner := ⟨.program ⟨257⟩, ⟨9570⟩⟩
def transferEvent : Nat := 186701
def frameStart : Nat := 186619
def rule : BoundRule := .product (.predecessor 0 186699 .coefficient) (.predecessor 1 186700 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186699 .coefficient)
      LeftBound186697.bound (LeftBound186697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186700 .coefficient)
      LeftBound186694.bound (LeftBound186694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186694.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186697.bound LeftBound186694.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186697.bound, LeftBound186694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186697.actual selector witness) * (LeftBound186694.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186701

namespace LeftBound186706
def owner : Owner := ⟨.program ⟨257⟩, ⟨17141⟩⟩
def transferEvent : Nat := 186706
def frameStart : Nat := 186619
def rule : BoundRule := .sum [.predecessor 0 186704 .coefficient, .predecessor 1 186705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186704 .coefficient)
      LeftBound186701.bound (LeftBound186701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186705 .coefficient)
      LeftBound186678.bound (LeftBound186678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186701.bound, LeftBound186678.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186701.bound, LeftBound186678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186701.actual selector witness, LeftBound186678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186706

namespace LeftBound186710
def owner : Owner := ⟨.program ⟨257⟩, ⟨17395⟩⟩
def transferEvent : Nat := 186710
def frameStart : Nat := 186619
def rule : BoundRule := .product (.predecessor 0 186708 .coefficient) (.predecessor 1 186709 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186708 .coefficient)
      LeftBound186706.bound (LeftBound186706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186709 .coefficient)
      LeftAuthority186663.bound (LeftAuthority186663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186663.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186706.bound LeftAuthority186663.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186706.bound, LeftAuthority186663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186706.actual selector witness) * (LeftAuthority186663.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186710

namespace LeftBound186721
def owner : Owner := ⟨.program ⟨257⟩, ⟨15814⟩⟩
def transferEvent : Nat := 186721
def frameStart : Nat := 186619
def rule : BoundRule := .product (.predecessor 0 186719 .coefficient) (.predecessor 1 186720 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186719 .coefficient)
      LeftAuthority186674.bound (LeftAuthority186674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186720 .coefficient)
      LeftAuthority186717.bound (LeftAuthority186717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority186674.bound LeftAuthority186717.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186674.bound, LeftAuthority186717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority186674.actual selector witness) * (LeftAuthority186717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186721

namespace LeftBound186729
def owner : Owner := ⟨.program ⟨257⟩, ⟨15815⟩⟩
def transferEvent : Nat := 186729
def frameStart : Nat := 186619
def rule : BoundRule := .sum [.predecessor 0 186727 .coefficient, .predecessor 1 186728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186727 .coefficient)
      LeftAuthority186725.bound (LeftAuthority186725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186725.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186728 .coefficient)
      LeftBound186721.bound (LeftBound186721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority186725.bound, LeftBound186721.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186725.bound, LeftBound186721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority186725.actual selector witness, LeftBound186721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186729

namespace LeftBound186733
def owner : Owner := ⟨.program ⟨257⟩, ⟨17396⟩⟩
def transferEvent : Nat := 186733
def frameStart : Nat := 186619
def rule : BoundRule := .sum [.predecessor 0 186731 .coefficient, .predecessor 1 186732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186731 .coefficient)
      LeftBound186729.bound (LeftBound186729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186732 .coefficient)
      LeftBound186710.bound (LeftBound186710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186729.bound, LeftBound186710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186729.bound, LeftBound186710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186729.actual selector witness, LeftBound186710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186733

namespace LeftBound186746
def owner : Owner := ⟨.program ⟨257⟩, ⟨17394⟩⟩
def transferEvent : Nat := 186746
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186744 .coefficient, .predecessor 1 186745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186744 .coefficient)
      LeftBound186567.bound (LeftBound186567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186745 .coefficient)
      LeftBound186550.bound (LeftBound186550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events728.exact186557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186567.bound, LeftBound186550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186567.bound, LeftBound186550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186567.actual selector witness, LeftBound186550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186746

namespace LeftBound186749
def owner : Owner := ⟨.program ⟨257⟩, ⟨17394⟩⟩
def transferEvent : Nat := 186749
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186743 .summary, .result 186557 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186743 .summary)
      LeftBound186569.bound (LeftBound186569.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16322⟩⟩) (rawTerms := some (Proof.Events729.exact186743RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186557 .summary)
      LeftBound186552.bound (LeftBound186552.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17393⟩⟩) (rawTerms := some (Proof.Events728.exact186557RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186569.bound, LeftBound186552.bound]
def bound : CoeffClass := .finite ⟨2997816280693142192128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186569.bound, LeftBound186552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186569.actual selector witness, LeftBound186552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186749

namespace LeftBound186753
def owner : Owner := ⟨.program ⟨257⟩, ⟨17847⟩⟩
def transferEvent : Nat := 186753
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 186751 .coefficient) (.predecessor 1 186752 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186751 .coefficient)
      LeftBound186746.bound (LeftBound186746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186746.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186752 .coefficient)
      LeftAuthority186472.bound (LeftAuthority186472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events728.exact186473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186472.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186746.bound LeftAuthority186472.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186746.bound, LeftAuthority186472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186746.actual selector witness) * (LeftAuthority186472.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186753

namespace LeftBound186754
def owner : Owner := ⟨.program ⟨257⟩, ⟨17847⟩⟩
def transferEvent : Nat := 186754
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17845⟩⟩]⟩ [⟨.result 186473 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186473 .coefficient)
      LeftAuthority186472.bound (LeftAuthority186472.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17845⟩⟩) (rawTerms := some (Proof.Events728.exact186473RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186472.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority186472.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority186472.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound186754

namespace LeftBound186755
def owner : Owner := ⟨.program ⟨257⟩, ⟨17847⟩⟩
def transferEvent : Nat := 186755
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 186750 .summary) (.transfer 186754) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186750 .summary)
      LeftBound186749.bound (LeftBound186749.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17394⟩⟩) (rawTerms := some (Proof.Events729.exact186750RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 186754)
      LeftBound186754.bound (LeftBound186754.actual selector witness) := by
  exact .transfer (LeftBound186754.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound186749.bound LeftBound186754.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186749.bound, LeftBound186754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound186749.actual selector witness) * (LeftBound186754.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186755

namespace LeftBound186766
def owner : Owner := ⟨.program ⟨257⟩, ⟨16658⟩⟩
def transferEvent : Nat := 186766
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 186764 .coefficient) (.value (.predecessor 1 186765 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186764 .coefficient)
      LeftAuthority186762.bound (LeftAuthority186762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186765 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority186762.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186762.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority186762.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound186766

namespace LeftBound186770
def owner : Owner := ⟨.program ⟨257⟩, ⟨16659⟩⟩
def transferEvent : Nat := 186770
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 186768 .coefficient) (.predecessor 1 186769 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186768 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186769 .coefficient)
      LeftBound186766.bound (LeftBound186766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events729.exact186767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound186766.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound186766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound186766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186770

namespace LeftBound186771
def owner : Owner := ⟨.program ⟨257⟩, ⟨16659⟩⟩
def transferEvent : Nat := 186771
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨16656⟩⟩]⟩ [⟨.result 186763 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186763 .coefficient)
      LeftAuthority186762.bound (LeftAuthority186762.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨16656⟩⟩) (rawTerms := some (Proof.Events729.exact186763RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority186762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority186762.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority186762.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority186762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority186762.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound186771

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
