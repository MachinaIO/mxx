import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard850

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound129676
def owner : Owner := ⟨.program ⟨257⟩, ⟨63006⟩⟩
def transferEvent : Nat := 129676
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129674 .coefficient, .predecessor 1 129675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129674 .coefficient)
      LeftBound129672.bound (LeftBound129672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129675 .coefficient)
      LeftAuthority129460.bound (LeftAuthority129460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129460.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129672.bound, LeftAuthority129460.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129672.bound, LeftAuthority129460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129672.actual selector witness, LeftAuthority129460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129676

namespace LeftBound129680
def owner : Owner := ⟨.program ⟨257⟩, ⟨66322⟩⟩
def transferEvent : Nat := 129680
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129678 .coefficient, .predecessor 1 129679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129678 .coefficient)
      LeftBound129676.bound (LeftBound129676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129679 .coefficient)
      LeftAuthority129437.bound (LeftAuthority129437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129676.bound, LeftAuthority129437.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129676.bound, LeftAuthority129437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129676.actual selector witness, LeftAuthority129437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129680

namespace LeftBound129684
def owner : Owner := ⟨.program ⟨257⟩, ⟨66323⟩⟩
def transferEvent : Nat := 129684
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129682 .coefficient, .predecessor 1 129683 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129682 .coefficient)
      LeftBound129680.bound (LeftBound129680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129683 .coefficient)
      LeftAuthority129414.bound (LeftAuthority129414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129680.bound, LeftAuthority129414.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129680.bound, LeftAuthority129414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129680.actual selector witness, LeftAuthority129414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129684

namespace LeftBound129688
def owner : Owner := ⟨.program ⟨257⟩, ⟨66324⟩⟩
def transferEvent : Nat := 129688
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129686 .coefficient, .predecessor 1 129687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129686 .coefficient)
      LeftBound129684.bound (LeftBound129684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129687 .coefficient)
      LeftAuthority129391.bound (LeftAuthority129391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129391.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129684.bound, LeftAuthority129391.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129684.bound, LeftAuthority129391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129684.actual selector witness, LeftAuthority129391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129688

namespace LeftBound129692
def owner : Owner := ⟨.program ⟨257⟩, ⟨66325⟩⟩
def transferEvent : Nat := 129692
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129690 .coefficient, .predecessor 1 129691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129690 .coefficient)
      LeftBound129688.bound (LeftBound129688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129691 .coefficient)
      LeftAuthority129368.bound (LeftAuthority129368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129368.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129688.bound, LeftAuthority129368.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129688.bound, LeftAuthority129368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129688.actual selector witness, LeftAuthority129368.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129692

namespace LeftBound129696
def owner : Owner := ⟨.program ⟨257⟩, ⟨66326⟩⟩
def transferEvent : Nat := 129696
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129694 .coefficient, .predecessor 1 129695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129694 .coefficient)
      LeftBound129692.bound (LeftBound129692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129695 .coefficient)
      LeftAuthority129345.bound (LeftAuthority129345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129345.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129692.bound, LeftAuthority129345.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129692.bound, LeftAuthority129345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129692.actual selector witness, LeftAuthority129345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129696

namespace LeftBound129700
def owner : Owner := ⟨.program ⟨257⟩, ⟨66327⟩⟩
def transferEvent : Nat := 129700
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129698 .coefficient, .predecessor 1 129699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129698 .coefficient)
      LeftBound129696.bound (LeftBound129696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129699 .coefficient)
      LeftAuthority129322.bound (LeftAuthority129322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129322.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129696.bound, LeftAuthority129322.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129696.bound, LeftAuthority129322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129696.actual selector witness, LeftAuthority129322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129700

namespace LeftBound129704
def owner : Owner := ⟨.program ⟨257⟩, ⟨66328⟩⟩
def transferEvent : Nat := 129704
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129702 .coefficient, .predecessor 1 129703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129702 .coefficient)
      LeftBound129700.bound (LeftBound129700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129703 .coefficient)
      LeftAuthority129299.bound (LeftAuthority129299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events505.exact129300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129299.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129700.bound, LeftAuthority129299.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129700.bound, LeftAuthority129299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129700.actual selector witness, LeftAuthority129299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129704

namespace LeftBound129708
def owner : Owner := ⟨.program ⟨257⟩, ⟨66329⟩⟩
def transferEvent : Nat := 129708
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129706 .coefficient, .predecessor 1 129707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129706 .coefficient)
      LeftBound129704.bound (LeftBound129704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129707 .coefficient)
      LeftAuthority129276.bound (LeftAuthority129276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events504.exact129277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129704.bound, LeftAuthority129276.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129704.bound, LeftAuthority129276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129704.actual selector witness, LeftAuthority129276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129708

namespace LeftBound129712
def owner : Owner := ⟨.program ⟨257⟩, ⟨66330⟩⟩
def transferEvent : Nat := 129712
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129710 .coefficient, .predecessor 1 129711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129710 .coefficient)
      LeftBound129708.bound (LeftBound129708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129711 .coefficient)
      LeftAuthority129253.bound (LeftAuthority129253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events504.exact129254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129253.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129708.bound, LeftAuthority129253.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129708.bound, LeftAuthority129253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129708.actual selector witness, LeftAuthority129253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129712

namespace LeftBound129715
def owner : Owner := ⟨.program ⟨257⟩, ⟨66331⟩⟩
def transferEvent : Nat := 129715
def frameStart : Nat := 129211
def rule : BoundRule := .identity (.predecessor 0 129714 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129714 .coefficient)
      LeftBound129712.bound (LeftBound129712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129712.derived selector witness)

def rawBound : CoeffClass := LeftBound129712.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound129712.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound129715

namespace LeftBound129732
def owner : Owner := ⟨.program ⟨257⟩, ⟨69071⟩⟩
def transferEvent : Nat := 129732
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129730 .coefficient, .predecessor 1 129731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129730 .coefficient)
      LeftBound129715.bound (LeftBound129715.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound129715.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129731 .coefficient)
      LeftAuthority129728.bound (LeftAuthority129728.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority129728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129715.bound, LeftAuthority129728.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129715.bound, LeftAuthority129728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129715.actual selector witness, LeftAuthority129728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129732

namespace LeftBound129735
def owner : Owner := ⟨.program ⟨257⟩, ⟨69072⟩⟩
def transferEvent : Nat := 129735
def frameStart : Nat := 129211
def rule : BoundRule := .identity (.predecessor 0 129734 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129734 .coefficient)
      LeftBound129732.bound (LeftBound129732.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound129732.derived selector witness)

def rawBound : CoeffClass := LeftBound129732.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound129732.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound129735

namespace LeftBound129741
def owner : Owner := ⟨.program ⟨257⟩, ⟨69073⟩⟩
def transferEvent : Nat := 129741
def frameStart : Nat := 129211
def rule : BoundRule := .product (.predecessor 0 129739 .coefficient) (.predecessor 1 129740 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129739 .coefficient)
      LeftAuthority129737.bound (LeftAuthority129737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129740 .coefficient)
      LeftBound129735.bound (LeftBound129735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129735.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority129737.bound LeftBound129735.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority129737.bound, LeftBound129735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority129737.actual selector witness) * (LeftBound129735.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound129741

namespace LeftBound129817
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 129817
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129815 .coefficient, .predecessor 1 129816 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129815 .coefficient)
      LeftAuthority129813.bound (LeftAuthority129813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129813.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129816 .coefficient)
      LeftAuthority129810.bound (LeftAuthority129810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129810.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority129813.bound, LeftAuthority129810.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority129813.bound, LeftAuthority129810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority129813.actual selector witness, LeftAuthority129810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129817

namespace LeftBound129821
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 129821
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129819 .coefficient, .predecessor 1 129820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129819 .coefficient)
      LeftBound129817.bound (LeftBound129817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129820 .coefficient)
      LeftAuthority129807.bound (LeftAuthority129807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129808RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129807.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129817.bound, LeftAuthority129807.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129817.bound, LeftAuthority129807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129817.actual selector witness, LeftAuthority129807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129821

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
