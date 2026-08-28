import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound771
def owner : Owner := ⟨.program ⟨257⟩, ⟨65983⟩⟩
def transferEvent : Nat := 771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 769 .coefficient, .predecessor 1 770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 769 .coefficient)
      LeftBound767.bound (LeftBound767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 770 .coefficient)
      LeftBound619.bound (LeftBound619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound619.bound, RecordedBoundRefines] <;> decide)
      (LeftBound619.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound767.bound, LeftBound619.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound767.bound, LeftBound619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound767.actual selector witness, LeftBound619.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound771

namespace LeftBound775
def owner : Owner := ⟨.program ⟨257⟩, ⟨65984⟩⟩
def transferEvent : Nat := 775
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 773 .coefficient, .predecessor 1 774 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 773 .coefficient)
      LeftBound771.bound (LeftBound771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 774 .coefficient)
      LeftBound609.bound (LeftBound609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound609.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound771.bound, LeftBound609.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound771.bound, LeftBound609.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound771.actual selector witness, LeftBound609.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound775

namespace LeftBound779
def owner : Owner := ⟨.program ⟨257⟩, ⟨65985⟩⟩
def transferEvent : Nat := 779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 777 .coefficient, .predecessor 1 778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 777 .coefficient)
      LeftBound775.bound (LeftBound775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 778 .coefficient)
      LeftBound599.bound (LeftBound599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound599.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound775.bound, LeftBound599.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound775.bound, LeftBound599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound775.actual selector witness, LeftBound599.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound779

namespace LeftBound783
def owner : Owner := ⟨.program ⟨257⟩, ⟨65986⟩⟩
def transferEvent : Nat := 783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 781 .coefficient, .predecessor 1 782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 781 .coefficient)
      LeftBound779.bound (LeftBound779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 782 .coefficient)
      LeftBound589.bound (LeftBound589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound779.bound, LeftBound589.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound779.bound, LeftBound589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound779.actual selector witness, LeftBound589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound783

namespace LeftBound787
def owner : Owner := ⟨.program ⟨257⟩, ⟨65987⟩⟩
def transferEvent : Nat := 787
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 785 .coefficient, .predecessor 1 786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 785 .coefficient)
      LeftBound783.bound (LeftBound783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 786 .coefficient)
      LeftBound579.bound (LeftBound579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound579.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound783.bound, LeftBound579.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound783.bound, LeftBound579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound783.actual selector witness, LeftBound579.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound787

namespace LeftBound791
def owner : Owner := ⟨.program ⟨257⟩, ⟨65988⟩⟩
def transferEvent : Nat := 791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 789 .coefficient, .predecessor 1 790 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 789 .coefficient)
      LeftBound787.bound (LeftBound787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 790 .coefficient)
      LeftBound569.bound (LeftBound569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound569.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound787.bound, LeftBound569.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound787.bound, LeftBound569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound787.actual selector witness, LeftBound569.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound791

namespace LeftBound795
def owner : Owner := ⟨.program ⟨257⟩, ⟨65989⟩⟩
def transferEvent : Nat := 795
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 793 .coefficient, .predecessor 1 794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 793 .coefficient)
      LeftBound791.bound (LeftBound791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 794 .coefficient)
      LeftBound559.bound (LeftBound559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound791.bound, LeftBound559.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound791.bound, LeftBound559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound791.actual selector witness, LeftBound559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound795

namespace LeftBound799
def owner : Owner := ⟨.program ⟨257⟩, ⟨65990⟩⟩
def transferEvent : Nat := 799
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 797 .coefficient, .predecessor 1 798 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 797 .coefficient)
      LeftBound795.bound (LeftBound795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 798 .coefficient)
      LeftBound549.bound (LeftBound549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound795.bound, LeftBound549.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound795.bound, LeftBound549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound795.actual selector witness, LeftBound549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound799

namespace LeftBound803
def owner : Owner := ⟨.program ⟨257⟩, ⟨67296⟩⟩
def transferEvent : Nat := 803
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 801 .coefficient, .predecessor 1 802 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 801 .coefficient)
      LeftBound799.bound (LeftBound799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 802 .coefficient)
      LeftBound539.bound (LeftBound539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound799.bound, LeftBound539.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound799.bound, LeftBound539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound799.actual selector witness, LeftBound539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound803

namespace LeftBound807
def owner : Owner := ⟨.program ⟨257⟩, ⟨67297⟩⟩
def transferEvent : Nat := 807
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 805 .coefficient) (.predecessor 1 806 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 805 .coefficient)
      LeftBound803.bound (LeftBound803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 806 .coefficient)
      LeftAuthority33.bound (LeftAuthority33.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact34RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority33.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority33.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound803.bound LeftAuthority33.bound
def bound : CoeffClass := .finite ⟨129487453721564830525623756109199326648648454370020576915769440214653297240146884935985507454637473562755301973985894189786961452056995229483380260772801058312692018843752923511499260178276307468288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound803.bound, LeftAuthority33.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound803.actual selector witness) * (LeftAuthority33.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound807

namespace LeftBound1330
def owner : Owner := ⟨.program ⟨257⟩, ⟨67648⟩⟩
def transferEvent : Nat := 1330
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1328 .coefficient) (.predecessor 1 1329 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1328 .coefficient)
      LeftAuthority1326.bound (LeftAuthority1326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1329 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1326.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1326.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1326.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1330

namespace LeftBound1338
def owner : Owner := ⟨.program ⟨257⟩, ⟨48477⟩⟩
def transferEvent : Nat := 1338
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1336 .coefficient) (.predecessor 1 1337 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1336 .coefficient)
      LeftAuthority1334.bound (LeftAuthority1334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1337 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1334.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1334.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1334.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1338

namespace LeftBound1346
def owner : Owner := ⟨.program ⟨257⟩, ⟨45797⟩⟩
def transferEvent : Nat := 1346
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1344 .coefficient) (.predecessor 1 1345 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1344 .coefficient)
      LeftAuthority1342.bound (LeftAuthority1342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1345 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1342.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1342.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1342.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1346

namespace LeftBound1354
def owner : Owner := ⟨.program ⟨257⟩, ⟨43120⟩⟩
def transferEvent : Nat := 1354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1352 .coefficient) (.predecessor 1 1353 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1352 .coefficient)
      LeftAuthority1350.bound (LeftAuthority1350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1353 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1350.bound LeftAuthority562.bound
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1350.bound, LeftAuthority562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1350.actual selector witness) * (LeftAuthority562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1354

namespace LeftBound1362
def owner : Owner := ⟨.program ⟨257⟩, ⟨40440⟩⟩
def transferEvent : Nat := 1362
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1360 .coefficient) (.predecessor 1 1361 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1360 .coefficient)
      LeftAuthority1358.bound (LeftAuthority1358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1358.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1361 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1358.bound LeftAuthority572.bound
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1358.bound, LeftAuthority572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1358.actual selector witness) * (LeftAuthority572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1362

namespace LeftBound1370
def owner : Owner := ⟨.program ⟨257⟩, ⟨37757⟩⟩
def transferEvent : Nat := 1370
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 1368 .coefficient) (.predecessor 1 1369 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 1368 .coefficient)
      LeftAuthority1366.bound (LeftAuthority1366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events005.exact1367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1366.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 1369 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1366.bound LeftAuthority582.bound
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1366.bound, LeftAuthority582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority1366.actual selector witness) * (LeftAuthority582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound1370

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
