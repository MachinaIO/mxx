import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1967

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound290676
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 290676
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290674 .coefficient, .predecessor 1 290675 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290674 .coefficient)
      LeftBound290672.bound (LeftBound290672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290675 .coefficient)
      LeftAuthority290634.bound (LeftAuthority290634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290634.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290672.bound, LeftAuthority290634.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290672.bound, LeftAuthority290634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290672.actual selector witness, LeftAuthority290634.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290676

namespace LeftBound290680
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 290680
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290678 .coefficient, .predecessor 1 290679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290678 .coefficient)
      LeftBound290676.bound (LeftBound290676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290679 .coefficient)
      LeftAuthority290631.bound (LeftAuthority290631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290676.bound, LeftAuthority290631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290676.bound, LeftAuthority290631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290676.actual selector witness, LeftAuthority290631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290680

namespace LeftBound290684
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 290684
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290682 .coefficient, .predecessor 1 290683 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290682 .coefficient)
      LeftBound290680.bound (LeftBound290680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290683 .coefficient)
      LeftAuthority290628.bound (LeftAuthority290628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290680.bound, LeftAuthority290628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290680.bound, LeftAuthority290628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290680.actual selector witness, LeftAuthority290628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290684

namespace LeftBound290688
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 290688
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290686 .coefficient, .predecessor 1 290687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290686 .coefficient)
      LeftBound290684.bound (LeftBound290684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290684.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290687 .coefficient)
      LeftAuthority290625.bound (LeftAuthority290625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290625.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290625.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290684.bound, LeftAuthority290625.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290684.bound, LeftAuthority290625.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290684.actual selector witness, LeftAuthority290625.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290688

namespace LeftBound290692
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 290692
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290690 .coefficient, .predecessor 1 290691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290690 .coefficient)
      LeftBound290688.bound (LeftBound290688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290691 .coefficient)
      LeftAuthority290622.bound (LeftAuthority290622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290622.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290688.bound, LeftAuthority290622.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290688.bound, LeftAuthority290622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290688.actual selector witness, LeftAuthority290622.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290692

namespace LeftBound290696
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 290696
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290694 .coefficient, .predecessor 1 290695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290694 .coefficient)
      LeftBound290692.bound (LeftBound290692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290695 .coefficient)
      LeftAuthority290619.bound (LeftAuthority290619.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290619.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290619.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290692.bound, LeftAuthority290619.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290692.bound, LeftAuthority290619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290692.actual selector witness, LeftAuthority290619.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290696

namespace LeftBound290700
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 290700
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290698 .coefficient, .predecessor 1 290699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290698 .coefficient)
      LeftBound290696.bound (LeftBound290696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290699 .coefficient)
      LeftAuthority290616.bound (LeftAuthority290616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290616.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290696.bound, LeftAuthority290616.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290696.bound, LeftAuthority290616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290696.actual selector witness, LeftAuthority290616.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290700

namespace LeftBound290704
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 290704
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290702 .coefficient, .predecessor 1 290703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290702 .coefficient)
      LeftBound290700.bound (LeftBound290700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290703 .coefficient)
      LeftAuthority290613.bound (LeftAuthority290613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290613.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290700.bound, LeftAuthority290613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290700.bound, LeftAuthority290613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290700.actual selector witness, LeftAuthority290613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290704

namespace LeftBound290708
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 290708
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290706 .coefficient, .predecessor 1 290707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290706 .coefficient)
      LeftBound290704.bound (LeftBound290704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290707 .coefficient)
      LeftAuthority290610.bound (LeftAuthority290610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290610.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290704.bound, LeftAuthority290610.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290704.bound, LeftAuthority290610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290704.actual selector witness, LeftAuthority290610.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290708

namespace LeftBound290712
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 290712
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290710 .coefficient, .predecessor 1 290711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290710 .coefficient)
      LeftBound290708.bound (LeftBound290708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290711 .coefficient)
      LeftAuthority290607.bound (LeftAuthority290607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290708.bound, LeftAuthority290607.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290708.bound, LeftAuthority290607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290708.actual selector witness, LeftAuthority290607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290712

namespace LeftBound290716
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 290716
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290714 .coefficient, .predecessor 1 290715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290714 .coefficient)
      LeftBound290712.bound (LeftBound290712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290715 .coefficient)
      LeftAuthority290604.bound (LeftAuthority290604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290604.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290604.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290712.bound, LeftAuthority290604.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290712.bound, LeftAuthority290604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290712.actual selector witness, LeftAuthority290604.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290716

namespace LeftBound290720
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 290720
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290718 .coefficient, .predecessor 1 290719 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290718 .coefficient)
      LeftBound290716.bound (LeftBound290716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290719 .coefficient)
      LeftAuthority290601.bound (LeftAuthority290601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290716.bound, LeftAuthority290601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290716.bound, LeftAuthority290601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290716.actual selector witness, LeftAuthority290601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290720

namespace LeftBound290724
def owner : Owner := ⟨.program ⟨257⟩, ⟨69066⟩⟩
def transferEvent : Nat := 290724
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290722 .coefficient, .predecessor 1 290723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290722 .coefficient)
      LeftBound290720.bound (LeftBound290720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290723 .coefficient)
      LeftBound290580.bound (LeftBound290580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290720.bound, LeftBound290580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290720.bound, LeftBound290580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290720.actual selector witness, LeftBound290580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290724

namespace LeftBound290728
def owner : Owner := ⟨.program ⟨257⟩, ⟨71049⟩⟩
def transferEvent : Nat := 290728
def frameStart : Nat := 290050
def rule : BoundRule := .product (.predecessor 0 290726 .coefficient) (.predecessor 1 290727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290726 .coefficient)
      LeftBound290724.bound (LeftBound290724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290727 .coefficient)
      LeftAuthority290565.bound (LeftAuthority290565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290565.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound290724.bound LeftAuthority290565.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290724.bound, LeftAuthority290565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound290724.actual selector witness) * (LeftAuthority290565.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound290728

namespace LeftBound290807
def owner : Owner := ⟨.program ⟨257⟩, ⟨67343⟩⟩
def transferEvent : Nat := 290807
def frameStart : Nat := 290050
def rule : BoundRule := .product (.predecessor 0 290805 .coefficient) (.predecessor 1 290806 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290805 .coefficient)
      LeftAuthority290576.bound (LeftAuthority290576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290806 .coefficient)
      LeftAuthority290803.bound (LeftAuthority290803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290803.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290803.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority290576.bound LeftAuthority290803.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority290576.bound, LeftAuthority290803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority290576.actual selector witness) * (LeftAuthority290803.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound290807

namespace LeftBound290815
def owner : Owner := ⟨.program ⟨257⟩, ⟨67349⟩⟩
def transferEvent : Nat := 290815
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290813 .coefficient, .predecessor 1 290814 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290813 .coefficient)
      LeftAuthority290811.bound (LeftAuthority290811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290814 .coefficient)
      LeftBound290807.bound (LeftBound290807.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290807.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority290811.bound, LeftBound290807.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority290811.bound, LeftBound290807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority290811.actual selector witness, LeftBound290807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290815

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
