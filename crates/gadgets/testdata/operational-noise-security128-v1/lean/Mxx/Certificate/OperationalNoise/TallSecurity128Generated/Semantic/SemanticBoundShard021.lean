import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard020

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6723
def owner : Owner := ⟨.program ⟨257⟩, ⟨21950⟩⟩
def transferEvent : Nat := 6723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6721 .coefficient, .predecessor 1 6722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6721 .coefficient)
      LeftBound6719.bound (LeftBound6719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6722 .coefficient)
      LeftBound6694.bound (LeftBound6694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6719.bound, LeftBound6694.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6719.bound, LeftBound6694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6719.actual selector witness, LeftBound6694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6723

namespace LeftBound6727
def owner : Owner := ⟨.program ⟨257⟩, ⟨31970⟩⟩
def transferEvent : Nat := 6727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6725 .coefficient, .predecessor 1 6726 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6725 .coefficient)
      LeftBound6723.bound (LeftBound6723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6726 .coefficient)
      LeftBound6686.bound (LeftBound6686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6723.bound, LeftBound6686.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6723.bound, LeftBound6686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6723.actual selector witness, LeftBound6686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6727

namespace LeftBound6731
def owner : Owner := ⟨.program ⟨257⟩, ⟨51034⟩⟩
def transferEvent : Nat := 6731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6729 .coefficient, .predecessor 1 6730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6729 .coefficient)
      LeftBound6727.bound (LeftBound6727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6730 .coefficient)
      LeftBound6678.bound (LeftBound6678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6727.bound, LeftBound6678.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6727.bound, LeftBound6678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6727.actual selector witness, LeftBound6678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6731

namespace LeftBound6735
def owner : Owner := ⟨.program ⟨257⟩, ⟨54014⟩⟩
def transferEvent : Nat := 6735
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6733 .coefficient, .predecessor 1 6734 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6733 .coefficient)
      LeftBound6731.bound (LeftBound6731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6734 .coefficient)
      LeftBound6670.bound (LeftBound6670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6731.bound, LeftBound6670.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6731.bound, LeftBound6670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6731.actual selector witness, LeftBound6670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6735

namespace LeftBound6739
def owner : Owner := ⟨.program ⟨257⟩, ⟨56994⟩⟩
def transferEvent : Nat := 6739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6737 .coefficient, .predecessor 1 6738 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6737 .coefficient)
      LeftBound6735.bound (LeftBound6735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6738 .coefficient)
      LeftBound6662.bound (LeftBound6662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6662.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6662.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6735.bound, LeftBound6662.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6735.bound, LeftBound6662.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6735.actual selector witness, LeftBound6662.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6739

namespace LeftBound6743
def owner : Owner := ⟨.program ⟨257⟩, ⟨59974⟩⟩
def transferEvent : Nat := 6743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6741 .coefficient, .predecessor 1 6742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6741 .coefficient)
      LeftBound6739.bound (LeftBound6739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6742 .coefficient)
      LeftBound6654.bound (LeftBound6654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6654.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6739.bound, LeftBound6654.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6739.bound, LeftBound6654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6739.actual selector witness, LeftBound6654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6743

namespace LeftBound6747
def owner : Owner := ⟨.program ⟨257⟩, ⟨62954⟩⟩
def transferEvent : Nat := 6747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6745 .coefficient, .predecessor 1 6746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6745 .coefficient)
      LeftBound6743.bound (LeftBound6743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6746 .coefficient)
      LeftBound6646.bound (LeftBound6646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6648RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6646.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6743.bound, LeftBound6646.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6743.bound, LeftBound6646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6743.actual selector witness, LeftBound6646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6747

namespace LeftBound6751
def owner : Owner := ⟨.program ⟨257⟩, ⟨66100⟩⟩
def transferEvent : Nat := 6751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6749 .coefficient, .predecessor 1 6750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6749 .coefficient)
      LeftBound6747.bound (LeftBound6747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6750 .coefficient)
      LeftBound6638.bound (LeftBound6638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6638.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6747.bound, LeftBound6638.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6747.bound, LeftBound6638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6747.actual selector witness, LeftBound6638.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6751

namespace LeftBound6755
def owner : Owner := ⟨.program ⟨257⟩, ⟨66101⟩⟩
def transferEvent : Nat := 6755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6753 .coefficient, .predecessor 1 6754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6753 .coefficient)
      LeftBound6751.bound (LeftBound6751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6754 .coefficient)
      LeftBound6630.bound (LeftBound6630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6630.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6751.bound, LeftBound6630.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6751.bound, LeftBound6630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6751.actual selector witness, LeftBound6630.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6755

namespace LeftBound6759
def owner : Owner := ⟨.program ⟨257⟩, ⟨66102⟩⟩
def transferEvent : Nat := 6759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6757 .coefficient, .predecessor 1 6758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6757 .coefficient)
      LeftBound6755.bound (LeftBound6755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6758 .coefficient)
      LeftBound6622.bound (LeftBound6622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6622.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6755.bound, LeftBound6622.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6755.bound, LeftBound6622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6755.actual selector witness, LeftBound6622.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6759

namespace LeftBound6763
def owner : Owner := ⟨.program ⟨257⟩, ⟨66103⟩⟩
def transferEvent : Nat := 6763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6761 .coefficient, .predecessor 1 6762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6761 .coefficient)
      LeftBound6759.bound (LeftBound6759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6762 .coefficient)
      LeftBound6614.bound (LeftBound6614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6759.bound, LeftBound6614.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6759.bound, LeftBound6614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6759.actual selector witness, LeftBound6614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6763

namespace LeftBound6767
def owner : Owner := ⟨.program ⟨257⟩, ⟨66104⟩⟩
def transferEvent : Nat := 6767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6765 .coefficient, .predecessor 1 6766 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6765 .coefficient)
      LeftBound6763.bound (LeftBound6763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6766 .coefficient)
      LeftBound6606.bound (LeftBound6606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6763.bound, LeftBound6606.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6763.bound, LeftBound6606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6763.actual selector witness, LeftBound6606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6767

namespace LeftBound6771
def owner : Owner := ⟨.program ⟨257⟩, ⟨66105⟩⟩
def transferEvent : Nat := 6771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6769 .coefficient, .predecessor 1 6770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6769 .coefficient)
      LeftBound6767.bound (LeftBound6767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6770 .coefficient)
      LeftBound6598.bound (LeftBound6598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6767.bound, LeftBound6598.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6767.bound, LeftBound6598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6767.actual selector witness, LeftBound6598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6771

namespace LeftBound6775
def owner : Owner := ⟨.program ⟨257⟩, ⟨66106⟩⟩
def transferEvent : Nat := 6775
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6773 .coefficient, .predecessor 1 6774 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6773 .coefficient)
      LeftBound6771.bound (LeftBound6771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6774 .coefficient)
      LeftBound6590.bound (LeftBound6590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6771.bound, LeftBound6590.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6771.bound, LeftBound6590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6771.actual selector witness, LeftBound6590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6775

namespace LeftBound6779
def owner : Owner := ⟨.program ⟨257⟩, ⟨66107⟩⟩
def transferEvent : Nat := 6779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6777 .coefficient, .predecessor 1 6778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6777 .coefficient)
      LeftBound6775.bound (LeftBound6775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6778 .coefficient)
      LeftBound6582.bound (LeftBound6582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6775.bound, LeftBound6582.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6775.bound, LeftBound6582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6775.actual selector witness, LeftBound6582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6779

namespace LeftBound6783
def owner : Owner := ⟨.program ⟨257⟩, ⟨66108⟩⟩
def transferEvent : Nat := 6783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6781 .coefficient, .predecessor 1 6782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 6781 .coefficient)
      LeftBound6779.bound (LeftBound6779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events026.exact6780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 6782 .coefficient)
      LeftBound6574.bound (LeftBound6574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6779.bound, LeftBound6574.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6779.bound, LeftBound6574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound6779.actual selector witness, LeftBound6574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6783

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
